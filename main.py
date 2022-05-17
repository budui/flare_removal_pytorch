import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from datetime import datetime
import fire
import torch
import torch.nn.init as init
import torchvision
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import networks
import synthesis
import utils
from data import UnpairedDataset, PairedDataset
from losses.focal_frequency_loss import FocalFrequencyLoss
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from test import evaluate_fn

# Small number added to near-zero quantities to avoid numerical instability.
_EPS = 1e-7


def build_criterion(config, device):
    loss_weights = config.loss.weight

    criterion = dict()

    def _empty_l(*args, **kwargs):
        return 0

    def valid_l(name):
        return (
            loss_weights["flare"].get(name, 0.0) > 0
            or loss_weights["scene"].get(name, 0.0) > 0
        )

    criterion["l1"] = torch.nn.L1Loss().to(device) if valid_l("l1") else _empty_l
    criterion["lpips"] = LPIPSLoss().to(device) if valid_l("lpips") else _empty_l
    criterion["ffl"] = FocalFrequencyLoss().to(device) if valid_l("ffl") else _empty_l
    criterion["perceptual"] = (
        PerceptualLoss(**config.loss.perceptual).to(device)
        if valid_l("perceptual")
        else _empty_l
    )

    return criterion


def add_flare(scene, flare, config):
    apply_fn = getattr(synthesis, config.augment.way)
    kwargs = config.augment[config.augment.way]
    return apply_fn(scene, flare, **kwargs)


def init_weights(net, init_type="xavier_uniform", init_gain=1):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method:
            normal | xavier_normal | xavier_uniform | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_in", nonlinearity="relu"
                )
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logger.info("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def backup_config(config, output_dir):
    cur = datetime.now()
    config_name = f'config@{cur.strftime("%Y-%m-%d_%H-%M-%S")}.yml'

    (output_dir / config_name).write_text(OmegaConf.to_yaml(config))
    logger.debug(f"backup config at {output_dir / config_name}")


def train(config):
    device = torch.device("cuda")

    output_dir = Path(config.work_dir) / f"checkpoints" / config.name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.add(output_dir / "file_{time}.log")
    backup_config(config, output_dir)

    generator = utils.instantiate(networks, config.model.generator)
    generator = generator.to(device)
    init_weights(generator)
    logger.info(f"build generator over: {generator.__class__.__name__}")
    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=config.optimizer.lr
    )
    logger.info(f"build optimizer over: {optimizer}")

    criterion = build_criterion(config, device)

    train_dataset = UnpairedDataset(**config.train.dataset)
    logger.info(f"build train_dataset over: {train_dataset}")
    logger.info(f"{len(train_dataset)=}")
    train_dataloader = DataLoader(train_dataset, **config.train.dataloader)
    logger.info(f"build train_dataloader with config: {config.train.dataloader}")

    tb_path = Path(config.work_dir) / f"tb_logs" / config.name
    if not tb_path.exists():
        tb_path.mkdir(parents=True)
        logger.info(f"mkdir {tb_path}")
    tb_writer = SummaryWriter(tb_path.as_posix())

    running_scalars = defaultdict(float)

    start_epoch = 1
    if config.get("resume_from", None) is not None:
        checkpoint_path = Path(config.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
        ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))

        start_epoch = ckp["epoch"] + 1
        generator.load_state_dict(ckp["g"])
        optimizer.load_state_dict(ckp["g_optim"])
        logger.success(f"load state_dict from {checkpoint_path}")

    evaluate_dataset = PairedDataset(**config.evaluate.dataset)
    evaluate_dataloader = DataLoader(evaluate_dataset, **config.evaluate.dataloader)

    def flare_generator(images):
        pred_scene = generator(images).clamp(0.0, 1.0)
        pred_flare = synthesis.remove_flare(images, pred_scene)
        return pred_flare

    for epoch in range(start_epoch, config.train.num_epoch + 1):
        epoch_start_time = time.time()
        logger.info(f"EPOCH[{epoch}/{config.train.num_epoch}] START")
        generator.train()
        for iteration, batch in tqdm(
            enumerate(train_dataloader, 1),
            total=len(train_dataloader),
            leave=False,
            ncols=120,
        ):
            scene, flare = batch["a"]["image"], batch["b"]["image"]
            scene = scene.to(device, non_blocking=True)
            flare = flare.to(device, non_blocking=True)

            scene, flare, combined, gamma = add_flare(scene, flare, config)

            pred_scene = generator(combined).clamp(0.0, 1.0)
            pred_flare = synthesis.remove_flare(combined, pred_scene, gamma)

            flare_mask = synthesis.get_highlight_mask(flare)
            # Fill the saturation region with the ground truth, so that no L1/L2
            # loss and better for perceptual loss since
            # it matches the surrounding scenes.
            masked_scene = pred_scene * (1 - flare_mask) + scene * flare_mask
            masked_flare = pred_flare * (1 - flare_mask) + flare * flare_mask

            loss = dict()

            for t, pred, gt in [
                ("scene", masked_scene, scene),
                ("flare", masked_flare, flare),
            ]:
                l = dict(
                    l1=criterion["l1"](pred, gt),
                    ffl=criterion["ffl"](pred, gt),
                    lpips=criterion["lpips"](pred, gt, value_range=(0, 1)),
                    perceptual=criterion["perceptual"](pred, gt, value_range=(0, 1)),
                )
                for k in l:
                    if config.loss.weight[t].get(k, 0.0) > 0:
                        loss[f"{t}_{k}"] = config.loss.weight[t].get(k, 0.0) * l[k]

            optimizer.zero_grad()
            total_loss = sum(loss.values())
            total_loss.backward()
            optimizer.step()

            ##########################################
            # log stuff
            for k, v in loss.items():
                running_scalars[k] = running_scalars[k] + v.detach().mean().item()

            global_step = (epoch - 1) * len(train_dataloader) + iteration

            if global_step % config.log.tensorboard.scalar_interval == 0:
                tb_writer.add_scalar(
                    "metric/total_loss", total_loss.detach().cpu().item(), global_step
                )
                for k in running_scalars:
                    v = running_scalars[k] / config.log.tensorboard.scalar_interval
                    running_scalars[k] = 0.0
                    tb_writer.add_scalar(f"loss/{k}", v, global_step)

            if global_step % config.log.tensorboard.image_interval == 0:
                images = utils.grid_transpose(
                    [combined, scene, pred_scene, flare, pred_flare]
                )
                images = torchvision.utils.make_grid(
                    images, nrow=5, value_range=(0, 1), normalize=True
                )
                tb_writer.add_image(
                    f"train/combined|real_scene|pred_scene|real_flare|pred_flare",
                    images,
                    global_step,
                )

        logger.info(
            f"EPOCH[{epoch}/{config.train.num_epoch}] END "
            f"Taken {(time.time() - epoch_start_time) / 60.0:.4f} min"
        )
        if epoch % config.log.checkpoint.interval_epoch == 0:
            to_save = dict(
                g=generator.state_dict(), g_optim=optimizer.state_dict(), epoch=epoch
            )
            torch.save(to_save, output_dir / f"epoch_{epoch:03d}.pt")
            logger.info(f"save checkpoint at {output_dir / f'epoch_{epoch:03d}.pt'}")
        if epoch % config.log.evaluate.interval_epoch == 0:
            generator.eval()
            metrics = evaluate_fn(
                config, evaluate_dataloader, flare_generator, device=device
            )
            generator.train()
            logger.info(
                f"EPOCH[{epoch}/{config.train.num_epoch}] metrics "
                + "\t".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            )
            for m, v in metrics.items():
                tb_writer.add_scalar(f"evaluate/{m}", v, epoch * len(train_dataloader))

    logger.success(f"train over.")


def main(config, *omega_options, gpus="all"):
    config = Path(config)
    assert config.exists(), f"config file {config} do not exists."

    omega_options = [str(o) for o in omega_options]
    cli_config = OmegaConf.from_cli(omega_options)
    if len(cli_config) > 0:
        logger.info(f"set options from cli:\n{OmegaConf.to_yaml(cli_config)}")

    config = OmegaConf.merge(OmegaConf.load(config), cli_config)

    if gpus != "all":
        gpus = gpus if isinstance(gpus, Iterable) else [gpus]
        gpus = ",".join([str(g) for g in gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        logger.info(f"set CUDA_VISIBLE_DEVICES={gpus}")

    train(config)


if __name__ == "__main__":
    fire.Fire(main)

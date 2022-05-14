import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import fire
import torch
import torchvision
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import networks
import synthesis
import utils
from data import UnpairedDataset
from losses.focal_frequency_loss import FocalFrequencyLoss
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss

# Small number added to near-zero quantities to avoid numerical instability.
_EPS = 1e-7


def remove_flare(combined, flare, gamma=2.2):
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    combined = combined.clamp(_EPS, 1.0)
    flare = flare.clamp(_EPS, 1.0)

    combined_linear = torch.pow(combined, gamma)
    flare_linear = torch.pow(flare, gamma)

    scene_linear = combined_linear - flare_linear
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    scene_linear = scene_linear.clamp(_EPS, 1.0)
    scene = torch.pow(scene_linear, 1.0 / gamma)
    return scene


def get_highlight_mask(image, threshold=0.99):
    binary_mask = image.mean(dim=1, keepdim=True) > threshold
    binary_mask = binary_mask.to(image.dtype)
    return binary_mask


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
    criterion["perceptual"] = PerceptualLoss(**config.loss.perceptual)

    return criterion


def train(config):
    device = torch.device("cuda")

    generator = networks.UNet(**config.model.generator).to(device)
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

    start_epoch = 0
    if config.get("resume_from", None) is not None:
        checkpoint_path = Path(config.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
        ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))

        start_epoch = ckp["epoch"] + 1
        generator.load_state_dict(ckp["g"])
        optimizer.load_state_dict(ckp["g_optim"])

    for epoch in range(start_epoch, config.train.num_epoch):
        epoch_start_time = time.time()
        for iteration, batch in tqdm(
            enumerate(train_dataloader, 1), total=len(train_dataloader)
        ):
            scene, flare = batch["a"]["image"], batch["b"]["image"]
            scene = scene.to(device, non_blocking=True)
            flare = flare.to(device, non_blocking=True)

            scene, flare, combined, gamma = synthesis.add_flare(
                scene,
                flare,
                resize_scale=(0.5, 1.5),
                apply_affine=True,
                apply_random_white_balance=False,
                resolution=512,
                flare_max_gain=2.0,
                noise_strength=0.01,
            )

            pred_scene = generator(combined).clamp(0.0, 1.0)
            pred_flare = remove_flare(combined, pred_scene, gamma)

            flare_mask = get_highlight_mask(flare)
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

            global_step = epoch * len(train_dataloader) + iteration

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
            f"EPOCH[{epoch}/{config.train.num_epoch}] over. "
            f"Taken {(time.time() - epoch_start_time) / 60.0} min"
        )
        if (epoch + 1) % config.log.checkpoint.interval_epoch == 0:
            save_dir = Path(config.work_dir) / f"checkpoints" / config.name
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            to_save = dict(
                g=generator.state_dict(), g_optim=optimizer.state_dict(), epoch=epoch
            )
            torch.save(to_save, save_dir / f"epoch_{epoch}.pt")
            logger.info(f"save checkpoint at {save_dir / f'epoch_{epoch}.pt'}")

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

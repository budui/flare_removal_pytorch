import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import fire
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import networks
import synthesis
import utils
from data import UnpairedDataset, PairedDataset, ImageDataset


def add_flare(scene, flare, config):
    apply_fn = getattr(synthesis, config.augment.way)
    kwargs = config.augment[config.augment.way]
    return apply_fn(scene, flare, **kwargs)


def pred_to_image(mask, scale=4):
    return torch.sigmoid(
        F.interpolate(
            mask,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
    ).repeat(1, 3, 1, 1)


def train(config):
    device = torch.device("cuda")

    output_dir = Path(config.work_dir) / f"checkpoints" / config.name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.add(output_dir / "train_{time}.log")

    segmentor = networks.TopFormer(out_channels=1, out_act_conf=None).to(device)
    logger.info(f"build segmentor over: {segmentor.__class__.__name__}")
    optimizer = torch.optim.Adam(
        [p for p in segmentor.parameters() if p.requires_grad], lr=config.optimizer.lr
    )
    logger.info(f"build optimizer over: {optimizer}")

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    train_dataset = UnpairedDataset(**config.train.dataset)
    logger.info(
        f"build train_dataset over: {train_dataset}. with config {config.train.dataset}"
    )
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
        segmentor.load_state_dict(ckp["g"])
        optimizer.load_state_dict(ckp["g_optim"])
        logger.success(f"load state_dict from {checkpoint_path}")

    evaluate_dataset = PairedDataset(**config.evaluate.dataset)
    evaluate_dataloader = DataLoader(evaluate_dataset, **config.evaluate.dataloader)

    for epoch in range(start_epoch, config.train.num_epoch + 1):
        epoch_start_time = time.time()
        logger.info(f"EPOCH[{epoch}/{config.train.num_epoch}] START")
        segmentor.train()
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

            flare_mask = synthesis.flare_to_mask(flare)

            pred_mask = segmentor(combined)

            loss = dict(
                bce=criterion(
                    pred_mask,
                    F.interpolate(
                        flare_mask,
                        pred_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                )
            )

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
                with torch.no_grad():
                    images = utils.grid_transpose(
                        [
                            combined,
                            scene,
                            flare,
                            flare_mask.repeat(1, 3, 1, 1),
                            pred_to_image(pred_mask),
                        ]
                    )[: 5 * 8]
                    images = torchvision.utils.make_grid(
                        images, nrow=5, value_range=(0, 1), normalize=True
                    )
                    tb_writer.add_image(
                        f"train/combined|scene|flare|flare_mask|pred_mask",
                        images,
                        global_step,
                    )

        logger.info(
            f"EPOCH[{epoch}/{config.train.num_epoch}] END "
            f"Taken {(time.time() - epoch_start_time) / 60.0:.4f} min"
        )
        if epoch % config.log.checkpoint.interval_epoch == 0:
            to_save = dict(
                g=segmentor.state_dict(), g_optim=optimizer.state_dict(), epoch=epoch
            )
            torch.save(to_save, output_dir / f"epoch_{epoch:03d}.pt")
            logger.info(f"save checkpoint at {output_dir / f'epoch_{epoch:03d}.pt'}")
        if epoch % config.log.evaluate.interval_epoch == 0 or epoch == 1:
            segmentor.eval()
            test_images = []
            with torch.no_grad():
                for batch in evaluate_dataloader:
                    images, gt_images = batch["a"]["image"], batch["b"]["image"]
                    images = images.to(device, non_blocking=True)
                    gt_images = gt_images.to(device, non_blocking=True)
                    pred_mask = segmentor(images)
                    pred_gt_mask = segmentor(gt_images)

                    test_image = torch.cat(
                        [
                            images,
                            pred_to_image(pred_mask),
                            gt_images,
                            pred_to_image(pred_gt_mask),
                        ],
                        dim=0,
                    )
                    test_images.append(
                        torchvision.utils.make_grid(
                            test_image,
                            nrow=4,
                            value_range=(0, 1),
                            normalize=True,
                        )
                    )
                    if len(test_images) > 8:
                        break
            segmentor.train()

            test_images = torchvision.utils.make_grid(test_images, nrow=1)

            tb_writer.add_image(
                f"test/with_flare|with_flare_seg|without_flare|without_flare_seg",
                test_images,
                epoch * len(train_dataloader),
            )

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


def test(image_folder, output_folder, resume_from, device="cuda"):
    device = torch.device(device)
    torch.set_grad_enabled(False)
    segmentor = networks.TopFormer(out_channels=1, out_act_conf=None).to(device)
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
    ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))
    segmentor.load_state_dict(ckp["g"])
    logger.success(f"load model weights from {checkpoint_path} over")

    segmentor.eval()

    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir()

    test_dataset = ImageDataset(
        folders=image_folder,
        transform=["ToTensor", dict(Resize=dict(size=256))],
        recursive=True,
        return_image_path=True,
    )
    logger.info(f"build test_dataset over: {test_dataset}")
    logger.info(f"{len(test_dataset)=}")

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for iteration, batch in tqdm(
        enumerate(test_dataloader, 1),
        total=len(test_dataloader),
        ncols=120,
    ):
        images = batch["image"]
        images = images.to(device, non_blocking=True)

        mask = segmentor(images)

        test_image = torch.cat([images, pred_to_image(mask)], dim=0)

        test_image = torchvision.utils.make_grid(
            test_image,
            nrow=2,
            value_range=(0, 1),
            normalize=True,
        )
        image_name = Path(batch["path"][0]).name
        torchvision.utils.save_image(test_image, output_folder / image_name)


if __name__ == "__main__":
    fire.Fire(dict(train=main, test=test))

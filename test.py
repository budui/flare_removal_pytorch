from collections import defaultdict
from pathlib import Path

import cv2
import fire
import kornia as K
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
import synthesis
import utils
from data import PairedDataset, ImageDataset


def batch_remove_flare(
    images,
    flare_generator,
    resolution=512,
    high_resolution=2048,
):
    _, _, h, w = images.shape

    if min(h, w) >= high_resolution:
        images = T.functional.center_crop(images, [high_resolution, high_resolution])
        images_low = F.interpolate(images, (resolution, resolution), mode="area")
        pred_flare_low = flare_generator(images_low)
        pred_flare = T.functional.resize(
            pred_flare_low, [high_resolution, high_resolution], antialias=True
        )
        pred_scene = synthesis.remove_flare(images, pred_flare)
    else:
        images = T.functional.center_crop(images, [resolution, resolution])
        pred_flare = flare_generator(images)
        pred_scene = synthesis.remove_flare(images, pred_flare)

    try:
        pred_blend = synthesis.blend_light_source(images.cpu(), pred_scene.cpu())
    except cv2.error as e:
        logger.error(e)
        pred_blend = pred_scene
    return dict(
        input=images.cpu(),
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu(),
    )


def save_batch_images(result, output_path, resolution=512):
    images = []
    for k in ["input", "pred_blend", "pred_scene", "pred_flare"]:
        image = result[k]
        if max(image.shape[-1], image.shape[-2]) > resolution:
            image = T.functional.resize(image, [resolution, resolution], antialias=True)
        images.append(image)
    images = utils.grid_transpose(images)
    images = torchvision.utils.make_grid(
        images, nrow=5, value_range=(0, 1), normalize=True
    )
    torchvision.utils.save_image(images, output_path)


def generate_fn(
    config,
    generate_dataloader,
    flare_generator,
    output_folder,
    device=torch.device("cuda"),
):
    resolution = config.resolution
    high_resolution = config.high_resolution
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir()

    for iteration, batch in tqdm(
        enumerate(generate_dataloader, 1),
        total=len(generate_dataloader),
        ncols=120,
    ):
        images = batch["image"]
        images = images.to(device, non_blocking=True)

        results = batch_remove_flare(
            images,
            flare_generator,
            resolution=resolution,
            high_resolution=high_resolution,
        )

        image_name = Path(batch["path"][0]).name
        save_batch_images(results, output_folder / image_name, resolution)


def evaluate_fn(
    config,
    evaluate_dataloader,
    flare_generator,
    save_images=False,
    output_folder=None,
    device=torch.device("cuda"),
    remain_pbar=False,
):
    resolution = config.resolution
    high_resolution = config.high_resolution
    if output_folder is not None:
        output_folder = Path(output_folder)
        if not output_folder.exists():
            output_folder.mkdir()
    assert not (save_images and output_folder is None)

    metrics = defaultdict(float)

    for iteration, batch in tqdm(
        enumerate(evaluate_dataloader, 1),
        total=len(evaluate_dataloader),
        leave=remain_pbar,
        ncols=120,
    ):
        images, gt_images = batch["a"]["image"], batch["b"]["image"]
        images = images.to(device, non_blocking=True)

        results = batch_remove_flare(
            images,
            flare_generator,
            resolution=resolution,
            high_resolution=high_resolution,
        )

        if save_images:
            image_name = Path(batch["a"]["path"]).name
            save_batch_images(results, output_folder / image_name, resolution)

        metrics["PSNR"] += K.metrics.psnr(results["pred_blend"], gt_images, 1.0).item()
        metrics["SSIM"] += (
            K.metrics.ssim(results["pred_blend"], gt_images, 11).mean().item()
        )

    for k in metrics:
        metrics[k] = metrics[k] / len(evaluate_dataloader)

    return metrics


def init_generator(config, resume_from, device=torch.device("cuda")):
    generator = utils.instantiate(networks, config.model.generator).to(device)

    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
    ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))
    generator.load_state_dict(ckp["g"])
    logger.success(f"load model weights from {checkpoint_path} over")

    generator.eval()

    def flare_generator(images):
        pred_scene = generator(images).clamp(0.0, 1.0)
        pred_flare = synthesis.remove_flare(images, pred_scene)
        return pred_flare

    return flare_generator


def generate(
    config, resume_from, output_folder, image_folder=None, device=torch.device("cuda")
):
    config = OmegaConf.load(config)
    flare_generator = init_generator(config, resume_from, device)

    if image_folder is not None:
        config.generate.dataset.folders = [image_folder]

    generate_dataset = ImageDataset(**config.generate.dataset)
    logger.info(f"build generate_dataset over: {generate_dataset}")
    logger.info(f"{len(generate_dataset)=}")

    generate_dataloader = DataLoader(generate_dataset, **config.generate.dataloader)
    logger.info(f"build generate_dataloader with config: {config.generate.dataloader}")

    generate_fn(config, generate_dataloader, flare_generator, output_folder, device)


def evaluate(
    config,
    resume_from,
    save_images=False,
    output_folder=None,
    device=torch.device("cuda"),
):
    config = OmegaConf.load(config)

    flare_generator = init_generator(config, resume_from, device)

    evaluate_dataset = PairedDataset(**config.evaluate.dataset)
    logger.info(f"build evaluate_dataset over: {evaluate_dataset}")
    logger.info(f"{len(evaluate_dataset)=}")

    evaluate_dataloader = DataLoader(evaluate_dataset, **config.evaluate.dataloader)
    logger.info(f"build evaluate_dataloader with config: {config.evaluate.dataloader}")

    metrics = evaluate_fn(
        config, evaluate_dataloader, flare_generator, save_images, output_folder, device
    )
    logger.success(
        "evaluated metrics:\n"
        + "\n".join([f"\t{k}={v:.4f}" for k, v in metrics.items()])
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    fire.Fire(dict(evaluate=evaluate, generate=generate))

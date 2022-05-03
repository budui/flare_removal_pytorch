import torch
import networks
from pathlib import Path
import utils
import synthesis
from collections import defaultdict

from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
from losses.focal_frequency_loss import FocalFrequencyLoss
from torch.utils.tensorboard import SummaryWriter

from loguru import logger


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


def build_criterion(loss_weights, device):
    criterion = dict()

    def _empty_l(*args, **kwargs):
        return 0

    def valid_l(name):
        return (
            loss_weights["flare"].get(name, 0.0) > 0
            or loss_weights["scene"].get(name, 0.0) > 0
        )

    criterion["l1"] = torch.nn.L1Loss().to(device) if valid_l("l1") else _empty_l
    criterion["lpips"] = LPIPSLoss().to(device) if valid_l("l1") else _empty_l
    criterion["ffl"] = FocalFrequencyLoss().to(device) if valid_l("l1") else _empty_l

    return criterion


def train(
    num_epoch,
    log_scalar_interval_iteration=100,
    log_image_interval_iteration=2000,
    checkpoint_interval_epoch=2,
):
    device = torch.device("cuda")

    generator = networks.UNet(in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=1e-4
    )

    loss_weights = dict(
        flare=dict(
            ffl=100,
            lpips=1,
        ),
        scene=dict(
            l1=1,
            lpips=1,
        ),
    )

    criterion = build_criterion(loss_weights, device)

    train_dataloader = []

    tb_writer = SummaryWriter("runs/fashion_mnist_experiment_1")
    running_scalars = defaultdict(float)

    for epoch in range(num_epoch):
        for iteration, batch in enumerate(train_dataloader, 1):
            scene, flare = batch["A"], batch["B"]
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
                    if loss_weights[t].get(k, 0.0) > 0:
                        loss[f"{t}_{k}"] = loss_weights[t].get(k, 0.0) * l[k]

            optimizer.zero_grad()
            total_loss = sum(loss.values())
            total_loss.backward()
            optimizer.step()

            ##########################################
            # log stuff
            for k, v in loss.items():
                running_scalars[k] = running_scalars[k] + v.detach().mean().item()

            global_step = epoch * len(train_dataloader) + iteration

            if global_step % log_scalar_interval_iteration == 0:
                tb_writer.add_scalar(
                    "metric/total_loss", total_loss.detach().cpu().item(), global_step
                )
                for k in running_scalars:
                    v = running_scalars[k] / log_scalar_interval_iteration
                    running_scalars[k] = 0.0
                    tb_writer.add_scalar(f"loss/{k}", v, global_step)

            if global_step % log_image_interval_iteration == 0:
                pass

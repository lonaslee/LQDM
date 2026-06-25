# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional

import timm.loss
import torch
import torch.nn as nn
import torch.nn.functional as F

import io
import matplotlib.pyplot as plt
import wandb
from PIL import Image

from training.mask_classification_loss import MaskClassificationLoss, BoundaryBCELoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        boundary_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        if self.network.predict_boundaries:
            self.criterion2 = BoundaryBCELoss(boundary_coefficient=boundary_coefficient)

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, original_targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)

        if self.network.predict_boundaries:
            mask_logits_per_layer, class_logits_per_layer, boundary_logits_per_layer = self(crops)
        else:
            mask_logits_per_layer, class_logits_per_layer = self(crops)
            boundary_logits_per_layer = None

        targets = self.to_per_pixel_targets_semantic(original_targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 5:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

                if self.network.predict_boundaries and False:
                    fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

                    img = imgs[0].cpu().numpy().transpose(1, 2, 0)
                    axes[0].imshow(img)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')

                    boundary_logits = F.interpolate(boundary_logits_per_layer[i], self.img_size, mode="bilinear")
                    boundary_logits_reverted = self.revert_window_logits_semantic(
                        boundary_logits, origins, img_sizes
                    )

                    boundary_probs = torch.sigmoid(boundary_logits_reverted[0])  # [Q, 128, 128]
                    boundary_agg = boundary_probs.max(dim=0)[0].cpu().numpy()  # [128, 128]

                    img_h, img_w = img.shape[:2]
                    boundary_resized = F.interpolate(
                        torch.tensor(boundary_agg)[None, None, ...],  # [1, 1, 128, 128]
                        size=(img_h, img_w),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]  # [H, W]

                    axes[1].imshow(img)
                    boundary_mask = boundary_resized > 0.5
                    if boundary_mask.any():
                        axes[1].contour(boundary_mask.cpu().numpy(), colors='red', linewidths=2, alpha=0.8)
                    axes[1].set_title(f'Boundary Heatmap\n{boundary_mask.sum()} boundary pixels')
                    axes[1].axis('off')

                    axes[2].imshow(original_targets[0]['boundaries'][1].cpu().numpy(), cmap='gray')
                    axes[2].set_title(f'Ground truth')
                    axes[2].axis('off')

                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, facecolor="black")
                    plt.close(fig)
                    buf.seek(0)
                    block_postfix = self.block_postfix(i)
                    name = f"{log_prefix}_boundary_{batch_idx}{block_postfix}"
                    self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})


    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val", True)

    def on_validation_end(self):
        self._on_eval_end_semantic("val")

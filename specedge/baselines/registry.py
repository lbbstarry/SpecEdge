"""Baseline 模型注册.

统一接口: forward(image_NCHW) -> logits_NCHW (C = num_classes).
对二值分割, num_classes=1 时输出单通道 sigmoid logits;
num_classes=2 时输出双通道 softmax logits. 训练脚本会按配置处理.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_smp(name: str, encoder: str, encoder_weights: str | None, num_classes: int) -> nn.Module:
    import segmentation_models_pytorch as smp

    arch = {
        "smp_unet": smp.Unet,
        "smp_unetpp": smp.UnetPlusPlus,
        "smp_deeplabv3plus": smp.DeepLabV3Plus,
        "smp_fpn": smp.FPN,
        "smp_pspnet": smp.PSPNet,
    }[name]
    return arch(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


class HFSegformerWrapper(nn.Module):
    """HuggingFace SegFormer 包装为统一输出尺寸."""

    def __init__(self, pretrained: str, num_classes: int):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x).logits
        # SegFormer 输出 1/4 分辨率, 上采样回输入大小
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    name = model_cfg["name"]

    if name.startswith("smp_"):
        return _build_smp(
            name,
            encoder=model_cfg.get("encoder", "resnet50"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            num_classes=num_classes,
        )

    if name == "hf_segformer":
        return HFSegformerWrapper(
            pretrained=model_cfg.get("pretrained", "nvidia/mit-b2"),
            num_classes=num_classes,
        )

    if name == "mmseg":
        raise NotImplementedError(
            "mmseg adapter not wired yet — install mmsegmentation and implement loader for "
            f"{model_cfg.get('cfg')}"
        )

    raise ValueError(f"unknown model name: {name}")

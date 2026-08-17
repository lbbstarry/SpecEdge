# Baseline 训练脚手架

为论文计划 v2 第 7 节服务,统一训练/评估光刻 SEM 分割的 baseline 模型。

## 目录

```
scripts/baselines/
├── README.md                  # 本文档
├── configs/                   # 各模型配置(YAML)
│   ├── _base_.yaml            # 公共训练配置
│   ├── unet.yaml
│   ├── deeplabv3plus.yaml
│   ├── hrnet.yaml
│   ├── segformer.yaml
│   ├── pointrend.yaml
│   └── mask2former.yaml
├── train_baseline.py          # 统一训练入口
├── eval_baseline.py           # 统一评估入口(含 PSD 指标)
└── run_all.sh                 # 批量跑全部 baseline
```

支撑模块:
- `specedge/data/litho_dataset.py` — 光刻 SEM dataset
- `specedge/baselines/registry.py` — 模型注册与构建
- `specedge/metrics_psd.py` — 边缘 PSD 高频能量比指标

## 使用流程

### 0. 数据准备

期望目录:
```
dataset/litho/
├── images/{train,val,test}/*.png       # SEM 图像
├── masks/{train,val,test}/*.png        # GT mask (0/1 或 0/255)
└── splits.json                          # 可选: 显式划分
```

### 1. 单模型训练

```bash
python scripts/baselines/train_baseline.py \
    --config scripts/baselines/configs/unet.yaml \
    --data-root dataset/litho \
    --output output/baselines/unet
```

### 2. 评估(含 PSD 高频能量比)

```bash
python scripts/baselines/eval_baseline.py \
    --config scripts/baselines/configs/unet.yaml \
    --ckpt output/baselines/unet/best.pth \
    --data-root dataset/litho \
    --split test
```

输出:IoU、Dice、Boundary F1、Hausdorff95、**Edge PSD HF Ratio**、推理耗时。

### 3. 批量跑

```bash
bash scripts/baselines/run_all.sh
```

会依次训练并评估配置目录下的全部模型,结果汇总到 `output/baselines/summary.csv`。

## 依赖

- PyTorch >= 2.0
- `segmentation-models-pytorch`(U-Net/DeepLabV3+ 等 CNN baseline)
- `transformers`(SegFormer)
- `mmsegmentation`(可选,用于 PointRend/Mask2Former 等更复杂模型)

> 第一阶段只跑必跑 4 个(U-Net / DeepLabV3+ / HRNet / SegFormer),
> 验证"baseline 边缘 PSD 高频能量明显高于 GT"的现象后再扩展。

# SpecEdge

[English](README.md) | 中文

论文 **《Measurement-Error Characterization and Consistency Assessment
of Deep Segmentation for Lithography Metrology》** 的代码。

光刻量测从 SEM 图像中用阈值类算子提取轮廓，再从轮廓上读取关键尺寸（CD）与边缘
粗糙度。在先进制程节点，这类算子已经提取不出可用轮廓，学习式分割是候选的替代
方案。采用它，就等于把一个训练出来的模型放进了测量仪器内部，这带来一个确定性
路线从未面对的问题：如何证明一个分割模型"测得准"？目前领域的答案是分布内测试
划分上的掩膜重叠度。本仓库包含证明这一答案不成立的全部实验，以及取而代之的两
套机制。

## 不运行任何东西即可核对论文数字

聚合后的逐样本结果已包含在仓库中，论文里印出的每一个数字都能对回产生它的工件：

```bash
python scripts/verify_claims.py
```

该脚本保存了论文所印的 91 个数值，每个都配有对应的工件字段；它把工件值按论文
精度舍入后比对，任何不一致都会以非零退出码报错。预期输出：`91/91 claims verified`。

## 代码证明了什么

| 结论 | 出处 |
|---|---|
| 重叠度只读取边界位移场的一个泛函，因此固定重叠度下边界派生被测量的条件离散无界：单个窄 IoU 区间内 23 倍，588 幅交叉验证图像上 38 倍，前端身份只解释 0.1% | `revision_v4_analysis.py`（E7）、`e21_decoupling_cv.py` |
| 该推导经受控实验检验：用等 L1 范数、不同形状的位移场扰动 586 个掩膜，IoU 保持在 0.005 以内而 CD 误差变化 26 倍，衰减与 L/A 的相关系数 r = 0.989 | `e23_mechanism_synthetic.py` |
| 交叉验证重新训练下，四个前端之间的差异小于单个前端跨折的变化，且领先者随折次改变 | `e18_make_cv_folds.py`、`e18_run_cv.sh` |
| 工艺窗口外崩塌可复现，但不局限于某一种架构 | 同上，在 Extreme 划分上评估 |
| 以版图意图为准，Extreme 重叠度更高的前端产生 11 次错误设计判定，更低的产生 6 次 | `e13_layout_dfm_verdict.py`、`e12_dfm_verdict_fidelity.py` |
| 前端选择会沿前景占比代理轴移动工艺窗口边界的估计位置 | `e14_pw_boundary.py` |
| 无参考监测器检测工艺窗口外失效，AUROC 0.910 | `e4d_routing.py`、`e4c_loo_guard.py` |
| 监测器对比已发表的不确定度信号（max-softmax、熵、MC-dropout、深度集成） | `e20_uncertainty_baselines.py` |
| 监测器对比它必须胜过的策略：单独部署回退前端 | `e22_policy_baselines.py`、`e15_guard_protects_dfm.py` |
| 断点估计的 sup-F 自助检验、聚类自助法、Holm 校正 | `e16_statistical_corrections.py` |
| 条件风险模型、交叉拟合、窗口外校准漂移 | `e19_risk_model.py` |

## 目录结构

```
specedge/
  metrology.py          掩膜 -> 量测记录（CD、LWR、LER、PSD、拓扑）
  metrics.py            重叠度指标
  metrics_psd.py        边缘 PSD 指标
  baselines/registry.py 四个分割前端
  data/litho_dataset.py 图像/掩膜加载器
scripts/
  baselines/            训练与评估入口，每个前端一份配置
  prepare_*.py          数据集构建
  eval_metrology.py     对一个掩膜目录生成量测记录
  revision_v4_analysis.py   噪声下限、断点、监测器、IoU 分箱
  e1*.py e2*.py         上表列出的各实验
  verify_claims.py      将每个印出的数字对回其工件
  make_fig*.py replot_paper_figures.py   论文图
  prepare_teaser_sam_mask.py   首页图所用的直接分割掩膜（需要 SAM 权重）
output/revision_v4/     聚合结果（无图像），足以重新推出每一个数字
```

## 环境配置

```bash
conda create -n specedge python=3.10 && conda activate specedge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

前端依赖 `segmentation-models-pytorch`（U-Net、DeepLabV3+，以及经 `timm` 的
HRNet-W32 编码器）与 `transformers`（SegFormer）。

## 训练协议

四个前端使用完全相同的训练配置，因此任何差异都可归因于架构本身：512x512 输入、
BCE+Dice、AdamW（lr 1e-4，权重衰减 1e-4）、batch size 8、100 个 epoch 余弦退火
无预热、混合精度、随机种子 42，增广为翻转 / 90 度旋转 / ±0.1 亮度对比度。所有
编码器均从预训练权重出发（CNN 用 ImageNet，SegFormer 用 `nvidia/mit-b2`）。
报告的 checkpoint 取验证集 IoU 最优，分别落在第 63、84、75、61 个 epoch，均在
预算内。配置见 `scripts/baselines/configs/`。

## 复现

SEM 图像不可再分发（见下），因此依赖数据的步骤需要你自己的采集数据，按相同
布局组织：`images/{train,val,test}/*.png`，配套 `masks/`。

```bash
# 1. 训练并评估四个前端
bash scripts/baselines/run_all.sh dataset/litho output/baselines

# 2. 每个前端的量测记录
python scripts/eval_metrology.py --gt-dir dataset/litho/masks/test \
    --pred-dir output/baselines/segformer/preds/masks \
    --output output/metrology/segformer_test_metrics.csv

# 3. 主分析：噪声下限、断点、监测器、IoU 分箱
python scripts/revision_v4_analysis.py

# 4. 交叉验证（4 前端 x 5 折约 3 GPU 小时）
python scripts/e18_make_cv_folds.py --k 5 --seed 42 && bash scripts/e18_run_cv.sh

# 5. 设计侧、统计与策略实验（无需 GPU）
python scripts/e11_classical_baseline.py
python scripts/e12_dfm_verdict_fidelity.py
python scripts/e13_layout_dfm_verdict.py
python scripts/e14_pw_boundary.py
python scripts/e15_guard_protects_dfm.py
python scripts/e16_statistical_corrections.py --n-boot 2000
python scripts/e19_risk_model.py
python scripts/e21_decoupling_cv.py
python scripts/e22_policy_baselines.py
python scripts/e23_mechanism_synthetic.py    # 需要参考掩膜

# 6. 依赖 GPU 的探针
python scripts/e5a_inference_probe.py
python scripts/e20_uncertainty_baselines.py --mc-passes 20

# 7. 出图，然后核对每一个数字
python scripts/make_fig1_overview.py
python scripts/replot_paper_figures.py
python scripts/verify_claims.py
```

## 数据可用性

SEM 采集图像及由其导出的参考掩膜属于产线材料，不可再分发。参考掩膜由 LithoSeg
（[arXiv:2511.12005](https://arxiv.org/abs/2511.12005)）产生，那是作者与合作者
的另一项工作；该协议是本研究的输入而非组成部分。仓库中包含的是完整的分析流水
线、量测提取器、资格评定与监测协议、把掩膜变成论文数字的每一个脚本，以及这些
脚本产出的聚合逐样本结果。聚合结果只含测量值和零填充的样本编号，不含任何图像。

将该协议应用到其他数据，只需要一个二值掩膜目录和一份配套参考；
`specedge/metrology.py` 与监测器中没有任何为本数据集特化的部分。

## 未包含的内容

盲法重标注控制（论文附录 A）可由 `scripts/prepare_reannotation_kit.py` 复现，
但我们自己那一轮的揭盲密钥被有意保留，以便该控制可以被重复地盲做。

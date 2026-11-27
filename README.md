# MSAT

**Training-Free Zero-Shot Anomaly Detection via Multi-Scale Anchored Transport**

MSAT（Multi-Scale Anchored Transport）是一种无需训练的零样本视觉异常检测方法，使用冻结的视觉语言编码器，在不使用目标域训练图像、不更新模型参数的情况下，同时输出图像级异常分数与像素级异常图。

方法通过双尺度整图编码获取异常证据，无需滑动窗口推理。核心设计是分别处理定位所需的空间排序和图像判别所需的异常幅度：**AMT 为像素级定位进行同图分布锚定，GLAS 在传输前读取局部异常强度。**

## 研究动机

单一输入尺度难以同时捕获稳定的异常语义与细粒度局部缺陷。高分辨率分支可以补充细节，但直接融合会改变异常响应的空间排序与分数分布。MSAT 围绕这一问题构建共享的多尺度证据，并为定位和检测设计不同的读出方式。

## 方法概览

```text
测试图像 + 类别文本提示
             |
  冻结的 C-RADIOv4-h 双尺度整图编码
             |
  MSAM：多尺度结构感知异常映射
             |
       原生多尺度响应 U0
         /           \
        /             \
上采样与平滑       GLAS：传输前的局部上尾均值
      |                  + 全局异常间隔
AMT：同图参考分布锚定           |
      |                  图像级异常分数
像素级异常图
```

### MSAM — Multi-Scale Structure-Aware Anomaly Mapping

使用 `448 × 448` 与 `672 × 672` 两个输入尺度，结合 SigLIP2 对齐特征的异常语义、SAM3 对齐特征的物体部件结构以及骨干特征的内容亲和性，构建结构感知的异常响应。高分辨率响应通过精确面积重叠权重对齐到基础网格，再进行固定比例融合。

### AMT — Anchored Monotone Transport

以同一图像的基础尺度异常图作为参考分布，将多尺度异常图的响应排序映射到该分布。AMT 是满足单调约束与同值保持约束的一维 Wasserstein 投影：排序后，将每个相同源分数块映射到对应参考分数区间的均值，具有闭式解，无需学习或迭代优化。

这一过程保留多尺度异常响应的空间次序，同时使用基础尺度提供的分数尺度。没有源分数并列时，输出与参考图的经验分布一致；存在并列时，通过块均值保留同值约束。

### GLAS — Global–Local Anomaly Scoring

在 AMT 之前，从原生多尺度网格中选取响应最高的 `8` 个 patch 并计算均值，再与全局异常间隔等权融合。这样既利用整图语义，也保留稀疏局部缺陷的幅度信息。

正常与异常文本锚点分别记为 `t_normal` 和 `t_anomaly`，异常间隔为：

$$
\mu(v)=\langle v,t_{anomaly}\rangle-\langle v,t_{normal}\rangle.
$$

图像级分数为：

$$
s(x,c)=\tfrac{1}{2}G_0+\tfrac{1}{2}\operatorname{Mean}(\operatorname{Top8}(U_0)).
$$

## 实验范围

论文在 **7 个工业数据集与 7 个医疗数据集** 上评估，共覆盖 **66 个类别、20,277 张图像或原始帧**。

| 场景 | 数据集 | 任务 |
| --- | --- | --- |
| 工业异常检测 | MVTec-AD、VisA、BTAD、MPDD、KolektorSDD、DAGM、DTD-Synthetic | 图像级检测与像素级定位 |
| 医疗图像检测 | HeadCT、BrainMRI、Br35H | 图像级检测 |
| 医疗异常定位 | ISIC、CVC-ClinicDB、CVC-ColonDB、TN3K | 像素级定位 |

图像级指标包括 AUROC、max-F1 和 AP；像素级指标另包括 normalized AUPRO@0.3 与 PRO@1%FPR。指标先在数据集内按类别平均，再在相应任务的数据集间等权平均。

- [图像级检测对比表](results/table1.md)
- [像素级定位对比表](results/table2.md)

现有对比表中的对比方法数值仍有来源与协议核对事项，详见表内说明。已记录的复现范围与局限见 [复现记录](docs/reproducibility.md)。

## 快速开始

当前实现位于 `a40` 分支；Python 包与命令行入口保留为 `amt1`。

```bash
git clone --branch a40 https://github.com/liuweizz/MSAT.git
cd MSAT
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python -m amt1 --help
```

使用 Python 3.12 或更高版本，并根据本机 CUDA 环境安装合适的 PyTorch。依赖版本与方法参数分别见 [requirements.txt](requirements.txt) 和 [configs/method.json](configs/method.json)。

先按照 [模型准备说明](docs/assets.md) 配置本地模型源代码、权重与 `configs/assets.json`，再运行单图推理：

```bash
python -m amt1 predict --image /path/to/image.png --category bottle \
  --assets configs/assets.json --device cuda:0 --output outputs/image.npz
```

输出包含图像级分数 `J`、全局分数 `G`、原生多尺度响应 `native`，以及适用时的参考图 `A`、未锚定图 `U` 和锚定图 `Q`。分类专用医疗图像使用不同的几何设置，示例见 [数据与推理说明](docs/datasets.md)。

## 复现与文档

| 内容 | 入口 |
| --- | --- |
| 模型资产、版本与本地路径 | [docs/assets.md](docs/assets.md) |
| 数据准备、医疗推理与评估清单 | [docs/datasets.md](docs/datasets.md) |
| 验证范围、断点续传与已知差异 | [docs/reproducibility.md](docs/reproducibility.md) |
| 固定方法参数 | [configs/method.json](configs/method.json) |
| 数值检查 | [scripts/check_numerics.py](scripts/check_numerics.py) |
| 上游来源与许可证 | [third_party/NOTICES.md](third_party/NOTICES.md) |

完整的清单评估与缓存回放需要额外的评估清单、模型资产及相应数据，仓库不包含全部实验资产。现有记录中的固定观测回放与历史输出一致；有限样本的新 RGB 推理检查存在小幅最终异常图差异，尚未证明全量原图推理与历史结果逐位一致。具体数值与数据协议限制见复现记录。

## 仓库保存范围

`results/` 仅保存轻量的指标汇总表或图。特征张量、逐样本异常分数和异常图、模型权重、缓存与运行日志保存在仓库外或被 Git 忽略的本地目录中。`configs/data_preparation/` 是输入数据准备清单，不是模型运行结果。

## 许可证与致谢

项目新增代码采用 [Apache-2.0](LICENSE)。RADIOlogy、C-RADIO、SigLIP2 等上游代码与模型保留各自许可证及使用条件，详见 [第三方声明](third_party/NOTICES.md)。

# Semantic Region-Aware Style Transfer (SRAST)

基于 **MediaPipe** + **SCHP** + **BiSeNet** 多模型混合语义分割、 **CLIP** 语义对齐与 **VGG** 风格迁移的灰度图像上色框架。

## 📖 项目简介

本项目旨在解决传统风格迁移在**灰度图上色 (Colorization)** 任务中的结构丢失、颜色溢出和灰度陷阱问题。不同于全局风格迁移，本框架采用**区域感知 (Region-Aware)** 策略，利用高精度语义分割技术将图像解耦为 `Skin` (皮肤/身体)、`Hair` (毛发)、`Clothes` (衣物) 和 `Background` (背景) 等多个精细语义层，并利用 CLIP 模型进行跨模态颜色校准。


**核心特性：**
1.  **高精度混合语义分割**：构建了集成 BiSeNet、SCHP 与 MediaPipe 的混合分割系统。利用 BiSeNet 的双路架构捕捉面部五官的高频细节，结合 SCHP 的自校正机制处理复杂人体结构，并辅以 MediaPipe 进行关键点定位。相比单一几何规则，显著提升了对发丝边缘、遮挡区域及细微五官的分割鲁棒性。
2.  **区域感知初始化**：：在优化前，基于精细的语义掩码（Mask）进行 **区域级 AdaIN** 统计量对齐。强制将参考图的颜色特征精准“移植”到原图对应区域，解决优化起点的“灰度陷阱”。
3.  **CLIP 语义引导**：利用 CLIP 视觉-语言模型，自动为每个区域匹配最佳颜色描述（如 "Blue Skin", "Black Hair"），实现风格化与语义逻辑的统一。
4.  **亮度保持融合**：最终输出采用 YUV 空间融合，在注入色彩的同时，完美保留原图的高清纹理与光影细节。


## 🛠️ 环境安装

建议使用 Anaconda 创建独立环境。

### 1. 克隆项目
```bash
git clone [https://github.com/seaweed211/Colorization.git](https://github.com/seaweed211/Colorization.git)
cd Colorization
```

### 2. 创建环境
```bash
conda create -n colorization python=3.10
conda activate colorization
```

### 3. 安装依赖
```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装 OpenAI CLIP (必须从 GitHub 安装)
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```

## 🚀 快速开始

### 1. 准备数据
* 将**灰度原图**放入 `inputs/` 文件夹（例如 `in1.jpg`）。
* 将**彩色参考图**放入 `styles/` 文件夹（例如 `ref1.jpg`）。
* *注意：参考图最好包含清晰的人脸，以便提取肤色特征。*

### 2. 修改配置
打开 `main.py`，在顶部的配置区域修改文件名：

```python
# main.py
CONTENT_IMG_PATH = "inputs/in1.jpg"
STYLE_IMG_PATH = "styles/ref1.jpg"
```

### 3. 运行脚本
```bash
python main.py
```

### 4. 查看结果
运行结束后，结果将保存在 `outputs/` 文件夹中：
* `out1.jpg`: 最终上色结果。
* `debug_masks/`: 包含 `skin`, `hair`, `bg` 的中间分割掩码（用于调试分割效果）。

## ⚙️ 关键参数说明 (main.py)

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| `CONTENT_WEIGHT` | `0.0` | 灰度图上色任务建议设为 0，否则原图的灰色会抑制上色。 |
| `STYLE_WEIGHT` | `1e4` | 风格损失权重。数值越大，颜色越接近参考图。 |
| `CLIP_WEIGHT` | `5000.0` | 语义约束权重。数值越大，越强制符合 "Blue Skin" 等描述。 |
| `PROMPT_POOLS` | Dict | 定义了每个区域可选的文本描述池，CLIP 会从中自动选择最匹配的词。 |

## 🧩 项目结构

```
├── inputs/           # 输入图像目录
├── styles/           # 参考图像目录
├── outputs/          # 输出结果目录
├── main.py           # 主程序：包含 VGG 模型、优化循环与 Loss 计算
├── mask_utils.py     # 工具库：包含 MediaPipe 、 SCHP、 BiSeNet 分割逻辑与兜底机制
└── requirements.txt  # 依赖清单
```

## 🤝 小组成员
* Member 1 黄懿茗
* Member 2 梁娇
* Member 3 严跃倩
* Member 4 李灿

## 📝 To-Do
* [ ] 优化 MediaPipe 在侧脸情况下的分割精度//尝试用其他语义分割或者训一个模型（segmentation）
* [ ] 数据集！！！（1.最终测试集；2.如果要自己训segmentation还要标数据）
* [ ] 量化评估方法


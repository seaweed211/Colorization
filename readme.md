基于 CLIP 语义引导与 VGG 结构约束的区域感知图像风格迁移方法


## 📖 项目简介

本项目针对灰度图像上色任务，设计了一套包含 **预处理**、**语义分割**、**特征对齐**、**迭代优化**、**后处理** 的全流程技术框架。代码采用工程化的 **Factory Pattern** 设计：人脸分割部分集成了 **BiSeNet** (基于ResNet)、**SCHP** 与 **MediaPipe** 多种分割后端，能够根据不同场景（人脸特写/全身/复杂背景）灵活切换或组合。图像上色部分则通过 `AdaINRegionAware` 类实现基于 Mask 的特征初始化，并利用 `OptimizationSolver` 结合 VGG 与 CLIP 损失函数进行像素级优化。最终加入后处理模块 `smart_color_merge` 输出高保真结果。

**核心技术模块（对应代码实现）：**

1.  **多后端语义分割工厂 (Segmentation Factory)**：
    *   代码实现了 `BiSeNetSegmenter`、`MediaPipeSegmenter` 等多个分割类。
    *   **BiSeNet**：加载 `resnet34` 预训练权重，提供 19 类精细分割 (Skin, Hair, Eyes, Clothes 等)，专攻高精度人脸解析。
    *   **MediaPipe 增强版**：集成 `FaceMesh` 与 `SelfieSegmentation`，并在代码中加入了 **“动态阈值降级”** 与 **“几何拆分兜底”** 机制（如 `chin_y` 下巴定位法），解决了阿凡达等非真实人脸无法检测的问题。
    *   **Mask 优化**：引入 `refine_masks` 函数，通过 **形态学膨胀 (Dilate)** 与 **高斯模糊 (Blur)** 处理，消除了分割边缘的锯齿与缝隙。

2.  **区域感知特征对齐 (Region-Aware Initialization)**：
    *   核心类 `AdaINRegionAware`：摒弃了全局风格迁移，改为遍历 Mask 字典 (`for region in ['skin', 'hair', 'bg']`)。
    *   **逻辑**：分别计算内容图与参考图在特定区域（如头发对头发）的均值与方差，进行统计量对齐。这作为优化的**冷启动 (Warm Start)**，有效防止了颜色溢出（如背景色染到人脸）。

3.  **CLIP 语义与 VGG 风格联合优化 (Hybrid Guidance Solver)**：
    *   核心类 `OptimizationSolver` 与 `CLIPGuidance`。
    *   **VGG Loss**：确保生成的纹理与参考图风格一致。
    *   **CLIP Loss**：若启用 `--use_clip`，代码会提取参考图的视觉 Embedding，约束生成结果在语义层面（如“蓝色的皮肤”）与参考图对齐，支持跨域风格迁移。

4.  **亮度保持与智能融合 (Smart Post-Processing)**：
    *   核心函数 `smart_color_merge` 与 `transfer_color_presolve_luminance`。
    *   **Alpha Blending**：代码中实现了基于 Mask 羽化的透明度融合算法，处理边缘过渡。
    *   **YUV 空间重组**：将生成图像的色度信息 (Color) 与原图的亮度信息 (Luminance) 分离并重组，确保在上色的同时，原图的光影结构和高清纹理不丢失。






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

### 4. 数据集下载链接
①FFHQ：https://github.com/NVlabs/ffhq-dataset
②Goblin Portraits：https://www.kaggle.com/datasets/jerimee/goblin-portraits



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

## 🤝 小组成员及分工
* 黄懿茗 25216731 前期方法调研、上色部分内容和展示、视频制作
* 梁娇 25216732 前期方法调研、分割部分内容和展示、PPT部分美化
* 严跃倩 25216754 前期方法调研、经典方法复现、实验结果展示
* 李灿 25216743 前期方法调研、PPT主要制作、选题和背景展示




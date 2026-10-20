# CNNSession —— 手写数字识别（CNN）全流程项目

基于 PyTorch 的手写数字识别（MNIST）完整项目，覆盖 **数据预处理 → 模型训练 → Web 推理** 全流程：

- **CNNTrainer/**：CNN 训练模块，读取 CSV 元数据 + 图片，训练 `SimpleCNN` 模型
- **CNNAPP/**：Flask Web 应用，加载训练好的模型，提供手写数字识别 API 和网页界面
- 支持 CPU / GPU 两种运行方式，可用 Docker Compose 一键编排（含 LabelStudio、MLflow 辅助服务）

## 目录结构

```
.
├── CNNAPP/                    # Web 推理服务（Flask）
│   ├── serve.py               # Flask 应用：模型加载、预测 API、静态页面
│   └── static/index.html      # 前端手写数字识别页面
├── CNNTrainer/                # 训练模块
│   ├── cnn_trainer.py         # 训练脚本（自定义 Dataset + SimpleCNN）
│   ├── generate_csv.py        # 生成图片元数据 CSV
│   ├── mnist_image_processor.py  # MNIST 原始数据 → PNG 图片转换
│   └── config.json            # 训练配置（epochs、batch_size、lr 等）
├── Dockerfile                 # 训练/服务镜像
├── docker-compose.yml         # 服务编排（LabelStudio / MLflow / trainer）
├── start_windows.ps1          # Windows 一键启停脚本
├── requirements.txt           # Python 依赖
└── .env                       # 本机路径配置（不进仓库，见下方说明）
```

## 快速开始

### 方式一：直接运行 Web 服务（本地）

1. 准备数据：将 MNIST 原始数据（`train-images.idx3-ubyte` / `t10k-images.idx3-ubyte`）放入
   `data/cnn/MNSIT/`，用 `mnist_image_processor.py` 转换为 PNG，再用 `generate_csv.py` 生成元数据 CSV。
2. 训练模型（可选，仓库不含已训练模型）：
   ```bash
   cd CNNTrainer
   python cnn_trainer.py
   ```
   训练产物 `cnn_model_full.pt` 会生成在当前目录，复制到 `CNNAPP/` 下。
3. 启动 Web 服务：
   ```bash
   pip install -r requirements.txt
   cd CNNAPP
   python serve.py
   ```
4. 浏览器打开 http://localhost:5000

### 方式二：Docker Compose

```bash
# 先按 .env.example 配置本机路径（DATA_PATH / MODEL_PATH / MLFLOW_PATH / HUGGINGFACE_PATH）
docker build -t cnn-ai-service:latest .
./start_windows.ps1        # 或手动 docker compose up -d
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 主页面（手写数字识别前端） |
| GET | `/health` | 健康检查（含模型是否加载、运行设备） |
| POST | `/predict` | 上传图片推理，multipart/form-data，字段 `image` |
| GET | `/stats` | 今日识别次数统计 |

`POST /predict` 返回示例：

```json
{ "prediction": 7, "confidence": 0.9965 }
```

## 数据说明

- 使用公开 MNIST 手写数字数据集（约 7 万张训练图 + 1 万张测试图）。
- **本仓库不包含原始数据和生成的 PNG 图片**（体积大且可再生），数据目录结构如下：

```
data/cnn/MNSIT/
├── train-images.idx3-ubyte   # 原始 MNIST 训练集（自行下载）
├── t10k-images.idx3-ubyte    # 原始 MNIST 测试集（自行下载）
└── CONV/
    ├── train-images/         # 生成的训练 PNG
    └── test-images/          # 生成的测试 PNG
```

- MNIST 原始数据可从官网或常用镜像下载。

## 模型

- 结构：`SimpleCNN` —— 2 层卷积（32/64 通道）+ 池化 + 全连接分类头（10 类）。
- 输入：28×28 RGB 图，经 `Grayscale → Normalize` 预处理。
- 训练配置见 [CNNTrainer/config.json](CNNTrainer/config.json)。
- 训练好的模型（`cnn_model_full.pt`）体积约 1.6MB，需自行训练或从 Release 获取。

## 配置说明（.env）

`.env` 保存本机路径（`DATA_PATH`、`MODEL_PATH`、`MLFLOW_PATH`、`HUGGINGFACE_PATH` 等），
因每台机器路径不同而**不进仓库**。部署时按需创建，Docker 启动脚本会读取它注入容器。

## 技术栈

PyTorch 2.2 · TorchVision · Flask · Docker Compose · MLflow · LabelStudio

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
# import mlflow.pytorch
# import mlflow
# from mlflow.tracking import MlflowClient
import os
import logging
import json
from datetime import datetime

# ==============================
# GPU 配置开关
# 设置为 True 使用 GPU，False 使用 CPU
# ==============================
USE_GPU = True

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义CNN模型架构（与训练脚本中相同）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取模块（卷积层+池化层）
        self.features = nn.Sequential(
            # 输入通道3（RGB），输出通道32，3x3卷积核，保持空间分辨率
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),  # 引入非线性
            nn.MaxPool2d(2, 2),  # 下采样，尺寸减半

            # 第二个卷积块
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 分类器模块（全连接层）
        self.classifier = nn.Sequential(
            # 展平后输入维度：64 * 7 * 7（经过两次池化）
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, num_classes)  # 输出层
        )
    
    # 前向传播定义
    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, 1)  # 展平操作（batch维度保留）
        return self.classifier(x)  # 分类决策

app = Flask(__name__)

# 统计数据文件路径
STATS_FILE = os.path.join(os.path.dirname(__file__), 'stats.json')

# 加载统计数据
def load_stats():
    """
    加载统计数据，包括今日识别次数
    
    返回:
        dict: 统计数据字典
    """
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            
            # 检查是否需要重置今日统计（新的一天）
            today = datetime.now().strftime('%Y-%m-%d')
            if stats.get('last_date') != today:
                stats['today_count'] = 0
                stats['last_date'] = today
            
            return stats
        except Exception as e:
            logger.error(f"加载统计数据失败: {e}")
    
    # 默认统计数据
    return {
        'today_count': 0,
        'last_date': datetime.now().strftime('%Y-%m-%d')
    }

# 保存统计数据
def save_stats(stats):
    """
    保存统计数据到文件
    
    参数:
        stats (dict): 统计数据字典
    """
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"保存统计数据失败: {e}")

# 更新今日识别次数
def update_today_count():
    """
    更新今日识别次数，每次调用+1
    """
    stats = load_stats()
    stats['today_count'] += 1
    save_stats(stats)
    return stats['today_count']

# 初始化统计数据
stats = load_stats()

# 配置设备（CPU/GPU）
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    if USE_GPU:
        logger.info("GPU requested but not available, using CPU instead")
    else:
        logger.info("Using CPU")

# # 从MLflow加载最新模型
# def load_latest_model():
#     """
#     从MLflow加载最新训练的CNN模型
#     
#     返回:
#         model: 加载好的PyTorch模型，已设置为评估模式
#     """
#     # 设置MLflow存储路径，优先使用环境变量，支持Windows调试时使用外部端口
#     mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
#     logger.info(f"MLflow URI: {mlflow_tracking_uri}")
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     client = MlflowClient()
#     
#     try:
#         # 获取实验信息
#         experiment = client.get_experiment_by_name("CNN_HandMade_MVP")
#         if not experiment:
#             logger.info("实验 CNN_HandMade_MVP 不存在，尝试使用默认实验")
#             experiment = client.get_experiment("0")
#         
#         logger.info(f"使用实验: {experiment.name} (ID: {experiment.experiment_id})")
#         
#         # 查找最新的运行
#         runs = client.search_runs(
#             experiment.experiment_id,
#             order_by=["start_time DESC"],
#             max_results=1
#         )
#         
#         if not runs:
#             logger.warning("没有找到运行记录")
#             return None
#         
#         latest_run = runs[0]
#         logger.info(f"加载模型: {latest_run.info.run_id}")
#         
#         # 下载并加载模型文件
#         import tempfile
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             try:
#                 # 下载模型文件
#                 client.download_artifacts(latest_run.info.run_id, "cnn_model_full.pt", tmp_dir)
#                 model_path = os.path.join(tmp_dir, "cnn_model_full.pt")
#                 
#                 # 直接加载完整模型，设置weights_only=False兼容PyTorch 2.6
#                 model = torch.load(model_path, weights_only=False)
#                 model = model.to(device)  # 移动到目标设备
#                 model.eval()  # 设置为评估模式
#                 logger.info("模型加载成功！")
#                 return model
#             except Exception as e:
#                 logger.error(f"模型文件下载或加载失败: {e}")
#                 return None
#     except Exception as e:
#         logger.error(f"MLflow模型加载失败: {e}")
#         return None

# 从本地文件加载模型的备选方案
def load_local_model():
    """
    从本地文件加载CNN模型
    
    返回:
        model: 加载好的PyTorch模型，已设置为评估模式
    """
    try:
        # 本地模型路径，优先使用环境变量，然后尝试默认路径
        local_model_path = os.getenv("LOCAL_MODEL_PATH", os.path.join(os.path.dirname(__file__), "cnn_model_full.pt"))
        
        if os.path.exists(local_model_path):
            # 设置weights_only=False兼容新版PyTorch
            model = torch.load(local_model_path, weights_only=False)
            model = model.to(device)  # 移动到目标设备
            model.eval()  # 设置为评估模式
            logger.info(f"本地模型加载成功：{local_model_path}")
            return model
        else:
            logger.warning(f"本地模型文件不存在：{local_model_path}")
            return None
    except Exception as e:
        logger.error(f"本地模型加载失败: {e}")
        return None

# 加载模型
logger.info("开始加载模型...")

# 直接从本地加载模型，跳过MLflow连接
logger.info("直接从本地加载模型，跳过MLflow连接...")
model = load_local_model()

if model is None:
    logger.error("本地模型加载失败！服务将继续运行，但预测功能不可用")

# 图像处理转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道灰度图
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

# 健康检查端点
@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点，用于确认服务是否正常运行
    
    返回:
        JSON响应，包含服务状态
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

# 推理端点
@app.route('/predict', methods=['POST'])
def predict():
    """
    推理端点，接收图像并返回预测结果
    
    请求:
        POST /predict
        Content-Type: multipart/form-data
        Form字段: image (上传的图像文件)
    
    返回:
        JSON响应，包含预测结果
    """
    if not model:
        return jsonify({"error": "模型未加载"}), 500
    
    # 检查是否有文件上传
    if 'image' not in request.files:
        return jsonify({"error": "没有图像文件"}), 400
    
    file = request.files['image']
    
    # 检查文件是否为空
    if file.filename == '':
        return jsonify({"error": "没有选择图像文件"}), 400
    
    try:
        # 读取图像
        img = Image.open(io.BytesIO(file.read()))
        
        # 应用转换
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
        img_tensor = img_tensor.to(device)  # 移动到目标设备
        
        # 进行预测
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
        
        # 更新今日识别次数
        update_today_count()
        
        return jsonify({
            "prediction": int(predicted.item()),
            "confidence": float(torch.softmax(output, 1).max().item())
        })
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({"error": str(e)}), 500

# 静态文件服务
# 使用环境变量或默认绝对路径指定静态文件目录
static_dir = os.getenv("STATIC_DIR", os.path.join(os.path.dirname(__file__), 'static'))
logger.info(f"静态文件目录: {static_dir}")
logger.info(f"静态文件目录是否存在: {os.path.exists(static_dir)}")
logger.info(f"index.html是否存在: {os.path.exists(os.path.join(static_dir, 'index.html'))}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    静态文件服务，用于提供前端页面和资源
    
    请求:
        GET /static/filename
    
    返回:
        静态文件内容
    """
    return send_from_directory(static_dir, filename)

# 获取统计数据端点
@app.route('/stats', methods=['GET'])
def get_stats():
    """
    获取系统统计数据，包括今日识别次数
    
    返回:
        JSON响应，包含统计数据
    """
    stats = load_stats()
    return jsonify({
        "today_count": stats.get('today_count', 0),
        "last_date": stats.get('last_date', datetime.now().strftime('%Y-%m-%d'))
    })

# 主页面
@app.route('/')
def index():
    """
    主页面，返回HTML表单用于图像上传和预测
    
    返回:
        HTML页面
    """
    # 直接返回index.html文件的内容
    try:
        index_path = os.path.join(static_dir, 'index.html')
        logger.info(f"尝试直接读取 index.html 文件: {index_path}")
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info("index.html文件读取成功")
        return content, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        logger.error(f"读取index.html文件失败: {e}")
        return f"Error reading index.html: {e}", 404

if __name__ == '__main__':
    # 启动Flask应用，使用默认端口5000（与docker-compose.yml映射保持一致）
    logger.info("启动Flask应用...")
    app.run(host='0.0.0.0', port=5000, debug=False)

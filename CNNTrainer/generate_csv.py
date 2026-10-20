import struct
import os
import pandas as pd

# 定义读取MNIST标签文件的函数
def read_mnist_labels(label_file):
    """
    读取MNIST标签文件，返回标签列表
    
    参数:
        label_file: str - MNIST标签文件路径（.idx1-ubyte格式）
    
    返回:
        list - 标签列表
    """
    with open(label_file, 'rb') as f:
        # 读取标签文件头部：魔数和标签数量
        magic, num_labels = struct.unpack('>ii', f.read(8))
        # 验证魔数
        if magic != 2049:
            raise ValueError("Invalid magic number. Expected 2049 for MNIST label file.")
        # 读取所有标签
        labels = struct.unpack(f'>{num_labels}B', f.read(num_labels))
    return labels

# 定义生成CSV文件的函数
def generate_csv(labels, output_csv, image_dir, prefix=''):
    """
    生成包含文件名和标签的CSV文件
    
    参数:
        labels: list - 标签列表
        output_csv: str - 输出CSV文件路径
        image_dir: str - 图像目录
        prefix: str - 文件名前缀（可选）
    """
    # 创建数据字典
    data = []
    for i, label in enumerate(labels):
        filename = f'{prefix}{i}.png' if prefix else f'{i}.png'
        data.append({
            'filename': filename,
            'label': label
        })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV文件已生成: {output_csv}")

# 主程序
if __name__ == "__main__":
    # 定义基本路径
    base_path = r'D:\workspace_ai\CNNSession\inputdata\MNSIT'
    conv_path = os.path.join(base_path, 'CONV')
    
    # 1. 处理训练集
    print("处理训练集...")
    train_labels_file = os.path.join(base_path, 'train-labels.idx1-ubyte')
    train_labels = read_mnist_labels(train_labels_file)
    train_csv = os.path.join(conv_path, 'train.csv')
    train_image_dir = os.path.join(conv_path, 'train-images')
    generate_csv(train_labels, train_csv, train_image_dir)
    
    # 2. 处理测试集
    print("处理测试集...")
    test_labels_file = os.path.join(base_path, 't10k-labels.idx1-ubyte')
    test_labels = read_mnist_labels(test_labels_file)
    test_csv = os.path.join(conv_path, 'test.csv')
    test_image_dir = os.path.join(conv_path, 'test-images')
    generate_csv(test_labels, test_csv, test_image_dir)
    
    print("所有CSV文件生成完成！")
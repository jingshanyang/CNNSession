# 导入struct模块，用于解析二进制数据
import struct
# 从PIL库导入Image模块，用于图像处理和保存
from PIL import Image
# 导入os模块，用于文件和目录操作
import os

# 定义处理MNIST图像文件的函数
def process_mnist_images(input_file, output_dir):
    """
    处理MNIST图像文件，将二进制数据转换为PNG图片
    
    参数:
        input_file: str - MNIST图像文件路径（.idx3-ubyte格式）
        output_dir: str - 输出PNG图片的目录
    """
    # 创建输出目录，如果目录已存在则不会抛出异常（exist_ok=True）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用with语句打开文件，自动管理文件资源，rb表示以二进制读取模式打开
    with open(input_file, 'rb') as f:
        # 读取文件头部的16个字节，使用struct.unpack解析为4个大端整数
        # >iiii表示：>（大端字节序）+ iiii（4个int类型，每个4字节）
        # 解析结果为：magic（魔数）、num_images（图像数量）、rows（行数）、cols（列数）
        magic, num_images, rows, cols = struct.unpack('>iiii', f.read(16))
        
        # 验证魔数是否为2051，这是MNIST图像文件的标准魔数
        # 如果魔数不匹配，说明文件格式错误，抛出ValueError异常
        if magic != 2051:
            raise ValueError("Invalid magic number. Expected 2051 for MNIST image file.")
        
        # 打印处理信息，使用f-string格式化字符串，显示图像总数和尺寸
        print(f"Processing {num_images} images from {os.path.basename(input_file)}, each {rows}x{cols} pixels...")
        
        # 使用for循环遍历所有图像，i从0到num_images-1
        for i in range(num_images):
            # 读取单张图像的数据，每张图像大小为rows*cols字节（28*28=784字节）
            img_data = f.read(rows * cols)
            
            # 使用struct.unpack将二进制数据解析为无符号字节数组
            # >{rows*cols}B表示：大端字节序 + 784个unsigned byte类型
            # list()将解析结果转换为Python列表，方便后续处理
            img_array = list(struct.unpack(f'>{rows*cols}B', img_data))
            
            # 将一维数组重塑为二维矩阵（28x28）
            # 使用列表推导式，j从0到rows-1，每次取cols个元素作为一行
            img_matrix = [img_array[j*cols : (j+1)*cols] for j in range(rows)]
            
            # 创建PIL Image对象，mode='L'表示8位灰度图像（0-255）
            # 参数为（宽度，高度），即（cols, rows）
            img = Image.new('L', (cols, rows))
            # 将二维矩阵转换为一维列表，用于Image.putdata()方法
            # 使用嵌套列表推导式，先遍历行，再遍历每行的像素
            img.putdata([pixel for row in img_matrix for pixel in row])
            
            # 构建输出文件路径，使用os.path.join拼接目录和文件名
            # f'{i}.png'使用f-string格式化，将索引i转换为字符串作为文件名
            output_path = os.path.join(output_dir, f'{i}.png')
            # 保存图像为PNG格式，'PNG'参数指定文件格式
            img.save(output_path, 'PNG')
            
            # 每处理1000张图像打印一次进度
            # (i + 1) % 1000 == 0 表示当处理数量是1000的倍数时打印
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_images} images")
        
        # 打印单个文件处理完成信息
        print(f"All images from {os.path.basename(input_file)} processed successfully!")

# 主程序入口
if __name__ == "__main__":
    # 定义MNIST数据集的基本路径
    base_path = r'D:\workspace_ai\CNNSession\inputdata\MNSIT'
    
    # 1. 处理训练集图像
    # 训练集输入文件路径
    train_input_file = os.path.join(base_path, 'train-images.idx3-ubyte')
    # 训练集输出目录
    train_output_dir = os.path.join(base_path, 'CONV', 'train-images')
    # 调用函数处理训练集
    process_mnist_images(train_input_file, train_output_dir)
    
    print("=" * 50)  # 分隔线，提高输出可读性
    
    # 2. 处理测试集图像
    # 测试集输入文件路径
    test_input_file = os.path.join(base_path, 't10k-images.idx3-ubyte')
    # 测试集输出目录
    test_output_dir = os.path.join(base_path, 'CONV', 'test-images')
    # 调用函数处理测试集
    process_mnist_images(test_input_file, test_output_dir)
    
    # 打印所有处理完成信息
    print("=" * 50)
    print("All MNIST datasets processed successfully!")

# 关于MNIST数据集底色的说明：
# 在MNIST原始数据中，像素值范围是0-255，其中：
# - 0表示黑色（底色）
# - 255表示白色（前景，数字部分）
# 这是因为MNIST数据集是通过扫描手写数字得到的，黑色背景上的白色数字
# 在转换为PNG时，PIL的'L'模式会保持这种映射关系，即0=黑色，255=白色

# CNN训练时的处理建议：
# 1. 必须进行像素值归一化：将0-255缩放到0-1之间
# 2. 可选进行标准化：减去均值并除以标准差
# 3. 不需要专门针对黑色底色做特殊处理，CNN会自动学习特征
# 4. 通常需要添加通道维度，将(28,28)转换为(28,28,1)
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn import CNN  # 请确保 cnn 模块存在并包含你的 CNN 模型定义

# 解析命令行参数
parser = argparse.ArgumentParser(description='Recognize digits using a trained CNN model')
parser.add_argument('--image', type=str, required=True, help='Path to the image for recognition')
args = parser.parse_args()

# 加载训练好的模型
model = CNN()
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义图像转换
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 确保输入是灰度图像
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

def recognize_from_image(image_path):
    pil_image = Image.open(image_path)

    # 数据预处理
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 进行识别
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    digit = predicted.item()
    print(f'Recognized Digit: {digit}')

# 调用图像识别函数
recognize_from_image(args.image)

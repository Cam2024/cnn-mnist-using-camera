import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn import CNN  # 导入你的CNN模型定义

# 加载训练好的模型
model = CNN()
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义图像预处理
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((28, 28)),  # 调整图像大小为28x28像素
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 数据预处理
    pil_image = Image.fromarray(gray)  # 将NumPy数组转换为PIL图像
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 进行识别
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    digit = predicted.item()

    # 在图像上绘制识别结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Digit: {digit}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow('Real-Time Digit Recognition', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

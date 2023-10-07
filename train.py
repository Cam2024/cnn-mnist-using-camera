import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cnn import CNN

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train a CNN on MNIST dataset')
parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
args = parser.parse_args()

# 检查GPU可用性并设置使用的设备
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

# 数据预处理和加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 创建模型实例并移动到GPU（如果可用）
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(args.epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # 每200个小批次打印一次损失
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

print('Training finished.')

# 保存训练后的模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')

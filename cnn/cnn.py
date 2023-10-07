import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积层：输入通道1（黑白图像），输出通道32，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # 第二个卷积层：输入通道32，输出通道64，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # 最大池化层：2x2池化窗口
        self.maxpool = nn.MaxPool2d(2)

        # 全连接层1：展平卷积层输出，64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # 全连接层2：128 -> 10（输出类别数，0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # 展平

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x


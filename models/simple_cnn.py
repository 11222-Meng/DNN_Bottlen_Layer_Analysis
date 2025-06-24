import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    定义一个更复杂的卷积神经网络（CNN）模型。
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层、激活函数、池化层和全连接层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 输入通道3，输出通道64
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输入通道64，输出通道128
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 输入通道128，输出通道256
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.fc1 = nn.Linear(256 * 64 * 64, 512)  # 全连接层，输入大小为256*64*64，输出大小为512
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)  # 全连接层，输入大小为512，输出大小为256
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)  # 输出层，假设有10个类别

    def forward(self, x):
        """
        定义前向传播过程。
        """
        x = self.pool1(self.relu1(self.conv1(x)))  # 卷积 -> ReLU -> 池化
        x = self.pool2(self.relu2(self.conv2(x)))  # 卷积 -> ReLU -> 池化
        x = self.pool3(self.relu3(self.conv3(x)))  # 卷积 -> ReLU -> 池化
        x = x.view(-1, 256 * 64 * 64)  # 展平操作，将多维张量展平为一维
        x = self.relu4(self.fc1(x))  # 全连接 -> ReLU
        x = self.relu5(self.fc2(x))  # 全连接 -> ReLU
        x = self.fc3(x)  # 输出层
        return x
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size=4):
    """
    加载 CIFAR-10 数据集。
    :param batch_size: 批量大小
    :return: 训练数据加载器
    """
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小
        transforms.ToTensor(),         # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 下载并加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",          # 数据集保存路径
        train=True,             # 加载训练集
        download=True,          # 如果本地没有数据集，则下载
        transform=transform     # 应用预处理
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 打印数据集信息
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Sample image shape: {train_dataset[0][0].shape}")

    return train_loader
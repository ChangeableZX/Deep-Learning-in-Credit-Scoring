import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def loader(other_columns,binary_columns,y,batch_size=32):
    # 计算连续特征的均值和标准差 (用于标准化)
    mean = other_columns.mean(axis=0)
    std = other_columns.std(axis=0)
    std[std == 0] = 1e-8  # 避免标准差为 0

    # 数据标准化
    other_columns = (other_columns - mean) / std

    other_columns = torch.tensor(other_columns.values, dtype=torch.float32)
    binary_columns = torch.tensor(binary_columns.values, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)  # 如果是分类问题，y应该是长整型

    # 创建数据集
    dataset = TensorDataset(binary_columns, other_columns, y)

    torch.manual_seed(42)
    # 划分训练集和测试集 (例如 80% 训练集，20% 测试集)
    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    test_size = len(dataset) - train_size  # 剩余部分用于测试
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader
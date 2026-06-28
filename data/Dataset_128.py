import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import random
# 定义 PointCloudDataset 类
class PointCloudDataset(Dataset):
    def __init__(self, file_list, normal):
        self.file_list = file_list
        self.normal = normal

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        point_cloud, label = self.load_single_point_cloud(file_path)
        # 提取文件名，不包含路径和扩展名
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long), file_name

    def load_single_point_cloud(self, file_path):
        data = np.loadtxt(file_path, delimiter=' ')
        xyz = data[:128, :3]  # 坐标
        xyz_old = xyz
        features = data[:128, 3:-1]  # 颜色
        labels = data[:128, -1].astype(int)  # 标签
        assert xyz.shape[0] == 128, "每个点云文件必须包含128个点"
        first_point = xyz[0, :]
        last_point = xyz[-1, :]
        dis = np.linalg.norm(last_point - first_point)
        if dis == 0:
            dis = 1e-8  # 设置一个很小的默认值来避免除零
        if self.normal=="nonormal":
            point_cloud = np.hstack((xyz_old, xyz, features))
            return point_cloud, labels[0]  # 假设每个文件整体有一个标签
        else:
            # 计算归一化
            if self.normal=="normalsub":
                xyz = xyz - first_point
                assert xyz.shape[0] == 128, "每个点云文件必须包含128个点"
                point_cloud = np.hstack((xyz_old, xyz, features))
                return point_cloud, labels[0]  # 假设每个文件整体有一个标签
            elif self.normal=="normaldiv":
                xyz = xyz - first_point
                xyz = xyz/dis
                assert xyz.shape[0] == 128, "每个点云文件必须包含128个点"
                point_cloud = np.hstack((xyz_old, xyz, features))
                return point_cloud, labels[0]  # 假设每个文件整体有一个标签


def load_dataset(normal, data_folder="/data/guosc24/rock-mass/data/shuyanti-32w", batch_size=32, train_precent=0.8):
    # 获取点云文件路径列表
    #data_folder = 'padded_files'  # 替换为你的点云文件夹路径
    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt') and re.match(r'^\d', f)]  # 根据你的文件格式调整
    # print(all_files[:10])
    # a=input()

    # 打乱文件顺序
    #np.random.shuffle(all_files)

    # 分割训练集和测试集
    split = int(train_precent * len(all_files))
    train_files = all_files[:split]
    test_files = all_files[split:]

    # 创建数据集对象
    train_dataset = PointCloudDataset(train_files, normal)
    test_dataset = PointCloudDataset(test_files, normal)

    # 设置批量大小
    #batch_size = 32

    # 创建数据加载器
    if train_precent != 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("loading dataset success!")
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_dataset(normal="normaldiv")
    print(len(train_loader), len(test_loader))
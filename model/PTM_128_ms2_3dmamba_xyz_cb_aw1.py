# Accuracy: 94.9%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.device import device

from datetime import datetime
from codebook import get_model
from mamba_ssm import Mamba


def square_distance(src, dst):
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)


        #后加的
        dist = dist.float()
        distance = distance.float()
        #后加的


        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def multi_scale(feature_map, img_size):
    """
    Args:
        feature_map: [batch_size, channels, img_size, img_size]
        img_size: The size of the input feature map, e.g., 64, 128, etc.
        target_size: The target size to resize the feature maps, e.g., (224, 224)
        
    Returns:
        combined_feature_map: [batch_size, channels, target_size, target_size]
    """
    # 第一个尺度: img_size x img_size -> target_size x target_size
    feature_map_large = feature_map
    
    # 第二个尺度: img_size/2 x img_size/2 -> target_size x target_size (提取中心区域)
    center_start_mid = img_size // 2 - img_size // 4
    center_end_mid = img_size // 2 + img_size // 4
    feature_map_mid = feature_map[:, :, center_start_mid:center_end_mid, center_start_mid:center_end_mid]

    # 第三个尺度: img_size/4 x img_size/4 -> target_size x target_size (提取中心区域)
    center_start_small = img_size // 2 - img_size // 8
    center_end_small = img_size // 2 + img_size // 8
    feature_map_small = feature_map[:, :, center_start_small:center_end_small, center_start_small:center_end_small]
    
    return feature_map_large, feature_map_mid, feature_map_small


    



class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)

        self.dwconv = nn.Conv1d(
            dim * 2, dim * 2,
            kernel_size=3,
            padding=1,
            groups=dim * 2
        )

        self.silu = nn.SiLU()

        # ✅ 官方 Mamba
        self.ssm = Mamba(
            d_model=dim * 2,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x1 = self.linear1(x)

        x2 = self.dwconv(x1.transpose(1, 2)).transpose(1, 2)
        x2 = self.silu(x2)

        # ✅ 官方 SSM（核心）
        x2 = self.ssm(x2)

        x1 = self.silu(x1)

        x = x1 + x2
        x = self.linear2(x)

        return x + residual

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        #print("new_points shape before permute:", new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, npoint, nsample, C) -> (B, C, nsample, npoint)
        #print("new_points shape after permute:", new_points.shape)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            #print(f"Conv {i}: in_channels = {conv.in_channels}, new_points channels = {new_points.shape[1]}")
            new_points = F.relu(bn(conv(new_points)))
            #print("new_points shape after relu:", new_points.shape)


        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        #print("new_points shape after max:", new_points.shape)
        #print("new_xyz shape:", new_xyz.shape)
        return new_xyz, new_points
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape#高分辨率
        _, S, _ = xyz2.shape#低分辨率

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
       #print("interpolated_points shape:", interpolated_points.shape)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        #print("new_points shape before permute:", new_points.shape)
        new_points = new_points.permute(0, 2, 1)
        #print("new_points shape after permute:", new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            #print(f"Conv {i}: in_channels = {conv.in_channels}, new_points channels = {new_points.shape[1]}")
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
            #print("new_points shape after relu:", new_points.shape)
        new_points = new_points.permute(0, 2, 1)

        return new_points

class PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1(nn.Module):
    def __init__(self, num_class=3):
        super(PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1, self).__init__()
        self.name = "PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1"
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128])#npoint:512->128
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128+3, mlp=[128, 128, 256])#npoint:128->64,nsample:64->32
        self.sa3 = PointNetSetAbstraction(npoint=32, radius=0.8, nsample=32, in_channel=256+3, mlp=[256, 512, 1024])#nsample:128->32
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+3, mlp=[256, 128, 128, 128])
        self.fc1 = nn.Linear(128, 128)
        self.mamba_layer = 4
        # self.order_indicator = OrderIndicator(dim=128)
        self.mamba_blocks = nn.Sequential(*[MambaBlock(dim=128) for _ in range(self.mamba_layer)])
        self.norm = nn.LayerNorm(128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 可学习权重参数
        self.alpha_large = nn.Parameter(torch.ones(1))  
        self.alpha_mid = nn.Parameter(torch.ones(1))    
        self.alpha_small = nn.Parameter(torch.ones(1))  
        self.mlp = nn.Sequential(
            nn.Linear(384, 512),  
            nn.BatchNorm1d(512),  
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_class)  # 最终分类
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(128, 128//4),
            nn.ReLU(),
            nn.Linear(128//4, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )


        self.codebook = get_model(concept_num=128, num_class=2, kpt_num=1, point_num=128, input_dims=128, emb_dims=128)
        # 两个可学习参数（初始为0即可）
        self.weights = nn.Parameter(torch.zeros(2))
        
    def forward(self, xyz, points, wh, seq, height):
        '''
        Step1:PointNet++升维
        将[8, 128, 3+3]升至[8, 128, 3+128]
        '''
        pointnet_start = datetime.now()
        batch_size, num_points, _ = xyz.shape

        l1_xyz, l1_points = self.sa1(xyz, points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

        x = l0_points #[8, 128, 128]
        

        pointnet_end = datetime.now()
        pointnet_time = pointnet_end - pointnet_start
        pointnet_seconds = pointnet_time.total_seconds()
        # print(f"pointnet用了 {pointnet_seconds} 秒")

        '''
        Step1.5:Mamba2:按投影顺序排序，维度不变
        '''
        mamba_start = datetime.now()
        # seq: (B, N, 3)
        mamba_outputs = []

        # ================================
        # 🔥 对3个序列分别做 Mamba
        # ================================
        for i in range(3):

            seq_i = seq[:, :, i]   # (B, N)

            # 排序
            _, sorted_indices = torch.sort(seq_i, dim=1)

            # gather排序
            sorted_x = torch.gather(
                x, 1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, x.size(2))
            )

            sorted_wh = torch.gather(
                wh, 1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, wh.size(2))
            )

            x_i = sorted_x
            wh_i = sorted_wh

            # # ================================
            # # 🔥 归一化（强烈建议加，防NaN）
            # # ================================
            
            

            # std = x_i.std(dim=1, keepdim=True)
            # std = torch.clamp(std, min=1e-2)   # 🔥关键

            # x_i = (x_i - x_i.mean(dim=1, keepdim=True)) / std
            # x_i = self.norm(x_i)
            # x_i = torch.clamp(x_i, -5, 5)

            # 反序
            x_reverse = torch.flip(x_i, dims=[1])

            # 正序 Mamba
            x_mamba = self.mamba_blocks(x_i)
            if torch.isnan(x_mamba).any():
                print(f"Warning: NaN in 正序mamba (axis {i})!")
                input("暂停")

            # 反序 Mamba
            x_reverse_mamba = self.mamba_blocks(x_reverse)
            if torch.isnan(x_reverse_mamba).any():
                print(f"Warning: NaN in 反序mamba (axis {i})!")
                input("暂停")

            # 翻转回来
            x_reverse2_mamba = torch.flip(x_reverse_mamba, dims=[1])

            # 双向融合（防爆）
            x_out = 0.5 * (x_mamba + x_reverse2_mamba)

            # ================================
            # 🔥 恢复原始顺序（关键！！）
            # ================================
            _, inverse_indices = torch.sort(sorted_indices, dim=1)

            x_out = torch.gather(
                x_out, 1,
                inverse_indices.unsqueeze(-1).expand(-1, -1, x_out.size(2))
            )

            mamba_outputs.append(x_out)

        # ================================
        # 🔥 三个轴结果求平均
        # ================================
        mamba_outputs = sum(mamba_outputs) / 3.0

        # 最终检查
        if torch.isnan(x).any():
            print("Warning: NaN in 最终输出!")
            input("暂停")

        mamba_end = datetime.now()
        mamba_time = mamba_end - mamba_start
        mamba_seconds = mamba_time.total_seconds()

        # print(f"mamba用了 {mamba_seconds} 秒")


        x = x + mamba_outputs



        '''
        Step1.6:codebook
        mamba得到的是[B,N=128,C=128]的特征，经过codebook应该不变
        '''
        

        center_xyz = xyz[:,0,:].unsqueeze(1)
        center_feat = x[:,0,:].unsqueeze(1)

        
        cb_result = self.codebook(xyz, x, center_xyz, center_feat)
        cb_feat = cb_result["pos_inv_part_feat"]
        

        w = F.softmax(self.weights, dim=0)
        w1 = self.gate_mlp(x)
        w2 = self.gate_mlp(cb_feat)

        x = w1*x + w2*cb_feat
        print("codebook成功")


        
        '''
        Step2:TriPlane
        将每个点的128维特征,根据之前算好的二维坐标,映射到三个平面中
        得到3个[8, 128, 16, 16],拼接起来得到[8, 384, 16, 16]
        '''
        # #推理用这个
        # x_cpu = x.cpu()
        # wh_cpu = wh.cpu().int()
        # # 初始化特征图
        # feature_maps = torch.zeros((batch_size, 3, 16, 16, 128), dtype=torch.float32)
        # counters = torch.zeros((batch_size, 3, 16, 16), dtype=torch.int32)
        
        #训练用这个
        x_cpu = x
        wh_cpu = wh.int()
        # 初始化特征图
        feature_maps = torch.zeros((batch_size, 3, 16, 16, 128), dtype=torch.float32, device=device)
        counters = torch.zeros((batch_size, 3, 16, 16), dtype=torch.int32, device=device)
        
        for_start = datetime.now()

        B, N, C = x.shape  # [batch_size, num_points, 128]

        # 拆坐标
        wh1 = wh_cpu[:, :, :2].long()   # [B, N, 2]
        wh2 = wh_cpu[:, :, 2:4].long()
        wh3 = wh_cpu[:, :, 4:6].long()


        # 初始化
        feature_maps = torch.zeros((B, 3, 16, 16, C), dtype=torch.float32, device=x.device)
        counters = torch.zeros((B, 3, 16, 16), dtype=torch.int32, device=x.device)

        # --------- 核心函数 ----------
        def scatter_plane(wh, feat, plane_id):
            u = wh[..., 0]  # [B, N]
            v = wh[..., 1]

            # flatten index: u * W + v
            idx = u * 16 + v  # [B, N]

            # reshape
            idx = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, N, C]

            # flatten feature map
            fmap = feature_maps[:, plane_id].view(B, -1, C)  # [B, 256, C]
            count = counters[:, plane_id].view(B, -1)        # [B, 256]

            # scatter add
            fmap.scatter_add_(1, idx, feat)

            # 计数（只需要1）
            ones = torch.ones_like(u, dtype=count.dtype)
            count.scatter_add_(1, u * 16 + v, ones)

        # --------- 三个平面 ----------
        scatter_plane(wh1, x_cpu, 0)
        scatter_plane(wh2, x_cpu, 1)
        scatter_plane(wh3, x_cpu, 2)

        # # 初始化坐标
        # wh1 = wh_cpu[:, :, :2]  # 第一个平面的坐标 [batch_size, 128, 2]
        # wh2 = wh_cpu[:, :, 2:4]  # 第二个平面的坐标 [batch_size, 128, 2]
        # wh3 = wh_cpu[:, :, 4:]  # 第三个平面的坐标 [batch_size, 128, 2]
        
        
        # for b in range(batch_size):
        #     # 将特征分配到特征图的对应像素位置
        #     for i in range(num_points):
        #         u, v = wh1[b, i]
        #         feature = x_cpu[b, i]  # [128]
        #         # 将特征加到对应的像素位置
        #         feature_maps[b, 0, u, v, :] += feature
        #         counters[b, 0, u, v] += 1

        #         u, v = wh2[b, i]
        #         feature = x_cpu[b, i]  # [128]
        #         # 将特征加到对应的像素位置
        #         feature_maps[b, 1, u, v, :] += feature
        #         counters[b, 1, u, v] += 1

        #         u, v = wh3[b, i]
        #         feature = x_cpu[b, i]  # [128]
        #         # 将特征加到对应的像素位置
        #         feature_maps[b, 2, u, v, :] += feature
        #         counters[b, 2, u, v] += 1


        feature_maps = feature_maps / (counters.unsqueeze(-1) + 1e-8)
        # print("有counter")
        
        for_end = datetime.now()
        for_time = for_end - for_start
        for_seconds = for_time.total_seconds()
        # print(f"for循环用了 {for_seconds} 秒")

        feature_maps = feature_maps.to(device=device)
        # print("****")
        # print("feature.device=",feature_maps.device)
        # print("*****")
        feature_maps = feature_maps.permute(0, 1, 4, 2, 3)
        B, _, C, H, W = feature_maps.shape
        feature_maps = feature_maps.reshape(B, -1, H, W)

        '''
        Step3:多尺度
        经过多尺度得到大中小[8, 384, 16, 16],[8, 384, 8, 8],[8, 384, 4, 4]
        '''

        multi_start = datetime.now()
        feature_maps_large, feature_maps_mid, feature_maps_small = multi_scale(feature_maps, feature_maps.shape[-1])
        multi_end = datetime.now()
        multi_time = multi_end - multi_start
        multi_seconds = multi_time.total_seconds()
        # print(f"多尺度用了 {multi_seconds} 秒")

        '''
        Step4:MLP
        分类
        '''
        mlp_start = datetime.now()
        feature_large_pooled = self.global_pool(feature_maps_large) 
        feature_mid_pooled = self.global_pool(feature_maps_mid)      
        feature_small_pooled = self.global_pool(feature_maps_small) 
        alphas = torch.softmax(torch.cat([self.alpha_large, self.alpha_mid, self.alpha_small]), dim=0)
        fused_feature = alphas[0] * feature_large_pooled + alphas[1] * feature_mid_pooled + alphas[2] * feature_small_pooled
        fused_feature = fused_feature.squeeze(-1).squeeze(-1)
        outputs = self.mlp(fused_feature)
        mlp_end = datetime.now()
        mlp_time = mlp_end - mlp_start
        mlp_seconds = mlp_time.total_seconds()
        # print(f"MLP用了 {mlp_seconds} 秒")
        return outputs




        


if __name__ == '__main__':
    input = torch.randn(8, 128, 6)
    xyz = input[:, :, :3]
    points = input[:, :, 3:]
    model = PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1()
    output = model(xyz, points)
    
    print(output.shape)
    
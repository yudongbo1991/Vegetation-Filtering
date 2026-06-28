import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image

def normalize(coords, min_val, max_val):
    """将坐标归一化到[0, 15]的范围"""
    epsilon = 1e-8
    range_val = max_val - min_val
    scaled = (coords - min_val) / (range_val + epsilon) * 15
    return np.round(scaled).clip(0, 15)  # 确保值在[0, 1]范围内

def project_points(points, plane_normal):
    """将点投影到平面上"""
    # 平面法向量归一化
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # 计算投影
    projections = points - np.outer(points @ plane_normal, plane_normal)
    return projections

def create_binary_image(points, scale):
    """根据归一化坐标生成黑白图像"""
    # 初始化图像矩阵
    image = np.zeros((scale, scale), dtype=np.uint8)
    
    # 将归一化坐标映射到离散坐标
    x_coords = (points[:, 0] * (scale - 1)).astype(int)
    y_coords = (points[:, 1] * (scale - 1)).astype(int)
    
    # 标记有点的格子为黑色（1）
    for x, y in zip(x_coords, y_coords):
        image[y, x] = 1
    
    return image

def process_file(input_path, output_folder, file_name):
    """处理单个文件"""
    # 读取数据（7列：xyz, nxnynz, label）
    data = np.loadtxt(input_path, dtype=np.float64)
    xyz = data[:, :3]  # 提取xyz坐标
    feature = data[:, 3:6]  # 提取法向量
    label = data[:, -1]  # 提取label
    # 对当前文件的点云进行PCA
    pca = PCA(n_components=3)
    pca.fit(xyz)
    plane_normals = pca.components_  # 得到 (3, 3) 矩阵，每一行是一个平面的法向量
    # plane_normals = [[0,1,0],[1,0,0],[0,0,1]]
    # 将点云投影到三个平面
    projections = []
    for normal in plane_normals:
        proj = project_points(xyz, normal)
        projections.append(proj)

    # 计算归一化参数
    normalized_coords = []
    for proj in projections:
        x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
        y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
        x_norm = normalize(proj[:, 0], x_min, x_max)
        y_norm = normalize(proj[:, 1], y_min, y_max)
        normalized_coords.append(np.column_stack((x_norm, y_norm)))
    
    # 合并新旧数据
    new_columns = np.hstack(normalized_coords)  # 将三个平面的归一化坐标合并
    label = np.expand_dims(label, axis=1)  # 将 label 转换为 2D 数组
    new_data = np.hstack((xyz, feature, new_columns, label))
    # 保存归一化数据
    
    output_path = os.path.join(output_folder, file_name)
    fmt = '%.6f' + ' %.6f'*5 + ' %d'*7
    np.savetxt(output_path, new_data, fmt=fmt)

    # # 生成黑白图像
    # scales = 16
    # for i, proj in enumerate(normalized_coords):
    #     # 生成图像
    #     image = create_binary_image(proj, scales)
        
    #     # 保存图像
    #     image_name = f"{os.path.splitext(file_name)[0]}_{scales}x{scales}_plane{i+1}.png"
    #     image_path = os.path.join(output_folder, image_name)
    #     Image.fromarray(image * 255).save(image_path)

def main(input_folder, output_folder):
    """主函数：处理所有文件"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    i=0
    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt') and file_name!= 'real_labels.txt':
            try:
                # 构造输入路径
                input_path = os.path.join(input_folder, file_name)

                # 处理当前文件
                process_file(input_path, output_folder, file_name)
                # print(f"Processed: {file_name}")
                if i%100==0:
                    print(i)
                i=i+1
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    input_folder = "/data/guosc24/rock-mass/data/yanjinghu-left-1w"  # 替换为你的输入文件夹路径
    output_folder = "/data/guosc24/rock-mass/data/yanjinghu-left-1w-2d-15"  # 替换为你的输出文件夹路径
    print(input_folder)
    print(output_folder)
    print("是否开始添加Y/N")
    a=input()
    if a=="Y":
        main(input_folder, output_folder)
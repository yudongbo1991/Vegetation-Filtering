import numpy as np
import os
from scipy.spatial import KDTree

# 文件路径设置
input_file = "/data/guosc24/rock-mass/OriginalData/hunluan2-32w.txt"
output_folder = "/data/guosc24/rock-mass/data/hunluan2-32w"
k=128
os.makedirs(output_folder, exist_ok=True)
print(input_file)
print(output_folder)
print("采集",k,"个邻域点")
# 读取点云数据
data = np.loadtxt(input_file)
points = data[:, :3]  # xyz
labels = data[:, -1].astype(int)  # label
print(len(points))
print("是否开始采样：Y/N")
choice = input()
if choice=='Y':
    # 构建 KDTree
    tree = KDTree(points)

    # 创建 real_labels.txt 文件
    real_labels_path = os.path.join(output_folder, "real_labels.txt")
    with open(real_labels_path, 'w') as label_file:

        # 遍历每个点，寻找最近的 512 个点
        for i in range(len(points)):
            if i%100==0:
                print(i)
            # 查询最近的 k 个点
            distances, indices = tree.query(points[i], k=k)

            # 提取最近邻的 xyz 和 label
            nearest_points = points[indices]
            nearest_labels = labels[indices]
            output_data = np.column_stack((nearest_points, nearest_labels))

            # 保存为 i.txt
            output_file = os.path.join(output_folder, f"{i}.txt")
            np.savetxt(output_file, output_data, fmt="%.6f %.6f %.6f %d")
            # 写入 real_labels.txt
            label_file.write(f"{i} {labels[i]}\n")

    print("处理完成，文件已保存至"+output_folder+"文件夹。")

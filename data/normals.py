import os
import numpy as np
import open3d as o3d


def load_point_cloud(file_path):
    # 加载点云文件
    data = np.loadtxt(file_path)
    points = data[:, :3]  # xyz坐标
    # colors = data[:, 3:6]  # rgb颜色
    labels = data[:, -1]  # 标签
    return points, labels


def estimate_normals(points, k_neighbors=30):
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    # 获取法向量
    normals = np.asarray(pcd.normals)
    return normals


def save_point_cloud(file_path, points, normals, labels):
    # 组合新的点云数据
    new_data = np.hstack((points, normals, labels.reshape(-1, 1)))
    fmt = ['%.6f'] * 6 + ['%d']
    # 保存到文件
    np.savetxt(file_path, new_data, fmt=fmt)


def process_point_clouds(folder_path):
    i=0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt') and file_name != 'real_labels.txt':
            if i%100==0:
                print(i)
            file_path = os.path.join(folder_path, file_name)

            # 加载点云数据
            points, labels = load_point_cloud(file_path)

            # 估计法向量
            normals = estimate_normals(points)

            # 保存新的点云数据
            save_point_cloud(file_path, points, normals, labels)
            i=i+1


if __name__ == "__main__":
    folder_path = '/data/guosc24/rock-mass/data/shuyanti-32w'  # 点云文件夹路径
    print(folder_path)
    print("是否开始添加法向量:Y/N")
    choice = input()
    if choice == 'Y':
        process_point_clouds(folder_path)
        print("所有点云文件添加法向量！")
        print(folder_path)

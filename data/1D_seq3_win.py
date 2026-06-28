import numpy as np
from sklearn.decomposition import PCA
import os
from tqdm import tqdm


# ================================
# 🔥 局部窗口排序（加入法向量）
# ================================
def local_reorder(points_window, normals_window, alpha=1.0, beta=0.3):
    """
    points_window: (k,3)
    normals_window: (k,3)

    距离 = α * 欧氏距离 + β * 法向量差
    """
    k = len(points_window)
    visited = np.zeros(k, dtype=bool)
    order = []

    current = 0
    order.append(current)
    visited[current] = True

    for _ in range(k - 1):
        remaining = np.where(~visited)[0]

        # ===== 空间距离 =====
        spatial_dist = np.linalg.norm(
            points_window[remaining] - points_window[current],
            axis=1
        )

        # ===== 法向量差（用1 - cos相似度）=====
        n1 = normals_window[current]
        n2 = normals_window[remaining]

        # 归一化（防止异常）
        n1 = n1 / (np.linalg.norm(n1) + 1e-8)
        n2 = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8)

        cos_sim = np.sum(n2 * n1, axis=1)
        normal_dist = 1.0 - cos_sim  # 越小越相似

        # ===== 综合距离 =====
        dists = alpha * spatial_dist + beta * normal_dist

        next_idx = remaining[np.argmin(dists)]

        order.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    return np.array(order)



# ================================
# 🔥 滑动窗口重排
# ================================
def sliding_window_reorder(points, normals, base_order,
                           window_size=16, step=8,
                           alpha=1.0, beta=0.3):
    N = len(points)
    reordered = base_order.copy()

    for start in range(0, N - window_size + 1, step):
        end = start + window_size

        window_idx = reordered[start:end]

        window_points = points[window_idx]
        window_normals = normals[window_idx]

        local_order = local_reorder(
            window_points,
            window_normals,
            alpha=alpha,
            beta=beta
        )

        reordered[start:end] = window_idx[local_order]

    return reordered

# ================================
# 🔥 单文件处理
# ================================
def process_file(input_path, output_path):
    data = np.loadtxt(input_path)

    if data.shape[1] != 13:
        raise ValueError(
            f"文件 {os.path.basename(input_path)} 列数应为13，实际为 {data.shape[1]}"
        )

    # xyz
    points = data[:, :3]

    # 法向量（你原数据第3~6列）
    normals = data[:, 3:6]

    # ================================
    # 🔥 PCA 主轴
    # ================================
    pca = PCA(n_components=3)
    pca.fit(points)
    principal_axes = pca.components_

    # 固定方向
    for i in range(3):
        if principal_axes[i, 0] < 0:
            principal_axes[i] = -principal_axes[i]

    # 投影
    projections = np.dot(points, principal_axes.T)

    # ================================
    # 🔥 三轴排序（升级版）
    # ================================
    order_columns = []

    for i in range(3):
        proj = projections[:, i]

        # 1️⃣ 全局排序
        base_order = np.argsort(proj)

        # 2️⃣ 滑动窗口优化（加入法向量）
        refined_order = sliding_window_reorder(
            points,
            normals,
            base_order,
            window_size=16,
            step=8,
            alpha=1.0,
            beta=0.3
        )

        # 3️⃣ 转rank
        order = np.zeros(len(data), dtype=int)
        order[refined_order] = np.arange(len(data))

        order_columns.append(order.reshape(-1, 1))

    order_columns = np.hstack(order_columns)

    # ================================
    # 🔥 计算点到3个主轴平面的距离
    # ================================

    # 参考点（第一个点）
    p0 = points[0]

    plane_distances = []

    for i in range(3):
        normal = principal_axes[i]

        # 确保是单位向量（保险）
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        # 向量差
        vec = points - p0  # (N,3)

        # 点到平面距离（绝对值）
        dist = np.abs(np.dot(vec, normal))  # (N,)

        plane_distances.append(dist.reshape(-1, 1))

    # (N,3)
    plane_distances = np.hstack(plane_distances)




    # ================================
    # 🔥 拼接输出
    # ================================
    new_data = np.hstack([
        data[:, :6].astype(float),
        data[:, 6:12].astype(int),
        order_columns.astype(int),
        plane_distances.astype(float),  # ⭐ 新增3列
        data[:, 12:].astype(int)
    ])


    fmt = ['%.6f'] * 6 + ['%d'] * 6 + ['%d'] * 3 + ['%.6f'] * 3 + ['%d']
    np.savetxt(output_path, new_data, fmt=fmt, delimiter=' ')


# ================================
# 🔥 文件夹处理
# ================================
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.endswith('.txt') and f != "real_labels.txt"
    ]

    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            process_file(input_path, output_path)
            tqdm.write(f"成功处理: {filename}")
        except Exception as e:
            tqdm.write(f"处理失败 {filename}: {str(e)}")


# ================================
# 🔥 主程序入口
# ================================
if __name__ == "__main__":
    input_folder = "hunluan2-1w-2d-15"
    output_folder = "hunluan2-1w-2d-3sw-3h-15"

    print("input_folder:", input_folder)
    print("output_folder:", output_folder)
    print("是否开始添加seq序号Y/N")

    a = input()
    if a == 'Y':
        process_folder(input_folder, output_folder)
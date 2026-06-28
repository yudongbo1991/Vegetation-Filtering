import os
import traceback
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from model.PTM_128_ms2_3dmamba_xyz_cb_aw1 import PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1 as PTM_128_ms2_3dmamba_xyz_cb_aw1
from data.Dataset_1282 import load_dataset
import argparse
from sklearn.neighbors import KDTree
from utils.device import device
from datetime import datetime





# 测试模型
def test_model(model, test_loader, device, data_name, normal):
    model.eval()
    save_path = model.name+"-hl1w"+normal+".pth"
    if os.path.exists(save_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(save_path, map_location=device))
        print('Model loaded from', save_path)
    print("模型正确吗Y/N")
    a=input()
    if a=='Y':
        correct = 0
        total = 0
        

        # 确保 'evaluate' 文件夹存在，如果不存在则创建
        if not os.path.exists('EvaluateData'):
            os.makedirs('EvaluateData')
        model_second_min = 100
        # 将标签文件保存到 'evaluate' 文件夹中
        file_path = os.path.join('EvaluateData', data_name + '_' + model.name + '_' + normal + '_labels.txt')
        with open(file_path, 'w') as f:
            with torch.no_grad():
                i=0
                for inputs, labels, file_names in tqdm(test_loader):

                    # if i%100==0:
                    #     print(i)
                    print(i)
                    inputs, labels = inputs.to(device), labels.to(device)
                    xyz_old = inputs[:, :128, :3]  # 取前3个通道作为坐标,真实坐标
                    xyz = inputs[:, :128, 3:6]
                    features = inputs[:, :128, 6:9]  # 取后三个通道作为特征
                    wh = inputs[:, :, 9:15]
                    wh = wh.to(torch.int)
                    wh2 = inputs[:, :, 9:15]
                    wh2 = wh2.to(torch.int)
                    seq = inputs[:, :, 15:18]
                    seq = seq.to(torch.int)
                    height = inputs[:, :, 18:21]
                    inputs = inputs[:, :, 3:9]
                    
                    model_start = datetime.now()
                    
                    if model.name == "PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1":
                        outputs = model(xyz, features, wh, seq, height)
                    


                    
                    model_end = datetime.now()
                    model_time = model_end - model_start
                    model_seconds = model_time.total_seconds()
                    print(f"model用了 {model_seconds} 秒")
                    if model_seconds < model_second_min:
                        model_second_min = model_seconds
                    # print("最小时间", model_second_min)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # print("predicted:", predicted)
                    # print("labels:", labels)
                    # a=input()
                    correct += (predicted == labels).sum().item()
                    centre_xyz = xyz_old[:, 0, :3]
                    for centre_xyz, pred_label in zip(centre_xyz, predicted):
                        x, y, z = centre_xyz[0].item(), centre_xyz[1].item(), centre_xyz[2].item()
                        # print(centre_xyz)
                        # print(x, y, z, pred_label.item())
                        f.write(f'{x:.6f} {y:.6f} {z:.6f} {pred_label.item()}\n')
                    i=i+1
                    # print("继续")
                    # a=input()
            print(f'Accuracy: {100 * correct / total}%')

def merge_xyz_label(xyz_file_path, label_file_path):
    # 检查行数是否相同
    with open(label_file_path, 'r') as label_file, open(xyz_file_path, 'r') as xyz_file:
        label_lines = label_file.readlines()
        xyz_lines = xyz_file.readlines()

    if len(label_lines) != len(xyz_lines):
        print("两个文件的行数不一致，无法进行替换操作。")
    else:
        # 行数一致，开始替换操作
        updated_lines = []
        for label_line, xyz_line in zip(label_lines, xyz_lines):
            # 提取数据
            label_columns = label_line.strip().split()
            xyz_columns = xyz_line.strip().split()
            
            # 替换前三列
            label_columns[:3] = xyz_columns[:3]
            
            # 保存更新后的行
            updated_lines.append(' '.join(label_columns) + '\n')

        # 将结果写回 label_file_path
        with open(label_file_path, 'w') as label_file:
            label_file.writelines(updated_lines)

        print("替换完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a point cloud classification model.")
    parser.add_argument('--data_folder', type=str, default='data/hunluan2-100000', help='path to the dataset')
    parser.add_argument('--data_normal', type=str, choices=['nonormal', 'normalsub','normaldiv'], default='normaldiv', help='process to the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--model', type=str, choices=['PTM_128_ms2_3dmamba_xyz_cb_aw1'],
                        help='model to use')

    args = parser.parse_args()
    data_name = args.data_folder.split('/')[-1]
    print(args)
    print("确定开始评估吗:Y/N")
    choice = input()
    if choice == 'Y':
        process = args.data_normal
        train_loader, test_loader = load_dataset(normal=process, data_folder=args.data_folder, batch_size=args.batch_size, train_precent=0)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.model == "PTM_128_ms2_3dmamba_xyz_cb_aw1":
            model = PTM_128_ms2_3dmamba_xyz_cb_aw1(num_class=3).to(device)
        

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # 获取当前时间
        start = datetime.now()
        # 格式化时间，精确到分钟
        starttime = start.strftime("%Y-%m-%d %H:%M")
        print(starttime)
        # 训练和测试
        try:
            test_model(model, test_loader, device, data_name, normal=process)
        except Exception as e:
            # 捕获异常信息
            with open('error_log.txt', 'w') as file:
                file.write("An error occurred:\n")
                # 使用 traceback.format_exc() 获取完整的错误信息
                file.write(traceback.format_exc())
            print("Error has been logged to 'error_log.txt'")


        file_path = os.path.join('EvaluateData', data_name + '_' + model.name + '_' + process + '_labels.txt')
        # 获取当前时间
        end = datetime.now()
        # 格式化时间，精确到分钟
        endtime = end.strftime("%Y-%m-%d %H:%M")
        print(endtime)
        print("在",args.data_folder,"上测试",model.name,"数据操作", process, "，从",starttime,"到",endtime)
        print("文件保存在", file_path)


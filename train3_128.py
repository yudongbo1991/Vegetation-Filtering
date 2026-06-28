import os
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from tqdm import tqdm
from model.PTM_128_ms2_3dmamba_xyz_cb_aw1 import PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1 as PTM_128_ms2_3dmamba_xyz_cb_aw1
from data.Dataset_128 import load_dataset
import argparse
from sklearn.neighbors import KDTree
from utils.device import device
from datetime import datetime
import math




# ==========================================================
# Chamfer Distance
# ==========================================================
def chamfer_distance(p1, p2):
    """
    p1,p2:
        [B,G,K,3]
    """

    B, G, K, _ = p1.shape

    p1 = p1.reshape(B * G, K, 3)
    p2 = p2.reshape(B * G, K, 3)

    dist = torch.cdist(p1, p2)

    mins1 = dist.min(dim=2)[0]
    mins2 = dist.min(dim=1)[0]

    loss = mins1.mean() + mins2.mean()

    return loss


# 训练模型
def train_model(model, train_loader, criterion, optimizer, device, normal, num_epochs=10):
    model.train()
    best_loss = float('inf')
    save_path = model.name+"-hl1w"+normal+".pth"

    print(save_path)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print('Model loaded from', save_path)
    else:
        print('No saved model found, training from scratch.')


    log_file = open("training_log.txt", "w")
    train_second_min=100
    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0):

        running_loss = 0.0
        epoch_loss = 0.0

        for i, (inputs, labels, filename) in enumerate(tqdm(train_loader)):
            batch_start = datetime.now()
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
            optimizer.zero_grad()
            
            if model.name == "PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1":
                outputs = model(xyz, features, wh, seq, height)
            
            loss = criterion(outputs, labels)
            back_start = datetime.now()
            loss.backward()
            back_end = datetime.now()
            # 计算时间差
            backtime = back_end - back_start
            # 提取秒数
            seconds = backtime.total_seconds()
            print(f"本次backward用了 {seconds} 秒")
            optimizer.step()
            running_loss += loss.item()
            batch_end = datetime.now()
            # 计算时间差
            batchtime = batch_end - batch_start
            # 提取秒数
            seconds = batchtime.total_seconds()
            print(f"本次batch用了 {seconds} 秒")
            if seconds < train_second_min:
                train_second_min = seconds
            # print("最小时间", train_second_min)
            if math.isnan(running_loss):
                print("第",epoch,"轮,loss出现了nan")
                print(f"Loss is NaN at epoch {epoch}, breaking the training loop.")
                # 将当前 epoch 写入文件
                log_file.write(f"Loss is NaN at epoch {epoch}, step {i+1}/{len(train_loader)}\n")
                log_file.flush()  # 确保内容被写入文件
                print("暂停，等待输入：")
                a=input()
                break
            epoch_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
            # print("继续")
            # a=input()
        if math.isnan(running_loss):
            break
        epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with loss {best_loss:.4f}')
    log_file.close()

# 测试模型
def test_model(model, test_loader, device, normal):
    model.eval()
    save_path = model.name+"-hl1w"+normal+".pth"
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print('Model loaded from', save_path)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, filename in test_loader:
            batch_start = datetime.now()
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
            
            if model.name == "PointNet2_Triplane2_MLP_128_multiscale2_3dmamba_xyz_cb_aw1":
                outputs = model(xyz, features, wh, seq, height)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_end = datetime.now()
            # 计算时间差
            batchtime = batch_end - batch_start
            # 提取秒数
            seconds = batchtime.total_seconds()
            print(f"本次batch用了 {seconds} 秒")
    print(f'Accuracy: {100 * correct / total}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a point cloud classification model.")
    parser.add_argument('--data_folder', type=str, default='data/padded_files', help='path to the dataset')
    parser.add_argument('--data_normal', type=str, choices=['nonormal', 'normalsub','normaldiv'], default='normaldiv', help='process to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', type=str, choices=['PTM_128_ms2_3dmamba_xyz_cb_aw1'],
                        help='model to use')
    args = parser.parse_args()
    print(args)
    print("确定开始训练吗:Y/N")
    choice = input()
    if choice == 'Y':
        process = args.data_normal
        train_loader, test_loader = load_dataset(normal=process, data_folder=args.data_folder, batch_size=args.batch_size, train_precent=0.8)
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
            # 训练和测试
            train_model(model, train_loader, criterion, optimizer, device, normal=process, num_epochs=args.num_epochs)
            print("///////")
            print("////")
            print("开始testY/N")
            print("////")
            print("///////")
            
            test_model(model, test_loader, device, normal=process)
        except Exception as e:
            # 捕获异常信息
            with open('error_log.txt', 'w') as file:
                file.write("An error occurred:\n")
                # 使用 traceback.format_exc() 获取完整的错误信息
                file.write(traceback.format_exc())
            print("Error has been logged to 'error_log.txt'")

        # 获取当前时间
        end = datetime.now()
        # 格式化时间，精确到分钟
        endtime = end.strftime("%Y-%m-%d %H:%M")
        print(endtime)
        print("在",args.data_folder,"上训练",model.name,"，训练",args.num_epochs,"，从",starttime,"到",endtime)


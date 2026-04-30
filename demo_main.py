"""Train the model: 单个场景， ADS谱作为输入"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as scio
from torch.utils.data import TensorDataset, DataLoader
import model.net as net
import model.data_loader as data_loader
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LinearSegmentedColormap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def read_large_mat_file(filepath, dataset_name):
    mat = h5py.File(filepath, 'r')
    data = mat.get(dataset_name)
    data = np.array(data)
    data_t = np.transpose(data)
    return np.expand_dims(data_t, axis=1)


def get_individual_rmse(n1,n2):
    n = np.power(n1-n2, 2)
    n = np.sqrt(np.sum(n, axis=1))
    return n.reshape(-1, 1)


def plot_error_hist(data):
    plt.hist(data, bins=30, edgecolor='black')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_ADS(matrix_data):
    colors = [(0.7, 0.7, 0.7), (0.4, 0.6, 0.8)]  # 从灰色到蓝色的渐变
    cmap = LinearSegmentedColormap.from_list('grayblue', colors)

    # # 创建自定义灰绿色cmap
    # colors = [(0.7, 0.7, 0.7), (0.5, 0.7, 0.5)]  # 从灰色到绿色的渐变
    # cmap = LinearSegmentedColormap.from_list('graygreen', colors)

    # 创建X和Y坐标
    x = np.arange(matrix_data.shape[1])  # 列数，即408
    y = np.arange(matrix_data.shape[0])  # 行数，即32
    x, y = np.meshgrid(x, y)

    # Z轴对应矩阵的幅度
    z = matrix_data

    # 创建一个3D图形
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D表面图
    surf = ax.plot_surface(x, y, z, cmap=cmap)

    # 取消坐标轴上的ticks
    ax.set_xticks([])  # 取消X轴的ticks
    ax.set_yticks([])  # 取消Y轴的ticks
    ax.set_zticks([])  # 取消Z轴的ticks


    # 设置图形标题和轴标签
    fz = 24
    ax.set_xlabel('Delay domain', fontsize=fz)
    ax.set_ylabel('Angle domain', fontsize=fz)
    ax.set_zlabel('Power', fontsize=fz)

    # 显示图形
    plt.grid()
    plt.show()
def training(model, optimizer, loss_fn, dataloader):
    # set model to training mode
    model.train()

    # Use tqdm for progress bar
    loss_list = []
    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available

        train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
        # compute model output and loss
        output_batch = model(train_batch)

        loss = loss_fn(output_batch,  labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.item())

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss

    return np.mean(np.array(loss_list))

def save_model_dict(model, path):
    torch.save(model.state_dict(), path)

def evaluating(model, dataloader):
    model.eval()
    numpy_data1 = []
    numpy_data2 = []
    error = []
    for i, (test_batch, labels_batch) in enumerate(dataloader):
        test_batch, labels_batch = test_batch.to(device), labels_batch.to(device)

        # compute model output and loss
        output_batch = model(test_batch)
        output_batch = output_batch.squeeze(0)



        # 将Tensor转换为NumPy数组
        nd1 = output_batch.detach().cpu().numpy()
        nd2 = labels_batch.detach().cpu().numpy()
        err = get_individual_rmse(nd1, nd2)
        if i == 0:
            numpy_data1 = nd1
            numpy_data2 = nd2
            error = err
        else:
            numpy_data1 = np.concatenate((numpy_data1,nd1), axis=0)
            numpy_data2 = np.concatenate((numpy_data2, nd2), axis=0)
            error = np.concatenate((error, err), axis=0)



    numpy_data = np.concatenate((numpy_data1,numpy_data2, error), axis=1)
    # 创建DataFrame对象
    df = pd.DataFrame(numpy_data, columns=['X-est', 'Y-est', "X-gt", "Y-gt", "Error"])

    # 保存为Excel文件
    mae = np.mean(error)
    # excel_file = r"./log/output_scene"+str(X)+"_mae"+str(mae)+".xlsx"
    # #plot_error_hist(error)
    # df.to_excel(excel_file, index=False)
    return mae



def plot_pos(pos):
    pos = pos.numpy()
    plt.scatter(pos[:, 0], pos[:, 1], c='blue', marker='o')
    plt.show()
def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, save_model = False):
    rmse_min = 10
    loss_list = []
    for epoch in tqdm(range(num_epochs)):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))


        # compute number of batches in one epoch (one full pass over the training set)
        loss = training(model, optimizer, loss_fn, train_dataloader)
        loss_list.append(loss.item())
        if epoch%10 == 0:
            print("epoch:",epoch," loss:", loss)
        if epoch > 100 and epoch%50 == 0:
            rmse_eval = evaluating(model, test_dataloader)
            print("Testing RMSE:", rmse_eval)
            if rmse_min > rmse_eval:
                rmse_min = rmse_eval
                # 保存模型的状态字典和优化器的状态
                if save_model:
                    if transfer_learning:
                        save_path = r'./exp_icassp/ckp_transfer_scene'+str(X)+'TUP_' + str(rmse_eval) + "_epoch_" +str(epoch) + '.pth'
                    else:
                        save_path = r'./exp_icassp/ckp_scratch_scene'+str(X)+'TUP_' + str(rmse_eval) + "_epoch_" +str(epoch)+ '.pth'

                    save_model_dict(model, path=save_path)
                    print("saving successfully")
    scio.savemat("./exp_icassp/loss_TUP_"+"scene"+str(X)+"_"+str(rmse_eval)+".mat", {"loss" : loss_list})









if __name__ == "__main__":

    X = 2
    batch_size = 512
    Pos_path = "D:/Huawei_competition_2024/初赛测试数据/初赛测试数据/Round0_data_distill/Position_scene"+ str(X) + ".mat"
    data = scio.loadmat(Pos_path)
    Pos_all = np.array(data["Pos"]).astype(float)
    Anch_bool_all = np.array(data["Anch_bool"]).astype(int)

    ADS_path = "D:/Huawei_competition_2024/初赛测试数据/初赛测试数据/Round0_ADS_new/ADS_scene_"+ str(X) +".mat"

    ADS_all = read_large_mat_file(ADS_path, "ADS_new_all")




    train_sample = 1500
    anch_sample = 500
    ADS_train = torch.tensor(ADS_all[:train_sample, :, :, :], dtype=torch.float32)
    Pos_train = torch.tensor(Pos_all[:train_sample, :], dtype=torch.float32)
    ADS_anch = torch.tensor(ADS_all[:anch_sample, :, :, :], dtype=torch.float32).to(device)
    Pos_anch = torch.tensor(Pos_all[:anch_sample, :], dtype=torch.float32).to(device)

    ADS_test = torch.tensor(ADS_all[train_sample:, :, :, :], dtype=torch.float32)
    Pos_test = torch.tensor(Pos_all[train_sample:, :], dtype=torch.float32)


    train_dataset = TensorDataset(ADS_train, Pos_train)
    test_dataset = TensorDataset(ADS_test, Pos_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    learning_rate = 5e-3
    num_epochs = 1001
    # use GPU if available
    cuda_is_available = torch.cuda.is_available()

    # #Set the random seed for reproducible experiments
    # torch.manual_seed(231)
    # if cuda_is_available:
    #     torch.cuda.manual_seed(231)
    

    # Define the model and optimizer
    model = net.PosModel_ADS(ADS_anch, Pos_anch).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')


    # model parameters

    #transfer_learning = True
    transfer_learning = False
    if transfer_learning:
        model = net.PosModel_ADS(ADS_anch, Pos_anch).to(device)
        dict_path = r"D:\Huawei_competition_2024\demo\exp_icassp\ckp_scratch_scene3TUP_4.413085_epoch_600.pth"
        state_dict = torch.load(dict_path)
        model.load_state_dict(state_dict)


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn_ale


    # Train the model
    logging.info("Starting training for {} epoch(s)".format(num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, loss_fn, save_model=True)

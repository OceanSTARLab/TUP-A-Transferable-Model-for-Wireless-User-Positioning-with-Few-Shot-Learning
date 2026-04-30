"""data util"""

import numpy as np
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns




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


def save_model_dict(model, path):
    torch.save(model.state_dict(), path)



def read_pos_mat(Pos_path):
    data = scio.loadmat(Pos_path)
    Pos_all = np.array(data["Pos"]).astype(float)
    Anch_bool_all = np.array(data["Anch_bool"]).astype(int)
    return Pos_all, Anch_bool_all


def read_fea_mat(Fea_path):
    data = scio.loadmat(Fea_path)
    Feature_all = np.array(data["ang_delay_feature_all"])
    #Feature_all_phase = np.angle(Feature_all) / np.pi
    Feature_all_phase = np.sin(np.angle(Feature_all))
    return Feature_all_phase


def get_scene_train(Pos_non_anch1, Pos_anch1, Pos_non_anch2, Pos_anch2, Pos_non_anch3, Pos_anch3):
    s_non_anch1 = np.zeros_like(Pos_non_anch1)
    s_non_anch1 = s_non_anch1[:, 0]
    s_anch1 = np.zeros_like(Pos_anch1)
    s_anch1 = s_anch1[:, 0]

    s_non_anch2 = np.zeros_like(Pos_non_anch2)
    s_non_anch2 = s_non_anch2[:, 0]+1
    s_anch2 = np.zeros_like(Pos_anch2)
    s_anch2 = s_anch2[:, 0]+1

    s_non_anch3 = np.zeros_like(Pos_non_anch3)
    s_non_anch3 = s_non_anch3[:, 0]+2
    s_anch3 = np.zeros_like(Pos_anch3)
    s_anch3 = s_anch3[:, 0]+2
    return np.concatenate((s_non_anch1, s_anch1, s_non_anch2, s_anch2, s_non_anch3, s_anch3), axis=0)

def get_scene_train2(Pos_non_anch1, Pos_anch1, Pos_non_anch2, Pos_anch2, Pos_non_anch3, Pos_anch3, Pos_anch4):
    s_non_anch1 = np.zeros_like(Pos_non_anch1)
    s_non_anch1 = s_non_anch1[:, 0]
    s_anch1 = np.zeros_like(Pos_anch1)
    s_anch1 = s_anch1[:, 0]

    s_non_anch2 = np.zeros_like(Pos_non_anch2)
    s_non_anch2 = s_non_anch2[:, 0]+1
    s_anch2 = np.zeros_like(Pos_anch2)
    s_anch2 = s_anch2[:, 0]+1

    s_non_anch3 = np.zeros_like(Pos_non_anch3)
    s_non_anch3 = s_non_anch3[:, 0]+2
    s_anch3 = np.zeros_like(Pos_anch3)
    s_anch3 = s_anch3[:, 0]+2

    s_anch4 = np.zeros_like(Pos_anch4)
    s_anch4 = s_anch4[:, 0]+3
    return np.concatenate((s_non_anch1, s_anch1, s_non_anch2, s_anch2, s_non_anch3, s_anch3, s_anch4), axis=0)


def get_scene_train3(Pos_non_anch1, Pos_anch1, Pos_non_anch2, Pos_anch2, Pos_non_anch3, Pos_anch3, Pos_anch4, Pos_anch5, Pos_anch6):
    s_non_anch1 = np.zeros_like(Pos_non_anch1)
    s_non_anch1 = s_non_anch1[:, 0]
    s_anch1 = np.zeros_like(Pos_anch1)
    s_anch1 = s_anch1[:, 0]

    s_non_anch2 = np.zeros_like(Pos_non_anch2)
    s_non_anch2 = s_non_anch2[:, 0]+1
    s_anch2 = np.zeros_like(Pos_anch2)
    s_anch2 = s_anch2[:, 0]+1

    s_non_anch3 = np.zeros_like(Pos_non_anch3)
    s_non_anch3 = s_non_anch3[:, 0]+2
    s_anch3 = np.zeros_like(Pos_anch3)
    s_anch3 = s_anch3[:, 0]+2

    s_anch4 = np.zeros_like(Pos_anch4)
    s_anch4 = s_anch4[:, 0]+3

    s_anch5 = np.zeros_like(Pos_anch5)
    s_anch5 = s_anch5[:, 0]+4

    s_anch6 = np.zeros_like(Pos_anch6)
    s_anch6 = s_anch6[:, 0]+5
    return np.concatenate((s_non_anch1, s_anch1, s_non_anch2, s_anch2, s_non_anch3, s_anch3, s_anch4, s_anch5, s_anch6), axis=0)








def get_scene_train4(Pos_non_anch3, Pos_anch3, Pos_anch4):
    s_non_anch3 = np.zeros_like(Pos_non_anch3)
    s_non_anch3 = s_non_anch3[:, 0]
    s_anch3 = np.zeros_like(Pos_anch3)
    s_anch3 = s_anch3[:, 0]

    s_anch4 = np.zeros_like(Pos_anch4)
    s_anch4 = s_anch4[:, 0]+1
    return np.concatenate((s_non_anch3, s_anch3, s_anch4), axis=0)





def get_scene_test(Pos_non_anch1):
    s_non_anch1 = np.zeros_like(Pos_non_anch1)
    s_non_anch1 = s_non_anch1[:, 0]
    return s_non_anch1


def get_anch_tensor(Fea_anch_phase1, Pos_anch1):
    Fea_anch_phase_tensor1 = torch.tensor(Fea_anch_phase1, dtype=torch.float32)
    Fea_anch_phase_tensor1 = Fea_anch_phase_tensor1.unsqueeze(0).cuda()

    Pos_anch_tensor1 = torch.tensor(Pos_anch1, dtype=torch.float32)
    Pos_anch_tensor1 = Pos_anch_tensor1.unsqueeze(0).cuda()
    return Fea_anch_phase_tensor1,  Pos_anch_tensor1


def spilit_anch(Feature_all_phase, Pos_all, Anch_bool_all):
    Feature_anch_phase = Feature_all_phase[Anch_bool_all[:, 0] == 1]
    #Feature_anch_phase_tensor = torch.tensor(Feature_anch_phase, dtype=torch.float32)
    #Feature_anch_phase_tensor = Feature_anch_phase_tensor.unsqueeze(0).cuda()
    Pos_anch = Pos_all[Anch_bool_all[:, 0] == 1]
    #Pos_anch_tensor = torch.tensor(Pos_anch, dtype=torch.float32)
    #Pos_anch_tensor = Pos_anch_tensor.unsqueeze(0).cuda()

    Feature_non_anch_phase = Feature_all_phase[Anch_bool_all[:, 0] == 0]
    Pos_non_anch = Pos_all[Anch_bool_all[:, 0] == 0]
    return Feature_anch_phase, Pos_anch, Feature_non_anch_phase, Pos_non_anch





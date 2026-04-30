"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tltorch import *
import math
from sparsemax import Sparsemax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class Forward_Multihead_Attention(nn.Module):
    def __init__(self, patches, embeded_dim, key_size, num_heads, attn_drop=0.2):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.patches = patches
        self.num_heads = num_heads
        # self.wq = nn.Linear(embeded_dim, key_size, bias=True)
        # self.wk = nn.Linear(embeded_dim, key_size, bias=True)
        c = 2

        self.wq = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())
        self.wk = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())

        # self.wq = nn.Sequential(nn.Linear(embeded_dim, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())
        # self.wk = nn.Sequential(nn.Linear(embeded_dim, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())


        # self.wq = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=False), nn.ReLU(), nn.Linear(c*key_size, key_size, bias=False))
        # self.wk = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=False), nn.ReLU(), nn.Linear(c*key_size, key_size, bias=False))
        self.qk_head_dim = key_size//num_heads
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.sparsemax = Sparsemax(dim=2)
        self.softmax = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm2d(self.num_heads)
        self.LeakyRelu = nn.LeakyReLU()
    def forward(self, x, Anch):
        #x(B,12) Anch(2000,12)
        B1, N1 = x.shape
        B2, N2 = Anch.shape
        q = self.wq(input=x) # (B,hN)
        k = self.wk(input=Anch)# (2000,hN)

        Q = self.bn(q.reshape(1, B1,  self.num_heads, self.qk_head_dim).transpose(1, 2))
        K = self.bn(k.reshape(1, B2,  self.num_heads, self.qk_head_dim).transpose(1, 2))

        # Q = q.reshape(B1, N1, self.num_heads, self.qk_head_dim).transpose(1, 2)
        # K = k.reshape(B2, N2, self.num_heads, self.qk_head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)#
        [scores, _] = torch.max(scores, axis=1) # h B B2 -> B B2
        scores = self.sparsemax(scores)
        #scores = self.softmax(scores)
        return scores










class Forward_Multihead_Attention_light(nn.Module):
    def __init__(self, patches, embeded_dim, key_size, num_heads, attn_drop=0.3):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.patches = patches
        self.num_heads = num_heads
        c = 1
        self.wq = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=True), nn.BatchNorm1d(c*key_size),  nn.ReLU(), nn.Linear(c*key_size, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())
        self.wk = nn.Sequential(nn.Linear(embeded_dim, c*key_size, bias=True), nn.BatchNorm1d(c*key_size), nn.ReLU(), nn.Linear(c*key_size, c*key_size, bias=True), nn.BatchNorm1d(c*key_size),  nn.ReLU(), nn.Linear(c*key_size, key_size, bias=True), nn.BatchNorm1d(key_size), nn.Sigmoid())


        self.qk_head_dim = key_size//num_heads
        self.att_drop=nn.Dropout(attn_drop)
        self.bn=nn.BatchNorm2d(self.patches)
        self.sparsemax = Sparsemax(dim=2)

        self.bn = nn.BatchNorm2d(self.num_heads)
        self.LeakyRelu = nn.LeakyReLU()
    def forward(self, x, scene, anch_fea_dict):
        #x(B,12) Anch(2000,12)
        B1, N1, M1 = x.shape
        B2, N2, M2 = anch_fea_dict.shape #(3,2000,58)

        q = self.wq(input=x.squeeze(1)).unsqueeze(1) # (B,hN)
        k_dict = self.wk(input=anch_fea_dict.view(-1, M2)).view(B2, N2, -1)# (3 2000 hn)
        k = k_dict[scene, :, :]

        Q = self.bn(q.reshape(B1, N1, self.num_heads, self.qk_head_dim).transpose(1, 2))
        K = self.bn(k.reshape(B1, N2, self.num_heads, self.qk_head_dim).transpose(1, 2))

        #scores = torch.einsum("ijkl, ijlm->ijkm", Q, K.transpose(-2, -1))/ math.sqrt(self.qk_head_dim)#
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        [scores, ind] = torch.max(scores, axis=1) # h B B2 -> B B2
        scores = self.att_drop(scores / math.sqrt(self.qk_head_dim))
        scores = self.sparsemax(scores)
        return scores








class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):

        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.drop2 = nn.Dropout2d(0.1)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.drop1(out)))
        out = self.extra(self.drop2(x)) + out
        out = F.relu(out)

        return out




class Fea_extract_layer(nn.Module):

    def __init__(self, input_channel=1):
        super(Fea_extract_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=(2, 9), stride=(1, 8), padding=0),
            nn.BatchNorm2d(8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 4), stride=3, padding=0),
            nn.BatchNorm2d(8)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(8, 8, stride=1)
        self.blk2 = ResBlk(8, 6, stride=1)
        self.blk3 = ResBlk(6, 1, stride=1)


        latent_dim = 28
        self.BN = nn.BatchNorm1d(latent_dim)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.blk1(x))
        x = F.relu(self.blk2(x))
        x = F.relu(self.blk3(x))
        x = x.view(x.size(0), -1)
        x = self.BN(x)
        return x



class PosModel_ADS(nn.Module):
    """
    12->2: Find the corresponding position with the feature extracted from H.
    """
    def __init__(self, Anch_ads, Anch_pos,  num_heads=4):
        super(PosModel_ADS, self).__init__()

        self.anch_ads = Anch_ads #features of anchors (2000,C, H, W)
        self.anch_pos = Anch_pos #positions of anchors  (2000,2)


        B, C, H, W = Anch_ads.shape
        self.Anch_num = B
        self.embeded_dim = 28
        key_size = num_heads*self.embeded_dim

        self.fea_extract_layer = Fea_extract_layer(input_channel=1)
        self.PosAttention = Forward_Multihead_Attention(self.Anch_num, self.embeded_dim, key_size, num_heads)
        self.re = nn.Sequential(nn.Linear(2*self.embeded_dim, 16, bias=True),  nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 8, bias=True), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2, bias=True))




    def forward(self, input): # input (B, C, H, W)
        B, C, H, W = input.shape

        anch_fea = self.fea_extract_layer(self.anch_ads)
        input_fea = self.fea_extract_layer(input)
        scores = self.PosAttention(input_fea, anch_fea) # 1*512*2000



        scores = scores.expand(2, -1, -1)
        scores = torch.permute(scores, [1,2,0]) # 512*2000*2

        anch_fea = anch_fea.expand(B, -1, -1) #512*2000*58
        anch_pos = self.anch_pos.expand(B, -1, -1) # 512*2000*2

        unknow_fea = input_fea.expand(self.Anch_num, -1, -1)
        unknow_fea = torch.permute(unknow_fea, [1, 0, 2])

        fea_concat = torch.concat((unknow_fea, anch_fea), dim=2)

        
        res = self.re(fea_concat.view(-1, 2*self.embeded_dim)).view(B, self.Anch_num, -1) # 512*2000*2
        pred = torch.sum((res + anch_pos) * scores, dim=1)  #512*2
        #pred = torch.sum((anch_pos) * scores, dim=1)  #512*2

        return pred


def loss_fn_ale(outputs, ground_truth):
    # Calculate RMSE: outputs (B,2); ground_truth (B,2)
    mse = F.mse_loss(outputs, ground_truth, reduction='none')
    mse = torch.sqrt(torch.sum(mse, axis = 1))
    return  torch.mean(mse)







class CNN3D(nn.Module):
    def __init__(self, input_channel):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)


        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                               padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)


        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                               padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)


        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.max_pool3d(x, kernel_size=(1, 1, 4))  

        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.max_pool3d(x, kernel_size=(1, 1, 4)) 

  
        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.max_pool3d(x, kernel_size=(1, 1, 2))

        x = x.view(x.size(0), x.size(1), -1)  

        x, _ = torch.max(x, dim=2)
        x = self.fc(x)  
        return x






if __name__ == '__main__':

    # input = torch.randn([1,512,58]).cuda()
    # anch = torch.randn([1,2000, 58]).cuda()
    # anch_pos = torch.randn([1,2000, 2]).cuda()
    # model1 = PosModel3(anch, anch_pos).cuda()
    # output = model1(input)

    # input = torch.randn([512, 1, 58]).cuda()
    # scene = torch.ones([512, 1]).long().squeeze(1).cuda()
    #
    # anch1 = torch.randn([1,2000, 58]).cuda()
    # anch_pos1 = torch.randn([1,2000, 2]).cuda()
    # anch2 = torch.randn([1, 2000, 58]).cuda()
    # anch_pos2 = torch.randn([1, 2000, 2]).cuda()
    # anch3 = torch.randn([1, 2000, 58]).cuda()
    # anch_pos3 = torch.randn([1, 2000, 2]).cuda()
    #
    # model_multiscene = PosModel_Multiscene_light(anch1, anch_pos1, anch2, anch_pos2, anch3, anch_pos3).cuda()
    # out = model_multiscene(input, scene)

    input = torch.randn([64, 1, 32, 408]).cuda()
    anch_input = torch.randn([10, 1, 32, 408]).cuda()
    anch_pos = torch.randn([10, 2]).cuda()
    model = PosModel_ADS(anch_input, anch_pos).cuda()
    x = model(input)




    #plt.show()



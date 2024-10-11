import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat

# 在transfromer模型中为输入序列添加位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 用于分类的神经网络，
# 接收高光谱（h）和激光雷达（l）数据以及它们的位置编码（h_p, l_p），然后输出分类结果。
class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()

        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=128, stride=1, out_channels=512)
        self.conv_h = torch.nn.Conv2d(kernel_size=1, in_channels=60, stride=1, out_channels=128)
        self.conv_l = torch.nn.Conv2d(kernel_size=1, in_channels=60, stride=1, out_channels=128)
        self.linear1 = nn.Linear(in_features=512, out_features=args.num_classes)


    def forward(self, h, l, h_p, l_p):
        x = self.conv1(torch.cat((h, l), 1))
        x = F.avg_pool2d(x, kernel_size=5).reshape(-1, 512)
        x = self.linear1(x)
        return x.squeeze(-1).squeeze(-1)


import torch.nn.functional as F
import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from .cvt import ConvAttention, PreNorm, FeedForward
from .gcn import Spatial_Basic_Block
import numpy as np
import math


class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                 ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage),
                 FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CvT_layer(nn.Module):
    def __init__(self, image_size, in_channels, dim, heads,A, depth=1, kernels=1, strides=1, pad=0,
                  dropout=0., pooling=False, scale_dim=4):
        super().__init__()

        self.dim = dim
        ##### Stage 1 #######

        self.conv_embed = nn.Sequential(
                  Spatial_Basic_Block(in_channels, dim * heads, A),
                  Rearrange('b c h w -> b (h w) c', h=image_size[0], w=image_size[1]),
            )
        self.transformer = nn.Sequential(
                 Transformer(dim=dim*heads, img_size=(image_size[0], image_size[1]), depth=depth, heads=heads, dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
                 Rearrange('b (h w) c -> b c h w', h = image_size[0], w = image_size[1])
            )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x):

        x = self.conv_embed(x)
        x = self.transformer(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.p = p
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        x = torch.cat(torch.chunk(x, self.p, 2), 0)
        x = self.conv(x)
        x = torch.cat(torch.chunk(x, self.p, 0), 2)

        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h, w)

class single_p_block(nn.Module):
    def __init__(self, part_num, in_channels, out_channels):
        super(single_p_block, self).__init__()
        self.p = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels)
        )
        self.t = nn.Sequential(
            nn.Conv1d(in_channels*part_num, out_channels*part_num, 3, 1, 1,  groups=part_num, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels * part_num)
        )

    def forward(self, x):
           p, n, c, s = x.size()
           x = self.p(x.permute(1,3,2,0).contiguous().view(n*s, c, p)).view(n, s, c, p)
           x = self.t(x.permute(0,3,2,1).contiguous().view(n, p*c, s)).view(n, p, c, s)
           x = x.permute(1,0,2,3).contiguous()
           return x

class local_p_block(nn.Module):
    def __init__(self, part_num, in_channels, out_channels):
        super(local_p_block, self).__init__()
        self.p = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels)
        )
        self.t = nn.Sequential(
            nn.Conv1d(in_channels*part_num, out_channels*part_num, 3, 1, 1,  groups=part_num, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels * part_num)
        )

    def forward(self, x):
           p, n, c, s = x.size()
           x = self.p(x.permute(1,3,2,0).contiguous().view(n*s, c, p)).view(n, s, c, p)
           x = self.t(x.permute(0,3,2,1).contiguous().view(n, p*c, s)).view(n, p, c, s)
           x = x.permute(1,0,2,3).contiguous()
           return x

class global_p_block(nn.Module):
    def __init__(self, part_num, in_channels, out_channels):
        super(global_p_block, self).__init__()
        self.p = nn.Sequential(
            ConvAttention(2, (1, part_num), heads=1, dim_head=2, dropout=0, last_stage=False),
            # nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0, bias=False),
            # nn.ReLU(inplace=True)
            nn.Sigmoid()
        )
        self.t = nn.Sequential(
            nn.Conv1d(in_channels*part_num, out_channels*part_num, 3, 1, 1, groups=part_num, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels * part_num)
        )

    def forward(self, x):
        p, n, c, s = x.size()
        _x = torch.cat([torch.max(x, 2, keepdim=True)[0], torch.mean(x, 2, keepdim=True)],2)
        _x = _x.permute(1, 3, 0, 2).contiguous().view(n * s, p, 2)
        x = (self.p2(self.p(_x).permute(0, 2, 1).contiguous())
             .view(n, s, 1, p).contiguous() * x.permute(1, 3, 2, 0).contiguous()) + x.permute(1, 3, 2, 0).contiguous()
        x = self.t(x.permute(0, 3, 2, 1).contiguous().view(n, p * c, s)).view(n, p, c, s)
        x = x.permute(1, 0, 2, 3).contiguous()
        return x

class MCM(nn.Module):
    def __init__(self, part_num, in_channels, out_channels):
        super(MCM, self).__init__()
        self.s1 = single_p_block(part_num, in_channels, out_channels)
        self.s2 = single_p_block(part_num, in_channels, out_channels)

        self.l1 = local_p_block(part_num, in_channels, out_channels)
        self.l2 = local_p_block(part_num, in_channels, out_channels)

        self.g1 = global_p_block(part_num, in_channels, out_channels)
        self.g2 = single_p_block(part_num, in_channels, out_channels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)



    def forward(self, x):
        # p, n, c, s = x.size()
        s = self.s2(self.s1(x))
        l = self.l2(self.l1(x))
        g = self.g2(self.g1(x))
        s = torch.max(s, 3)[0]
        l = torch.max(l, 3)[0]
        g = torch.max(g, 3)[0]
        out = torch.cat([s, l, g], 0)
        return out

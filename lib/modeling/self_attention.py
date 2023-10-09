import copy
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn.modules import activation
import torch.nn.functional as F


# Q ske_feature  K img_feature V ske_feature
# 通过QKT找到ske与img中注意力最高的变量，对softmax进行求1的补集，来得到与img最不相似的mask
# 将mask与ske_feature进行内积，得到一个与img_feature相关度最低的骨骼特征
# 把这部分特征再拼接在img_feature进行补充

# multi_head_attention
class Attention(nn.Module):
    def __init__(self, dim, ratio=8, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.ratio = ratio
        self.head_dim = (dim//ratio) // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Conv1d(dim, self.head_dim*self.num_heads, 1,1,0,bias=qkv_bias)
        self.k = nn.Conv1d(dim, self.head_dim*self.num_heads, 1,1,0,bias=qkv_bias)
        if self.ratio == 1:
            self.v = nn.Identity()
            self.up = nn.Identity()
        else:
            self.v = nn.Conv1d(dim, self.head_dim * self.num_heads, 1, 1, 0, bias=qkv_bias)
            self.up = nn.Conv1d(self.head_dim * self.num_heads, dim, 1, 1, 0, bias=qkv_bias)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, query, key):
        query = query.permute(1, 2, 0).contiguous()
        key = key.permute(1, 2, 0).contiguous()
        q = self.q(query)
        n,c,part = q.size()
        q = q.view(n,self.num_heads,self.head_dim,part)
        k = self.k(key)
        n,c,ske = k.size()
        k = k.view(n,self.num_heads,self.head_dim,ske)
        v = self.v(key)
        v = v.view(n,self.num_heads,self.head_dim,ske)


        attn = (q.permute(0, 1, 3, 2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3).contiguous().\
            view(n,part,self.head_dim *self.num_heads).permute(0, 2, 1).contiguous()
        x = self.up(x).permute(2, 0, 1).contiguous()
        return x






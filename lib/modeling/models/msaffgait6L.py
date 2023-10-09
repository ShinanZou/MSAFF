import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..base_model import BaseModel
from ..modules import SeparateBNNecks, SetBlockWrapper, BasicConv2d
from ..basic_blocks import SetBlock, MCM, CvT_layer
from ..self_attention import Attention
from ..gcn import Graph

class MsaffGait6L(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):

        graph = Graph("coco")
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.hidden_dim = model_cfg['hidden_dim']
        self.part_img = model_cfg['part_img']
        self.part_ske = model_cfg['part_ske']
        _set_in_channels_img = model_cfg['set_in_channels_img']
        _set_in_channels_ske = model_cfg['set_in_channels_ske']
        _set_channels = model_cfg['set_channels']

        self.set_block1 = nn.Sequential(BasicConv2d(_set_in_channels_img, _set_channels[0], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(_set_channels[0], _set_channels[0], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(_set_channels[0], _set_channels[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(_set_channels[1], _set_channels[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.set_block3 = nn.Sequential(BasicConv2d(_set_channels[1], _set_channels[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(_set_channels[2], _set_channels[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.layer1 = SetBlock(CvT_layer(image_size=(1, 17), in_channels=_set_in_channels_ske, dim=_set_channels[0], heads=1, A=A, depth=1, kernels=1,
                                         strides=1, pad=0), pooling=False)
        self.layer2 = SetBlock(CvT_layer(image_size=(1, 17), in_channels=_set_channels[0], dim=_set_channels[0], heads=2, A=A, depth=2, kernels=1,
                                         strides=1, pad=0), pooling=False)
        self.layer3 = SetBlock(CvT_layer(image_size=(1, 17), in_channels=_set_channels[1], dim=_set_channels[0], heads=4, A=A, depth=2, kernels=1,
                                         strides=1, pad=0), pooling=False)
        
        self.set_pool0 = MCM(self.part_img,  _set_channels[2],  _set_channels[2])
        self.set_pool1 = MCM(self.part_ske,  _set_channels[2],  _set_channels[2])
        self.set_pool2 = MCM(self.part_img,  _set_channels[2],  _set_channels[2])

        self.atten = Attention(_set_channels[2])
        self.atten1 = Attention(_set_channels[2])

        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_img*3, _set_channels[2]*2, self.hidden_dim)))  #
        self.fc_bin1 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_img*3, _set_channels[2], self.hidden_dim)))  #
        self.fc_bin2 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_img*3, _set_channels[2], self.hidden_dim)))
        self.fc_bin3 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_ske*3, _set_channels[2], self.hidden_dim)))
        self.full = BasicConv2d(_set_channels[2]*2, _set_channels[2], 1, 1, 0)

    def hp(self, f):
        feature = f.mean(0) + f.max(0)[0]
        return feature

    def ske_hp(self, f, view):
            f = self.hp(f).expand(view.size())
            return f

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0][0]  
        pose = ipts[1][0]  

        x = sils.unsqueeze(2)
        y = pose.unsqueeze(2).permute(0, 1, 4, 2, 3)

        x_1_s = self.set_block1(x)
        x_1_s = self.set_block2(x_1_s)
        x_1_s = self.set_block3(x_1_s).permute(4, 3, 0, 2, 1)

        x_1_s = self.hp(x_1_s) # p n c s
        x_1 = self.set_pool0(x_1_s) # (96, 32, 128)
        ######  step 2  ##########
        y_1_s = self.layer3(self.layer2(self.layer1(y))).permute(4, 3, 0, 2, 1).contiguous().squeeze(1)
        y_1 = self.set_pool1(y_1_s)
        #
        x_2 = torch.cat([x_1,  self.atten1(x_1, y_1) + self.ske_hp(y_1, x_1)], 2)

        p,n,c,s = x_1_s.size()
        k,n,c,s = y_1_s.size()

        x_3 = torch.cat([x_1_s, self.atten(x_1_s.permute(0, 1, 3, 2).contiguous().view(p, n * s, c),
                         y_1_s.permute(0, 1, 3, 2).contiguous().view(k, n * s, c)).view(p, n, s, c).
                         permute(0, 1, 3, 2).contiguous() + self.ske_hp(y_1_s, x_1_s)], 2)
        x_3 = self.full(x_3.permute(1, 2, 0, 3).contiguous()).permute(2, 0, 1, 3).contiguous()
        x_3 = self.set_pool2(x_3)

        x_2 = x_2.matmul(self.fc_bin)
        x_3 = x_3.matmul(self.fc_bin1)
        x_1 = x_1.matmul(self.fc_bin2)
        x_4 = y_1.matmul(self.fc_bin3)

        embed_1 = torch.cat([x_1, x_2, x_3, x_4], 0)

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]

        n, s, c, h, w = x.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': x.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval




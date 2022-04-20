import torch.nn as nn
import torch
import torch.nn.functional as F

class Mish(nn.Module):
    """Mish 激活函数"""
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x

class ResBlock_v1(nn.Module):
    """
    CONV BN ACT
    Identity Mappings in Deep Residual Networks.
    """
    def __init__(self, nIn, nOut, kernel_size, stride,
                 activation=None):
        super(ResBlock_v1, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.kernel_size = kernel_size
        self.stride = stride
        if activation == 'relu':
            self.activ = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activ = nn.LeakyReLU()
        elif activation == 'Mish':
            self.activ = Mish()
        else:
            raise ValueError('activation 不能是 None')
        self.res_conv = nn.Conv2d(nIn, nOut, (1,1), [1,1], [0,0])
        # 两层卷积
        self.block = nn.Sequential()
        self._block(self.block, self.nIn, self.nOut, kernel_size, stride) # 第一层
        # self._block('1', self.block_1, self.nOut, self.nOut, kernel_size, stride) # 第二层

    def _block(self, nn_sequnsial, nIn, nOut, kernel_size=None, stride=None):
        padding = [0, 0]
        for _i, _k in enumerate(kernel_size):
            padding[_i] = (_k - stride[_i])//2  # stride == 1
        # conv
        nn_sequnsial.add_module('CNN_0', nn.Conv2d(nIn, nOut, kernel_size, stride, padding))
        # BN
        nn_sequnsial.add_module('Bn_0', nn.BatchNorm2d(nOut))
        # activ
        nn_sequnsial.add_module('Act_0', self.activ)
        # conv
        nn_sequnsial.add_module('CNN_1', nn.Conv2d(nOut, nOut, kernel_size, stride, padding))
        # BN
        nn_sequnsial.add_module('Bn_1', nn.BatchNorm2d(nOut))
        # activ
        nn_sequnsial.add_module('Act_1', self.activ)

    def forward(self, x):
        if self.nIn == self.nOut:
            res = x
        else:
            res = self.res_conv(x)
        # conv
        x = self.block(x)
        return x + res


class ResCNN_v2(nn.Module):
    """
    based on ResBlock_v2
    """
    def __init__(self, n_In, n_Out, activation=None, conv_dropout=0,
                 n_filter=None, pooling=None, kernel_size=None, stride=None):
        super(ResCNN_v2, self).__init__()
        self.ResBlock = ResBlock_v1
        self.PrePropess = nn.Sequential()
        self.ResBlocks = nn.Sequential()
        self.PostPropess = nn.Sequential()
        self.Mish = Mish()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # prepropess
        self.PrePropess.add_module('Bn_Pre', nn.BatchNorm2d(n_In))
        self.PrePropess.add_module('CNN_Pre', nn.Conv2d(n_In, n_filter[0], kernel_size=5, stride=2, padding=2))

        # res conv
        def generate_ResBlock(ResBlocks_Sq, activation="Mish", conv_dropout=0,
                              n_filter=None, pooling=None, kernel_size=None, stride=None):
            for _i in range(len(n_filter)-1):
                # channels of in and out
                nIn = n_filter[_i]
                nOut = n_filter[_i+1] if _i+1 < len(n_filter) else n_filter[_i]
                # res block
                ResBlocks_Sq.add_module('ResBlock_{}'.format(_i),
                                        self.ResBlock(nIn, nOut, kernel_size[_i], stride[_i], activation))
                # pool
                if pooling[_i] != 0:
                    ResBlocks_Sq.add_module('Pool_{}'.format(_i),
                                            nn.AvgPool2d(pooling[_i], pooling[_i]))
                # dropout
                if conv_dropout is not None:
                    ResBlocks_Sq.add_module('Dropout_{}'.format(_i), nn.Dropout(conv_dropout))
        generate_ResBlock(self.ResBlocks, activation, conv_dropout, n_filter, pooling, kernel_size, stride)

        # post bn
        self.PostPropess.add_module('Bn_Post_0', nn.BatchNorm2d(n_filter[-1]))
        # post conv
        self.PostPropess.add_module('CNN_Post', nn.Conv2d(n_filter[-1], n_filter[-1],
                                                             kernel_size=1, stride=1, padding=0))
        self.PostPropess.add_module('Bn_Post_2', nn.BatchNorm2d(n_filter[-1]))
        # GAP
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        # linear
        self.Last_linear = nn.Sequential(
            nn.Linear(n_filter[-1], n_filter[-1]), self.Mish,
            nn.Dropout(p=0.0), nn.Linear(n_filter[-1], n_filter[-1]))

    def load(self, state_dict=None):
        if state_dict is None:
            raise ValueError('state_dict is None')
        else:
            self.load_state_dict(state_dict=state_dict)

    def forward(self, x):
        x = self.PrePropess(x)
        x = self.ResBlocks(x)
        x = self.PostPropess(x)
        x = self.GAP(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.Last_linear(x)
        # x = self.sigmoid(x)
        return x


import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

class RobustModule(nn.Module):
    def __init__(self):
        super(RobustModule, self).__init__()
        self.r = True
    def robust(self):
        self.r = True
    def unrobust(self):
        self.r = False

class RobustLinear(RobustModule):
    def __init__(self, in_features, out_features, bias=True, non_negative = False):
        super(RobustLinear, self).__init__()
        self.base_type = nn.Linear(in_features = in_features, out_features = out_features, bias = True)
        self.in_features = in_features
        self.out_features = out_features
        if non_negative:
            self.weight = Parameter(torch.rand(out_features, in_features) * 1/math.sqrt(in_features))
        else:
            self.weight = Parameter(torch.randn(out_features, in_features) * 1/math.sqrt(in_features))
            
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.non_negative = non_negative

    def forward(self, input):
        if not self.r:
            return F.linear(input, self.weight, self.bias)
        input_p = input[:input.shape[0]//2]
        input_n = input[input.shape[0]//2:]
        if self.non_negative:
            out_p = F.linear(input_p, F.relu(self.weight), self.bias)
            out_n = F.linear(input_n, F.relu(self.weight), self.bias)
            return torch.cat([out_p, out_n], 0)
        
        u = (input_p + input_n)/2
        r = (input_p - input_n)/2
        out_u = F.linear(u, self.weight, self.bias)
        out_r = F.linear(r, torch.abs(self.weight), None)
        return torch.cat([out_u + out_r, out_u - out_r], 0)
    

class RobustConv2d(RobustModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, non_negative = False):
        super(RobustConv2d, self).__init__()
        self.base_type = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if non_negative:
            self.weight = Parameter(torch.rand(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        else:
            self.weight = Parameter(torch.randn(out_channels, in_channels//groups, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.non_negative = non_negative

    def forward(self, input):
        if not self.r:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        input_p = input[:input.shape[0]//2]
        input_n = input[input.shape[0]//2:]
        if self.non_negative:
            out_p = F.conv2d(input_p, F.relu(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
            out_n = F.conv2d(input_n, F.relu(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
            return torch.cat([out_p, out_n],0)
            
        u = (input_p + input_n)/2
        r = (input_p - input_n)/2
        out_u = F.conv2d(u, self.weight,self.bias, self.stride, self.padding, self.dilation, self.groups)
        out_r = F.conv2d(r, torch.abs(self.weight), None, self.stride, self.padding, self.dilation, self.groups)
        return torch.cat([out_u + out_r, out_u - out_r], 0)
    
class RobustConv2dTranspose(RobustModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True, non_negative = False):
        super(RobustConv2dTranspose, self).__init__()
        self.base_type = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=stride, padding=padding, output_padding=0, dilation=dilation, groups=groups, bias=bias)
        if non_negative:
            self.weight = Parameter(torch.rand(in_channels//groups, out_channels, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        else:
            self.weight = Parameter(torch.randn(in_channels//groups, out_channels, kernel_size, kernel_size) * 1/math.sqrt(kernel_size * kernel_size * in_channels//groups))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.non_negative = non_negative
        self.output_padding = output_padding

    def forward(self, input):
        if not self.r:
            return F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        input_p = input[:input.shape[0]//2]
        input_n = input[input.shape[0]//2:]
        if self.non_negative:
            out_p = F.conv_transpose2d(input_p, F.relu(self.weight), self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            out_n = F.conv_transpose2d(input_n, F.relu(self.weight), self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            return torch.cat([out_p, out_n],0)
            
        u = (input_p + input_n)/2
        r = (input_p - input_n)/2
        out_u = F.conv_transpose2d(u, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        out_r = F.conv_transpose2d(r, torch.abs(self.weight), None, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return torch.cat([out_u + out_r, out_u - out_r], 0)    
    
class ImageNorm(nn.Module):
    def __init__(self, mean, std):
        super(ImageNorm, self).__init__()
        self.mean = torch.from_numpy(np.array(mean)).view(1,3,1,1).cuda().float()
        self.std = torch.from_numpy(np.array(std)).view(1,3,1,1).cuda().float()
        
    def forward(self, input):
        input = torch.clamp(input, 0, 1)
        return (input - self.mean)/self.std
        
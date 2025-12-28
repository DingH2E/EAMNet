import torch
from torch import nn
from torch.nn import BatchNorm2d as BatchNorm
import torch.nn.functional as F
from einops import rearrange

import model.backbone.resnet as models
import model.backbone.vgg as vgg_models
from model.supmodel.dynamic_trans import DynamicTransformerLayer, GRN, RepConvBlock, SEModule
from model.supmodel.FPN import FPN
from model.comparable.DCP.ASPP import ASPP
from model.supmodel.g_GCN import g_GCN


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, classes=2):
        super(Generator, self).__init__()
        self.se = SEModule(in_dim)
        self.se_cell = SEModule(1)
        self.base = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
            nn.BatchNorm2d(out_dim),
        )
        self.cell = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.input_gate = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.output_gate = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.cell_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.cell_task = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        self.output_task = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        self.weight_cls = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_dim, classes, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.ones(1) * 5e-3)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.blocker = nn.BatchNorm2d(reduce_dim, eps=1e-04)
        self.blocker = nn.Sequential(
            nn.BatchNorm2d(1, eps=1e-04),
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-04),
        )
        self.sim_blocker = nn.BatchNorm2d(2, eps=1e-04)
        self.beta = 0.4

    def forward(self, x, y):
        weight_query = self.se(x) + x
        current_task = torch.cat([x, y], 1)  # current input
        weight_soft = self.weight_cls(weight_query).sum(1, True)
        base_gate = self.base(torch.mul(self.cell, current_task))
        input_gate = self.input_gate(base_gate)
        output_gate = self.output_gate(base_gate) + self.alpha * input_gate
        cell_gate = self.cell_gate(base_gate)
        cell = torch.add(torch.mul(cell_gate, self.cell), torch.mul(input_gate, self.cell_task(base_gate)))  # cell2
        cell = self.blocker(cell) + cell
        fuse_factor = self.blocker(
            torch.mul(self.output_task(cell), output_gate) * weight_soft) * self.beta
        return fuse_factor, weight_soft
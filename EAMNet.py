import torch
from torch import nn
from torch.nn import BatchNorm2d as BatchNorm
import torch.nn.functional as F
from einops import rearrange

import model.backbone.resnet as models
import model.backbone.vgg as vgg_models
from model.FPN import FPN
from model.generate_factor import Generator

# Initial Related
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        # print("Nx=",Nx.shape)
        # assert False
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.layers = args.layers
        assert self.layers in [50, 101, 152]
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = True
        self.classes = 2

        # Backbone Related
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 18:
                resnet = models.resnet18(pretrained=self.pretrained)
            elif self.layers == 34:
                resnet = models.resnet34(pretrained=self.pretrained)
            elif self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            # stage 0
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            # stage 1-4 from res-50
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # Dimension Related
        if self.vgg:
            fea_dim = 512 + 256
            reduce_dim = 128
        else:
            fea_dim = 1024 + 512
            reduce_dim = 256

        # Output Related
        out_channel = reduce_dim*2 + 3
        self.res1 = nn.Sequential(
            nn.Conv2d(out_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, args.classes, kernel_size=1)
        )


        self.fpn = FPN(channel=reduce_dim)
        self.grn = GRN(dim=reduce_dim)
        self.weight_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, args.classes, kernel_size=1),
        )
        self.alpha = args.alpha
        self.sim_blocker = nn.BatchNorm2d(2, eps=1e-04)
        self.beta = args.beta
        self.factor = Generator(in_dim=reduce_dim, out_dim=reduce_dim)



    def forward(self, x, y, s_x, s_y):
        """

        :return:
        """
        b, c, h, w = x.shape
        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)
        query_feat = self.fpn(query_feat_1, query_feat_2, query_feat_3)

        # Support Feature
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        supp_feat_3_list = []
        supp_feat_4_list = []
        masked_supp_feat_list = []
        supp_pro_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                # print("s_x=", s_x.shape)
                # assert False
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)

                supp_feat_4 = self.layer4(supp_feat_3)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            supp_feat = self.fpn(supp_feat_1, supp_feat_2, supp_feat_3)
            mask = F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear',
                                 align_corners=True)
            supp_grn = self.grn(supp_feat)
            supp_pro = Weighted_GAP(supp_grn, mask)
            supp_feat_list.append(supp_feat)
            supp_feat_3_list.append(supp_feat_3)
            supp_feat_4_list.append(supp_feat_4)
            masked_supp_feat_list.append(supp_feat * mask)
            supp_pro_list.append(supp_pro)

        supp_feat = supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_list)):
                supp_feat = supp_feat + supp_feat_list[i]
            supp_feat = supp_feat / len(supp_feat_list)
        supp_feat_3 = supp_feat_3_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_3_list)):
                supp_feat_3 = supp_feat_3 + supp_feat_3_list[i]
            supp_feat_3 = supp_feat_3 / len(supp_feat_3_list)
        supp_feat_4 = supp_feat_4_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_4_list)):
                supp_feat_4 = supp_feat_4 + supp_feat_4_list[i]
            supp_feat_4 = supp_feat_4 / len(supp_feat_4_list)
        masked_supp = masked_supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(masked_supp_feat_list)):
                masked_supp += masked_supp_feat_list[i]
            masked_supp /= len(masked_supp_feat_list)

        supp_pro = torch.cat(supp_pro_list, 2).sum(2).unsqueeze(2)
        concat_feat = supp_pro.expand_as(query_feat)
        if self.shot == 1:
            similarity2 = get_similarity(query_feat_3, supp_feat_3, s_y)
            similarity1 = get_similarity(query_feat_4, supp_feat_4, s_y)
        else:
            similarity1 = [get_similarity(query_feat_4, supp_feat_4, mask=(s_y[:, i, :, :] == 1).unsqueeze(1)) for i in range(self.shot)]
            similarity2 = [get_similarity(query_feat_3, supp_feat_3, mask=(s_y[:, i, :, :] == 1).unsqueeze(1)) for i in range(self.shot)]
            similarity2 = torch.stack(similarity2, dim=1).mean(1)
            similarity1 = torch.stack(similarity1, dim=1).mean(1)

        if not self.vgg:
            similarity1 = F.interpolate(similarity1, size=(similarity2.size(2), similarity2.size(3)),
                                        mode='bilinear', align_corners=True)

        similarity = self.sim_blocker(torch.cat([similarity1, similarity2], dim=1)) * self.beta
        similarity = F.interpolate(similarity, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                   align_corners=True)
        query_feat_nom = self.grn(query_feat)

        factor, weight_soft = self.factor(query_feat, supp_feat)

        query_feat = torch.cat([query_feat_nom, concat_feat, factor, similarity], 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        query_feat = self.res3(query_feat)
        out = self.cls(query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        # Loss
        if self.training:
            weight_soft = F.interpolate(weight_soft, size=(h, w), mode='bilinear', align_corners=True).squeeze(1)
            main_loss = self.criterion(out, y.long())
            kd_loss = self.disstil_loss(y.float(), weight_soft.float())
            return out.max(1)[1], main_loss + kd_loss * self.alpha
        else:
            return out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss
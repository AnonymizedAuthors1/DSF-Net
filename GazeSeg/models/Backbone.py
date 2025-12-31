import torch
import torch.nn as nn
import timm
from GazeSeg.models.ViGUNet import ViG
from torch.nn import functional as F


class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True,
                          out_indices=(0, 1, 2, 3), pretrained_cfg_overlay=dict(file="D:\myPyProject/11-GazeTrajectoryPred/GazeTrajectorySegmentation/GazeSeg\models\pretrained_model\swin_tiny_patch4_window7_224.pth"))
        # self.proj = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in self.backbone.feature_info.channels()])

    def forward(self, x):
        feats = self.backbone(x)
        # feats = [p(f.permute(0, 3, 1, 2)) for p,f in zip(self.proj, feats)]
        feats = [f.permute(0, 3, 1, 2) for f in feats]
        return feats


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True,
                                          pretrained_cfg_overlay=dict(
                                              file="D:\myPyProject/11-GazeTrajectoryPred/GazeTrajectorySegmentation/GazeSeg\models\pretrained_model/resnet50_a1_0-14fe96d1.pth"))
    def forward(self, x):
        feats = self.backbone(x)
        return feats

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, features_only=True,
                        pretrained_cfg_overlay=dict(file="D:\myPyProject/11-GazeTrajectoryPred/GazeTrajectorySegmentation/GazeSeg\models\pretrained_model\S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"))
    def forward(self, x):
        feats = self.backbone(x)
        return feats

from torch.nn import Sequential as Seq, Conv2d


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), nn.BatchNorm2d(hidden_features))
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), nn.BatchNorm2d(out_features),)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + shortcut
        return x


class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 2), nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return self.conv(x)


class Grapher(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, r=1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels))
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, r)
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = x + _tmp
        return x


class GraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(GraphConv2d, self).__init__()
        self.gconv = MRConv2d(in_channels, out_channels)

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class BasicConv(Seq):
    def __init__(self, channels):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=True, groups=4))
            m.append(nn.BatchNorm2d(channels[-1], affine=True))
            m.append(nn.GELU())
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels])

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        x = self.nn(x)
        return x


class DyGraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels)
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x, y=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation)
        return self._dilated(edge_index)


def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.k = k

    def forward(self, edge_index):
        edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class ViGBackbone(torch.nn.Module):
    def __init__(self):
        super(ViGBackbone, self).__init__()
        k = 9
        blocks = [2, 2, 6, 2]

        channels = [80, 160, 400, 640]
        reduce_ratios = [4, 2, 1, 1]
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        self.stem = Stem(out_dim=channels[0])
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], k, idx // 4 + 1, reduce_ratios[i]),
                        FFN(channels[i], channels[i] * 4))]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        c = inputs.size(1)
        if c == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        elif c == 3:
            inputs = inputs
        else:
            raise ValueError('Wrong input channel: {}'.format(c))
        x = self.stem(inputs) + self.pos_embed
        feature_list = []
        # feature_list.append(x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [1, 4, 11, 14]:
                feature_list.append(x)
        return feature_list




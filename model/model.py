from functools import partial
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.layers import trunc_normal_

torch.backends.cudnn.benchmark = True
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
from spconv.pytorch.modules import SparseModule
import torch.nn.init as init
import functools

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None, ks=3, pd=1):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(   # SubMConv3d 是 submanifold 卷积
                spconv.SubMConv3d(in_channels, out_channels,  kernel_size=ks, padding=pd, bias=True, indice_key=indice_key)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=ks, padding=pd, bias=True, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=ks, padding=pd, bias=True, indice_key=indice_key)
        )

    def forward(self, input):

        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(input).features)

        return output

class rec(nn.Module):
    def __init__(self):
        super(rec, self).__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        channel_m = 8

        self.start_head = spconv.SparseSequential(
            spconv.SubMConv3d(1, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
        )

        self.path0_0 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),

        )

        self.downsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample0")
        )

        self.path1_0 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1),

        )

        self.downsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample1")
        )

        self.path2_0 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1),

        )

        self.downsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample2")
        )

        self.path3 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm3", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm3", ks=3, pd=1)
        )

        self.upsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample2")
        )

        self.path2_1 = spconv.SparseSequential(
            ResidualBlock(channel_m * 2, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1)
        )

        self.upsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample1")
        )

        self.path1_1 = spconv.SparseSequential(
            ResidualBlock(channel_m * 2, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1)
        )

        self.upsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample0")
        )

        self.path0_1 = spconv.SparseSequential(
            ResidualBlock(channel_m * 2, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1)
        )

        self.end_tail = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, 1, kernel_size=3, padding=1, bias=True, indice_key="subm0"),

        )

        self.to_dense = spconv.ToDense()

    def forward(self, inputs):
        raw_dense = inputs.squeeze(4)

        inputs_sp = spconv.SparseConvTensor.from_dense(inputs)

        start_out = self.start_head(inputs_sp)

        out0 = self.path0_0(start_out)

        ds_out0 = self.downsample0(out0)

        out1 = self.path1_0(ds_out0)

        ds_out1 = self.downsample1(out1)

        out2 = self.path2_0(ds_out1)

        ds_out2 = self.downsample2(out2)

        out3 = self.path3(ds_out2)

        up_out3 = self.upsample2(out3)

        out2 = out2.replace_feature(torch.cat((out2.features, up_out3.features), dim=1))

        out2 = self.path2_1(out2)

        up_out2 = self.upsample1(out2)

        out1 = out1.replace_feature(torch.cat((out1.features, up_out2.features), dim=1))

        out1 = self.path1_1(out1)

        up_out1 = self.upsample0(out1)

        out0 = out0.replace_feature(torch.cat((out0.features, up_out1.features), dim=1))

        out0 = self.path0_1(out0)

        out = self.end_tail(out0)

        spatial_shape = out.spatial_shape
        batch_size = out.batch_size
        mask = out.features > 0
        features_th = torch.masked_select(inputs_sp.features, mask)
        features_th = features_th.view(-1, 1)
        indices_th = torch.masked_select(inputs_sp.indices, mask)
        indices_th = indices_th.view(-1, 4)
        del out
        out = spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)
        out_dense = self.to_dense(out)

        return out, out_dense


class Dilation(nn.Module):
    def __init__(self):
        super(Dilation, self).__init__()

        self.point_expansion = spconv.SparseSequential(
            spconv.SparseConv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False, indice_key="point1"),
            spconv.SparseConv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False, indice_key="point2"),
            spconv.SparseConv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False, indice_key="point3"),

        )
        init.constant_(self.point_expansion[0].weight, 1.0)
        init.constant_(self.point_expansion[1].weight, 1.0)
        init.constant_(self.point_expansion[2].weight, 1.0)

        self.plane_expansion = spconv.SparseSequential(
            spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane1"),
            spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane2"),
            spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane3"),
            spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane4"),

            spconv.ToDense()
        )
        init.constant_(self.plane_expansion[0].weight, 1.0)
        init.constant_(self.plane_expansion[1].weight, 1.0)
        init.constant_(self.plane_expansion[2].weight, 1.0)
        init.constant_(self.plane_expansion[3].weight, 1.0)

    def forward(self, inputs):
        point_eout = self.point_expansion(inputs)
        plane_eout = self.plane_expansion(point_eout)

        return plane_eout


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_channels,
            norm_fn=None,
            indice_key=None,
            depth=4,
            groups=None,
            grid_size=None,
            bias=False,
    ):
        super().__init__()

        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        )
        self.centroid_gate = nn.ModuleList()

        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

            self.centroid_gate.append(
                nn.Sequential(
                    nn.Linear(embed_channels, 1, bias=False)

                )
            )

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.voxel_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)

            scores = self.centroid_gate[i](feat)

            scores_max = scatter(scores, cluster, reduce="max")[cluster]
            scores_exp = torch.exp(scores - scores_max)

            scores_sum = scatter(scores_exp, cluster, reduce="sum")[cluster] + 1e-6
            attention_weights = scores_exp / scores_sum

            weighted_pw = pw * attention_weights

            pw = pw - scatter(weighted_pw, cluster, reduce="sum")[cluster]
            pw = self.weight[i](pw)
            pw_max = scatter(pw, cluster, reduce="max")[cluster]
            pw = torch.exp(pw - pw_max)
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)

            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        return x


class DonwBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_channels,
            depth,
            sp_indice_key,
            point_grid_size,
            num_ref=16,
            groups=None,
            norm_fn=None,
            sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            embed_channels,
            depth,
            sp_indice_key,
            norm_fn=None,
            down_ratio=2,
            sub_indice_key=None,
    ):
        super().__init__()
        assert depth > 0
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                embed_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(
            self.fuse(torch.cat([x.features, skip_x.features], dim=1)) + x.features
        )
        return x


class OACNNs(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            embed_channels=32,
            enc_num_ref=[16, 16, 16],
            enc_channels=[64, 128, 256],
            groups=[2, 4, 8],
            enc_depth=[1, 2, 3],
            down_ratio=[2, 2, 2],
            dec_channels=[32, 64, 128],

            point_grid_size=[[4, 8, 16], [2, 4, 8], [1, 2, 4]],
            dec_depth=[2, 2, 2, 2],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                )
            )
            self.dec.append(
                UpBlock(
                    in_channels=(
                        enc_channels[-1]
                        if i == self.num_stages - 1
                        else dec_channels[i + 1]
                    ),
                    skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=dec_channels[i],
                    depth=dec_depth[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i}",
                )
            )

        self.final = spconv.SubMConv3d(dec_channels[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)
        self.to_dense = spconv.ToDense()

    def forward(self, input_dict):

        input_dict = input_dict.replace_feature(input_dict.features - 1)

        x = self.stem(input_dict)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        x = self.final(x)
        x = self.to_dense(x)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

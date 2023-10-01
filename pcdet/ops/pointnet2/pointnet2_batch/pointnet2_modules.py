from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
# from .Attention_2 import *


class BatchNorm(nn.BatchNorm1d):

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = super().forward(x)
        x = x.reshape(*shape)
        return x

class LCEF(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim, in_pts):
        super(LCEF, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_pts // 8,device="cuda")
        self.bn2 = nn.BatchNorm1d(in_pts // 8,device="cuda")
        self.bn3 = nn.BatchNorm1d(in_dim,device="cuda")
        self.bn = nn.BatchNorm1d(in_dim, affine=True,device="cuda")

        self.query_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_pts, out_channels=in_pts // 8, kernel_size=1, bias=False,device="cuda"),
            self.bn1,
            nn.ReLU())
        self.key_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_pts, out_channels=in_pts // 8, kernel_size=1, bias=False,device="cuda"),
            self.bn2,
            nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False,device="cuda"),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1)).cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.squeeze = nn.AdaptiveMaxPool1d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_dim, in_dim // 8, bias=False,device="cuda"),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 8, in_dim, bias=False,device="cuda"),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        # proj_value = self.value_conv(x)
        ###   se attention
        proj_value = x
        b, C, _ = proj_value.shape
        proj_value = self.squeeze(proj_value).view(b, C)
        proj_value = self.excitation(proj_value).view(b, C, 1)
        proj_value = x * proj_value.expand_as(x)

        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight

        out = self.alpha * out

        out = out + x

        return out


class PP3S1(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(PP3S1, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.avg_pool = nn.MaxPool1d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // rotio, 1, bias=False,device="cuda"),
            nn.ReLU(),
            nn.Conv1d(in_planes // rotio, in_planes, 1, bias=False,device="cuda"))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class PP3S2(nn.Module):
    def __init__(self, channels, t=16):
        super(PP3S2, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm1d(self.channels, affine=True,device="cuda")

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 2, 1).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz: (B, npoint, 3) tensor of the xyz coordinates of the grouping centers if specified
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float()
            idx_cnt_mask = idx_cnt_mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            new_features *= idx_cnt_mask
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.radii = []

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None
        self.confidence_mlp2 = None

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None,
                # cls_features: torch.Tensor = None
                ):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            a = len(self.sample_method_list)
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':

                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 'ry_FPS' or self.sample_method_list[i] == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for j in range(len(xyz_slice)):
                        per_xyz = xyz_slice[j]
                        ry = torch.atan(per_xyz[:, 0] / per_xyz[:, 1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1, 3)

                        per_idx_div = indince.view(part_num, -1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div, dim=0)
                    idx_div = torch.cat(idx_div, dim=0)
                    npointnpoint = self.npoint_list[i]
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, npointnpoint // part_num)

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npointnpoint).int()
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )

                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)

            if self.skip_connection:
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        for i in range(len(self.groupers)):

            ######################################################################
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            ######################################################################
            linear_features = new_features

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)


            idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            new_features *= idx_cnt_mask

            if self.pool_method == 'max_pool':

                pooled_features = torch.max(new_features, dim=-1)[0]
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            # new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

            if self.pre_aggregation_layers is not None or self.radii[1] == 6.4:
                new_features = self.pre_aggregation_layers[i](pooled_features.transpose(1, 2)).transpose(1, 2)

            # new_features_list.append(pooled_features)
            new_features_list.append(new_features)
        new_features = sum(new_features_list)

        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        if self.aggregation_bn_relu is not None:
            new_features = self.aggregation_bn_relu(new_features)

        # new_features = torch.cat(new_features_list, dim=1)

        psa1 = LCEF(in_dim=new_features.size(1), in_pts=new_features.size(2))
        new_features = psa1(new_features)

        if self.confidence_mlp is not None:
            xxx_feature = new_features

            se = PP3S1(in_planes=xxx_feature.size(1))
            c = se(xxx_feature)
            xxx_feature = torch.mul(xxx_feature, c)

            new_features = new_features + xxx_feature
            sss = nn.ReLU()
            new_features = sss(new_features)

            nam = PP3S2(new_features.size(1))
            new_features = nam(new_features)

            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_scores

        return new_xyz, new_features, None


class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 pre_aggregation_mlp: bool = False,
                 confidence_mlp: List[int] = None,
                 # confidence_mlp2: List[int] = None,
                 ):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.mlps3 = nn.ModuleList()

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                ### this
                self.groupers.append(
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                )
            former_radius = radius
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlp = []
            shared_mlp_3 = []
            for k in range(len(mlp_spec) - 1):
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            # for k in range(len(mlp_spec) - 1):
            #     if k == 0:
            #         shared_mlp.extend([
            #             nn.Linear(mlps[0][0], mlp_spec[-1], bias=False),
            #             nn.ReLU()
            #         ])
            #     else:
            #         shared_mlp.extend([
            #             nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
            #             nn.BatchNorm2d(mlp_spec[k + 1]),
            #             nn.ReLU()
            #         ])
            shared_mlp_3.extend([
                nn.Linear(mlps[0][0], mlp_spec[-1],
                          bias=False),
                nn.ReLU()
            ])
            self.mlps.append(nn.Sequential(*shared_mlp))
            self.mlps3.append(nn.Sequential(*shared_mlp_3))
            in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        # if aggregation_mlp is not None:
        #
        #     shared_mlp = []
        #     for k in range(len(aggregation_mlp)):
        #         shared_mlp.extend([
        #             nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
        #             nn.BatchNorm1d(aggregation_mlp[k]),
        #             nn.ReLU()
        #         ])
        #         out_channels = aggregation_mlp[k]
        #
        #     self.aggregation_mlp = nn.Sequential(*shared_mlp)
        # else:
        #     self.aggregation_mlp = None

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0) or self.radii[
            1] == 6.4:
            if pre_aggregation_mlp:
                ######################################################################
                self.aggregation_layer = None
                self.pre_aggregation_layers = nn.ModuleList()
                for i in range(len(radii)):
                    shared_mlp = []
                    out_channels = mlps[i][-1]
                    for k in range(len(aggregation_mlp)):
                        if k < len(aggregation_mlp) - 1:
                            shared_mlp.extend([
                                nn.Linear(out_channels, aggregation_mlp[k], bias=False),
                                BatchNorm(aggregation_mlp[k]),
                                nn.ReLU()
                            ])
                        else:
                            shared_mlp.extend([
                                nn.Linear(out_channels, aggregation_mlp[k]),
                            ])
                        out_channels = aggregation_mlp[k]
                    self.pre_aggregation_layers.append(nn.Sequential(*shared_mlp))
            #######################################################################
            elif self.radii[1] == 6.4:
                aggregation_mlp = [256]
                self.aggregation_layer = None
                self.pre_aggregation_layers = nn.ModuleList()
                for i in range(len(radii)):
                    shared_mlp = []
                    out_channels = mlps[i][-1]
                    for k in range(len(aggregation_mlp)):
                        if k < len(aggregation_mlp) - 1:
                            shared_mlp.extend([
                                nn.Linear(out_channels, aggregation_mlp[k], bias=False),
                                BatchNorm(aggregation_mlp[k]),
                                nn.ReLU()
                            ])
                        else:
                            shared_mlp.extend([
                                nn.Linear(out_channels, aggregation_mlp[k]),
                            ])
                        out_channels = aggregation_mlp[k]
                    self.pre_aggregation_layers.append(nn.Sequential(*shared_mlp))
            self.aggregation_bn_relu = nn.Sequential(
                nn.BatchNorm1d(aggregation_mlp[-1]),
                nn.ReLU()
            )
        else:
            self.aggregation_layer = None
            self.aggregation_bn_relu = None

        if confidence_mlp is not None:
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 1, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
        else:
            self.confidence_mlp = None


class PointnetSAModuleFS(PointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 pre_aggregation_mlp: bool = False,
                 confidence_mlp: List[int] = None,
                 # confidence_mlp2: List[int] = None
                 ):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=[radius], nsamples=[nsample],
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, pre_aggregation_mlp=pre_aggregation_mlp, confidence_mlp=confidence_mlp
            # ,confidence_mlp2=confidence_mlp2
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
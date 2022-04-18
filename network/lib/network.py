import torch
import torch.nn as nn
from network.lib.pspnet import PSPNet


class DeformNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024):
        # 关于nv,可能是因为论文中观测到的点云用V指代
        # 论文中提到观测到点云V ∈ (Nvx3)
        # 类先验点云 Mc ∈ (Ncx3)
        # 输出的形变场是Ncx3
        # 输出的corresponding matrix是Nv x Nc
        # 因此M = Mc + D   (Nc, 3)
        # A*M得到(Nv,3)的点云,可以与nocs进行比对差异
        super(DeformNet, self).__init__()
        self.n_cat = n_cat  # 类的数量
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')  # 提取实例
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.instance_geometry = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def forward(self, points, img, choose, cat_id, prior):
        """
        Args:
            points: bs x n_pts x 3  仅实例的点云
            img: bs x 3 x H x W     裁剪的实例图像rgb
            choose: bs x n_pts      rgb中属于物体的像素下标(1维)
            cat_id: bs              类标签
            prior: bs x nv x 3      类先验模型(来自mean_shape)

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        bs, n_pts = points.size()[:2]  # n_pts 点云中点的数量
        nv = prior.size()[1]  # 类先验中点的数量
        # instance-specific features
        points = points.permute(0, 2, 1)  # 为了卷积交换后两个维度
        # points: (bs, 3, n_pts)
        # points: (bs, 64, n_pts)
        points = self.instance_geometry(points)
        # img (bs, 3, H, W)
        # out_img
        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        # 处理rgb,32->64
        emb = self.instance_color(emb)
        # 将图像特征与颜色特征进行拼接
        # (bs, 64, n_pts) cat (bs, 64, n_pts) -> (bs, 128, n_pts)
        inst_local = torch.cat((points, emb), dim=1)     # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1
        # category-specific features
        # 类先验
        # (bs,nv,3)
        cat_prior = prior.permute(0, 2, 1)  # (bs,3,nv)
        cat_local = self.category_local(cat_prior)    # bs x 64 x nv
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1
        # 融合实例点云和rgb特征的inst_local (bs,128,n_pts)
        # 描述实例点云的全局特征inst_global (bs,1024,1)
        # 描述类先验的全局特征cat_global (bs,1024,1)
        # 为了拼接到一起,所以需要将两个_global进行repeat
        # 最终得到assign_feat (bs,2176,n_pts)
        # 计算Correspondence Matrix
        # bs x 2176 x n_pts
        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)
        # assign_mat (bs, n_cat*nv_prior, n_pts)
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv

        # deformation field
        # 计算形变场
        # 和assign_mat不同,其主体是cat_local (bs,64,nv)
        # 64+1024+1024=2112
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)  # (bs, n_cat*3, nv)
        # (bs, n_cat*3, nv) -> (bs*n_cat, 3, nv)
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        # 从deltas中,取第0维中,下标为index的点
        # (bs*n_cat, 3, nv)  ->  (bs, 3, nv)
        # index????
        # 为什么需要n_cat??
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return assign_mat, deltas

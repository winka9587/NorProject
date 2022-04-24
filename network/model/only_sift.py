import torch
import torch.nn as nn
import torch.nn.functional as F

class SIFT_Track(nn.Module):
    def __init__(self, emb_dim, n_pts, subseq_len=2):
        super(SIFT_Track, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, 512)
        # self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 3*n_pts)
        self.subseq_len = subseq_len

    def forward(self, data):
        """
        Args:
            data: list
            (B, 512)

        """
        # �����data��Ϊ������, ��һ֡�ͺ���֡
        # 1.��ʼ֡
        # �Ƿ�������� if else
        first_frame_data = data[0]


        # bs = embedding.size()[0]
        # out = F.relu(self.fc1(embedding))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # out_pc = out.view(bs, -1, 3)
        # return out_pc


    def set_data(self, data):
        # ��ȡ��Ҫ������,��ת�Ƶ�cpu,��ʹ���µ��ֵ����洢
        pass

    def update(self):
        # ����forward,������loss
        # self.forward(save=save)
        # if not no_eval:
        #     self.compute_loss(test=True, per_instance=save, eval_iou=True, test_prefix='test')
        # else:
        #     self.loss_dict = {}
        pass



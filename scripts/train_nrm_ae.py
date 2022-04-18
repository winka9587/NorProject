import os
import time
import argparse
import torch
import tensorflow as tf
from network.lib.auto_encoder import PointCloudAE
from network.lib.loss import ChamferLoss
from data.shape_dataset import ShapeDataset
from network.lib.utils import setup_logger
from configs.base_config import get_base_config
from os.path import join as pjoin
from utils import ensure_dir
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# 数据集中h5文件所在路径,原为'data/obj_models/ShapeNetCore_4096.h5'


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='number of points, needed if use points')
parser.add_argument('--emb_dim', type=int, default=512, help='dimension of latent embedding [default: 512]')
parser.add_argument('--h5_file', type=str, default=dataset_path, help='h5 file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6', help='GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train')
parser.add_argument('--resume_model', type=str, default='', help='resume from saved model')
parser.add_argument('--result_dir', type=str, default='results/ae_points', help='directory to save train results')
opt = parser.parse_args()

# 设置使用哪些cpu
# 如果用第1,2号显卡,则 os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
print(f'set CUDA_VISBLE_DEVICES={opt.gpu}')
device_ids = [0]
print(f'use GPU {device_ids}')

opt.repeat_epoch = 10
opt.decay_step = 5000
opt.decay_rate = [1.0, 0.6, 0.3, 0.1]

def train_net():
    print('ready dir to save result')
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    tb_writer = tf.summary.FileWriter(opt.result_dir)  # tensorflow 1.12.0
    # tb_writer = tf.summary.create_file_writer(opt.result_dir)  # tensorflow 2.6.xx
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))


    # model & loss
    # emb_dim和num_point的关系
    estimator = PointCloudAE(opt.emb_dim, opt.num_point)
    # try multi GPU training
    estimator = torch.nn.DataParallel(estimator, device_ids)
    estimator.cuda()
    criterion = ChamferLoss()
    if opt.resume_model != '':
        estimator.load_state_dict(torch.load(opt.resume_model))

    print('ready dataset')

    # dataset
    # ShaepDataset的默认npoints是2048,这里都没有显式设置,虽然读取的是4096的h5,但最终使用的就是2048
    train_dataset = ShapeDataset(opt.h5_file, mode='train', augment=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.num_workers)
    val_dataset = ShapeDataset(opt.h5_file, mode='val', augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)

    print('start train')
    # train
    st_time = time.time()
    global_step = ((train_dataset.length + opt.batch_size - 1) // opt.batch_size) * opt.repeat_epoch * (
                opt.start_epoch - 1)
    decay_count = -1
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate if needed
        if global_step // opt.decay_step > decay_count:
            decay_count += 1
            if decay_count < len(opt.decay_rate):
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
        batch_idx = 0
        estimator.train()
        for rep in range(opt.repeat_epoch):
            print(f'repeat {rep}/{opt.repeat_epoch}')
            for i, data in enumerate(train_dataloader):
                # label must be zero_indexed
                # 默认batch_size是32
                # batch_xyz是(32, 2048, 3), batch_label(32,)
                batch_xyz, batch_label = data
                batch_xyz = batch_xyz[:, :, :3].cuda()
                optimizer.zero_grad()
                embedding, point_cloud = estimator(batch_xyz)
                loss, _, _ = criterion(point_cloud, batch_xyz)  # 计算自编码器恢复的点云point_cloud和输入点云batch_xyz的差异loss
                summary = tf.Summary(value=[tf.Summary.Value(tag='learning_rate', simple_value=current_lr),
                                            tf.Summary.Value(tag='train_loss', simple_value=loss)])
                # backward
                loss.backward()
                optimizer.step()
                global_step += 1
                batch_idx += 1
                # write results to tensorboard
                tb_writer.add_summary(summary, global_step)
                if batch_idx % 10 == 0:
                    logger.info('Batch {0} Loss:{1:f}'.format(batch_idx, loss))
        print('train finish')
        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))
        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))
        estimator.eval()
        val_loss = 0.0
        for i, data in enumerate(val_dataloader, 1):
            batch_xyz, batch_label = data
            batch_xyz = batch_xyz[:, :, :3].cuda()
            embedding, point_cloud = estimator(batch_xyz)
            loss, _, _ = criterion(point_cloud, batch_xyz)
            val_loss += loss.item()
            logger.info('Batch {0} Loss:{1:f}'.format(i, loss))
        val_loss = val_loss / i
        summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss)])
        tb_writer.add_summary(summary, global_step)
        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        # save model after each epoch
        torch.save(estimator.state_dict(), '{0}/model_{1:02d}.pth'.format(opt.result_dir, epoch))


if __name__ == '__main__':
    print('traing nmap ae ...')
    base_cfg = get_base_config()
    dataset_path = base_cfg['dataset_path']
    save_path = base_cfg['result_path']

    # 最终是为了得到model_50.pth
    train_net()

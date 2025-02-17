import os
import cv2
import shutil
import datetime
import argparse
import numpy as np
import logging as logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
from sklearn.metrics import f1_score
from mydatasets import MyDataset
from losses import MyInfoNCE, MyLoss
from HRFormer.hrt_backbone import get_hrformer

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=512, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=4, help='resolution of input / output')
parser.add_argument('--train_bs', type=int, default=4, help='training batch size')
parser.add_argument('--test_bs', type=int, default=8, help='testing batch size')
parser.add_argument('--flist_path', type=str, default='../../flist/', help='data set path')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--metric', type=str, default='cosine', help='metric for loss and clustering')

args = parser.parse_args()

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now
device = torch.device('cuda:{}'.format(args.gpu))

class MPC(nn.Module):
    def __init__(self):
        super(MPC, self).__init__()
        self.lr = 1e-4
        self.cur_net = get_hrformer().to(device)
        self.optimizer = optim.AdamW(self.cur_net.parameters(), lr=self.lr)
        self.save_dir = 'weights/' + args.out_dir
        rm_and_make_dir(self.save_dir)
        self.myInfoNCE = MyInfoNCE(metric=args.metric)
        self.clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)

    def process(self, Ii, Mgs, isTrain=False):
        self.optimizer.zero_grad()

        if isTrain:
            Fo = self.cur_net(Ii)
            tmp_listxx = []
            for i in range(4):
                tmp_listxx.append(F.normalize(Fo[i].permute(0, 2, 3, 1), dim=3))
            Fo = tmp_listxx
            B, H, W, C = Fo[0].shape

            Fo_2 = self.cur_net(Ii)
            tmp_listyy = []
            for i in range(4):
                tmp_listyy.append(F.normalize(Fo_2[i].permute(0, 2, 3, 1), dim=3))
            Fo_2 = tmp_listyy
        else:
            with torch.no_grad():
                Fo = self.cur_net(Ii)
                tmp_listxx = []
                for i in range(4):
                    tmp_listxx.append(F.normalize(Fo[i].permute(0, 2, 3, 1), dim=3))
                Fo = tmp_listxx
                B, H, W, C = Fo[0].shape

        if isTrain:
            info_nce_loss = []
            out1x, out2x, out3x, out4x = Fo[0], Fo[1], Fo[2], Fo[3]
            out1y, out2y, out3y, out4y = Fo_2[0], Fo_2[1], Fo_2[2], Fo_2[3]

            final_listx = []
            final_listy = []

            for i in range(B):
                tmp_listx = []
                tmp_listx.append(out1x[i])
                tmp_listx.append(out2x[i])
                tmp_listx.append(out3x[i])
                tmp_listx.append(out4x[i])
                final_listx.append(tmp_listx)

                tmp_listy = []
                tmp_listy.append(out1y[i])
                tmp_listy.append(out2y[i])
                tmp_listy.append(out3y[i])
                tmp_listy.append(out4y[i])
                final_listy.append(tmp_listy)

            for idx in range(B):
                Fo_idx = final_listx[idx]
                Fo_idx_2 = final_listy[idx]
                Mg_idx = Mgs[idx][0]
                info_nce_loss.append(MyLoss(Fo_idx, Fo_idx_2, Mg_idx))

            batch_loss = torch.mean(torch.stack(info_nce_loss).squeeze())
            self.backward(batch_loss)
            return batch_loss
        else:
            Mo = None
            out1, out2, out3, out4 = Fo[0], Fo[1], Fo[2], Fo[3]
            tmp_list = []
            for i in range(B):
                outs = []
                outs.append(torch.unsqueeze(out1[i], dim=0).permute(0, 3, 1, 2))
                outs.append(
                    F.interpolate(
                        input=torch.unsqueeze(out2[i], dim=0).permute(0, 3, 1, 2),
                        size=outs[0].shape[2:],
                        mode='bilinear',
                        align_corners=True))
                outs.append(
                    F.interpolate(
                        input=torch.unsqueeze(out3[i], dim=0).permute(0, 3, 1, 2),
                        size=outs[0].shape[2:],
                        mode='bilinear',
                        align_corners=True))
                outs.append(
                    F.interpolate(
                        input=torch.unsqueeze(out4[i], dim=0).permute(0, 3, 1, 2),
                        size=outs[0].shape[2:],
                        mode='bilinear',
                        align_corners=True))
                fusion_feature = torch.cat(outs, dim=1).permute(0, 2, 3, 1)
                tmp_list.append(fusion_feature)

            Fo = torch.cat(tmp_list, dim=0)

            Fo = torch.flatten(Fo, start_dim=1, end_dim=2)
            result = self.clustering(x=Fo, k=2)
            Lo_batch = result.labels
            for idx in range(B):
                Lo = Lo_batch[idx]
                if torch.sum(Lo) > torch.sum(1 - Lo):
                    Lo = 1 - Lo
                Lo = Lo.view(H, W)[None, :, :, None]
                Mo = torch.cat([Mo, Lo], dim=0) if Mo is not None else Lo
            Mo = Mo.permute(0, 3, 1, 2)
            return Mo

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward(retain_graph=False)
            self.optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.cur_net.state_dict(),
                   self.save_dir + path + 'weights.pth')


class ForgeryForensics():
    def __init__(self):
        self.train_npy_list = [
            # name, repeat_time
            ('tampCOCO_sp_199999.npy', 1),
            ('tampCOCO_cm_199429.npy', 1),
            ('tampCOCO_bcm_199443.npy', 1),
            ('CASIA2_5123.npy', 40),
            ('IMD_2010.npy', 20), ]
        self.train_file = None
        for item in self.train_npy_list:
            self.train_file_tmp = np.load(args.flist_path + item[0])
            for _ in range(item[1]):
                self.train_file = np.concatenate(
                    [self.train_file, self.train_file_tmp]) if self.train_file is not None else self.train_file_tmp

        self.train_num = len(self.train_file)
        train_dataset = MyDataset(num=self.train_num, file=self.train_file, choice='train',
                                  input_size=args.input_size, gt_ratio=args.gt_ratio)

        self.val_npy_list = [
            # name, nickname
            ('CASIAv1_920.npy', 'CASIAv1'),
            ('DSO_100.npy', 'DSO'),
        ]
        self.val_file_list = []
        for item in self.val_npy_list:
            self.val_file_tmp = np.load(args.flist_path + item[0])
            self.val_file_list.append(self.val_file_tmp)

        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        self.mpc = MPC().to(device)
        self.n_epochs = 99999
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, num_workers=4,
                                       shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Validation on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        count, batch_losses = 0, []
        best_score = 0
        scheduler = ReduceLROnPlateau(self.mpc.optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-8)

        self.mpc.train()
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                count += self.train_bs
                Ii, Mg = (item.to(device) for item in items[:2])
                batch_loss = self.mpc.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())
                if count % (self.train_bs * 80) == 0:
                    logger.info('Train Num (%6d/%6d), Loss:%5.4f' % (count, self.train_num, np.mean(batch_losses)))
                if count % int((self.train_loader.dataset.__len__() / 100) // self.train_bs * self.train_bs) == 0:
                    self.mpc.save('latest/')
                    logger.info('Ep%03d(%6d/%6d), Loss:%5.4f' % (epoch, count, self.train_num, np.mean(batch_losses)))
                    tmp_score = self.val()
                    scheduler.step(tmp_score)
                    if tmp_score > best_score:
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best)' % best_score)

                        self.mpc.save('Ep%03d_%5.4f/' % (epoch, tmp_score))
                    else:
                        logger.info('Score: %5.4f' % tmp_score)

                    self.mpc.train()
                    batch_losses = []
            count = 0

    def val(self):
        tmp_F1 = []
        tmp_IOU = []
        for idx in range(len(self.val_file_list)):
            P_F1, P_IOU = ForensicTesting(self.mpc, bs=self.test_bs, test_npy=self.val_npy_list[idx][0],
                                                       test_file=self.val_file_list[idx])
            tmp_F1.append(P_F1)
            tmp_IOU.append(P_IOU)
            logger.info('%s(#%d): PF1:%5.4f, PIOU:%5.4f' % (
                self.val_npy_list[idx][1], len(self.val_file_list[idx]), P_F1, P_IOU))

        return np.mean(tmp_F1+ tmp_IOU)


# test
def ForensicTesting(model, bs=1, test_npy='', test_file=None):
    if test_file is None:
        test_file = np.load(args.flist_path + test_npy)
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, 2), shuffle=False)
    model.eval()

    f1, iou = [], []

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.to(device) for item in items[:-1])
        Mo = model.process(Ii, None, isTrain=False)
        Mg, Mo = convert(Mg), convert(Mo)
        for i in range(Mo.shape[0]):
            Mo_resized = thresholding(cv2.resize(Mo[i], (Mg[i].shape[:2][::-1])))[..., None]
            f1.append(f1_score(Mg[i].flatten() / 255., Mo_resized.flatten() / 255.))
            iou.append(metric_iou(Mo_resized / 255., Mg[i] / 255.))

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    return Pixel_F1, Pixel_IOU


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou


if __name__ == '__main__':
    model = ForgeryForensics()
    model.train()

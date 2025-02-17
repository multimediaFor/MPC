import os
import cv2
import shutil
import datetime
import argparse
import numpy as np
import logging as logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from losses import Sigmoid_Focal_Loss
from model import MyModel
from mydatasets import MyDataset

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
parser.add_argument('--stage1_net_weight', type=str, default='./stage1_weight/MPC_CATNet_stage1_weights.pth', help='net weight of stage1')

args = parser.parse_args()

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now
device = torch.device('cuda:{}'.format(args.gpu))


class MPC(nn.Module):
    def __init__(self):
        super(MPC, self).__init__()
        self.lr = 1e-6
        self.cur_net = MyModel().to(device)
        self.load(self.cur_net, args.stage1_net_weight)
        self.optimizer = optim.AdamW(self.cur_net.decoder.parameters(), lr=self.lr)
        self.save_dir = 'weights/' + args.out_dir
        rm_and_make_dir(self.save_dir)

    def process(self, Ii, Mg, isTrain=False):
        self.optimizer.zero_grad()

        if isTrain:
            Fo = self.cur_net(Ii)
        else:
            with torch.no_grad():
                Fo = self.cur_net(Ii)
        if isTrain:
            batch_loss = Sigmoid_Focal_Loss(Fo, Mg)
            self.backward(batch_loss)
            return batch_loss
        else:
            return torch.sigmoid(Fo)

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward(retain_graph=False)
            self.optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.cur_net.state_dict(),
                   self.save_dir + path + 'weights.pth')

    def load(self, extractor, path=''):
        weights_file = torch.load('weights/' + path)
        cur_weights = extractor.state_dict()
        for key in weights_file:
            if key in cur_weights.keys() and weights_file[key].shape == cur_weights[key].shape:
                cur_weights[key] = weights_file[key]
        extractor.load_state_dict(cur_weights)
        logger.info('Loaded weight from [%s]' % path)


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
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, num_workers=self.train_bs,
                                       shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Validation on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        count, batch_losses = 0, []
        best_score = 0
        scheduler = CosineAnnealingLR(self.mpc.optimizer, T_max=20, eta_min=1e-8)
        self.mpc.train()
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                count += self.train_bs
                Ii, Mg = (item.to(device) for item in items[:2])
                batch_loss = self.mpc.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())
                if count % (self.train_bs * 40) == 0:
                    logger.info('Train Num (%6d/%6d), Loss:%5.4f' % (count, self.train_num, np.mean(batch_losses)))
                if count % int((self.train_loader.dataset.__len__() / 100) // self.train_bs * self.train_bs) == 0:
                    self.mpc.save('latest/')
                    logger.info('Ep%03d(%6d/%6d), Loss:%5.4f' % (epoch, count, self.train_num, np.mean(batch_losses)))
                    tmp_score = self.val()
                    scheduler.step()
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

        return np.mean(tmp_F1 + tmp_IOU)


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


from mydatasets import MyDataset
from torch.utils.data import DataLoader
import argparse
import logging as logger
import torch.nn as nn
import os
import torch
import numpy as np
from model import MyModel
import cv2
import datetime
from sklearn.metrics import f1_score

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, default='./weights/MPC_CATNet_stage2_weights.pth',
                    help='weight path of trained model')
parser.add_argument('--input_size', type=int, default=512, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=1, help='resolution of input / output')
parser.add_argument('--save_result', type=bool, default=True, help='save test results')
parser.add_argument('--test_bs', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--flist_path', type=str, default='../../flist/', help='data set path')

args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Test_Result_%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now

device = torch.device('cuda:{}'.format(args.gpu))


class MPC(nn.Module):
    def __init__(self):
        super(MPC, self).__init__()
        self.cur_net = MyModel().to(device)
        self.load(self.cur_net, args.weight_path)

    def process(self, Ii):
        with torch.no_grad():
            Fo = self.cur_net(Ii)
        return torch.sigmoid(Fo)

    def load(self, model, path):
        weights = torch.load(path)
        model_state_dict = model.state_dict()

        loaded_layers = []
        missing_layers = []
        mismatched_shapes = []

        # 遍历加载的权重字典
        for name, param in weights.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)  # 更新模型的权重
                    loaded_layers.append(name)
                else:
                    mismatched_shapes.append(name)
            else:
                # 如果模型中没有该层，记录缺失的层
                missing_layers.append(name)

        # 打印加载成功的层
        if loaded_layers:
            logger.info(f"Successfully loaded the following layers: {', '.join(loaded_layers)}")

        # 打印形状不匹配的层
        if mismatched_shapes:
            logger.warning(f"The following layers have mismatched shapes: {', '.join(mismatched_shapes)}")

        # 打印缺失的层
        if missing_layers:
            logger.warning(f"The following layers are missing in the model: {', '.join(missing_layers)}")

        # 如果都加载成功，打印成功信息
        if not mismatched_shapes and not missing_layers:
            logger.info("All layers have been successfully loaded!")


class ForgeryForensics():
    def __init__(self):
        self.MPC = MPC().to(device)
        self.test_npy_list = [
            # name, nickname
            ('Columbia_160.npy', 'Columbia'),
            ('DSO_100.npy', 'DSO'),
            ('CASIAv1_920.npy', 'CASIAv1'),
            ('NIST_564.npy', 'NIST'),
            ('Coverage_100.npy', 'Coverage'),
            ('Korus_220.npy', 'Korus'),
            ('In_the_wild_201.npy', 'In_the_wild'),
            ('CoCoGlide_512.npy', 'CoCoGlide'),
            ('MISD_227.npy', 'MISD'),
            ('FFpp_1000.npy', 'FFpp'),
        ]
        self.test_file_list = []
        for item in self.test_npy_list:
            self.test_file_tmp = np.load(args.flist_path + item[0])
            self.test_file_list.append(self.test_file_tmp)
        self.test_bs = args.test_bs

        for idx, file_list in enumerate(self.test_file_list):
            logger.info('Test on %s (#%d).' % (self.test_npy_list[idx][0], len(file_list)))

    def test(self):
        tmp_F1 = []
        tmp_IOU = []
        result_file_path = os.path.join(args.out_dir, 'result.txt')
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, 'a') as result_file:  # Open the file for appending
            for idx in range(len(self.test_file_list)):
                P_F1, P_IOU = ForensicTesting(self.MPC, bs=self.test_bs,
                                              test_file=self.test_file_list[idx],
                                              test_set_name=self.test_npy_list[idx][1])
                tmp_IOU.append(P_IOU)
                tmp_F1.append(P_F1)
                result_str = '%s(#%d): F1:%5.4f, PIOU:%5.4f\n' % (
                    self.test_npy_list[idx][1], len(self.test_file_list[idx]), P_F1, P_IOU
                )
                result_file.write(result_str)


def ForensicTesting(model, bs=1, test_file=None, test_set_name=None):
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test', input_size=args.input_size, gt_ratio=args.gt_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, 2), shuffle=False)

    model.eval()
    f1, iou = [], []

    save_dir = os.path.join(args.out_dir, test_set_name)
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.to(device) for item in items[:-1])
        filename = items[-1]
        Mo = model.process(Ii)  # [B, 2, H, W]

        Mg, Mo = convert(Mg), convert(Mo)

        if args.save_result:

            Hg, Wg = Hg.cpu().numpy(), Wg.cpu().numpy()
            for i in range(Ii.shape[0]):
                res = cv2.resize(Mo[i], (Wg[i].item(), Hg[i].item()))
                res = thresholding(res)
                cv2.imwrite(os.path.join(save_dir, filename[i].split('.')[-2] + '.png'), res.astype(np.uint8))

        for i in range(Mo.shape[0]):
            Mo_resized = thresholding(cv2.resize(Mo[i], (Mg[i].shape[:2][::-1])))[..., None]
            f1.append(f1_score(Mg[i].flatten() / 255., Mo_resized.flatten() / 255.))
            iou.append(metric_iou(Mo_resized / 255., Mg[i] / 255.))

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    return Pixel_F1, Pixel_IOU


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x

def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()

def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou


if __name__ == '__main__':
    model = ForgeryForensics()
    model.test()

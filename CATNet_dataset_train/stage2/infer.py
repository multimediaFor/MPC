import cv2
import numpy as np
import os
import torchvision
from torchvision.datasets import ImageFolder
import argparse
import torch
import torch.nn as nn
from model import MyModel
from PIL import Image

class MPC(nn.Module):
    def __init__(self, weight_path, device):
        super(MPC, self).__init__()
        self.cur_net = MyModel().to(device)
        self.load(self.cur_net, weight_path)

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
            print(f"Successfully loaded the following layers: {', '.join(loaded_layers)}")

        # 打印形状不匹配的层
        if mismatched_shapes:
            print(f"The following layers have mismatched shapes: {', '.join(mismatched_shapes)}")

        # 打印缺失的层
        if missing_layers:
            print(f"The following layers are missing in the model: {', '.join(missing_layers)}")

        # 如果都加载成功，打印成功信息
        if not mismatched_shapes and not missing_layers:
            print("All layers have been successfully loaded!")

def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()

def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x

def args_parser():
    parser = argparse.ArgumentParser(description="MPC Model Inference")
    parser.add_argument('--weight-path', type=str, default="./weights/MPC_CATNet_stage2_weights.pth", help='Path to the model weights')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (e.g., cuda:0 or cpu)')
    parser.add_argument('--infer-folder', type=str, default='./data', help='Folder containing images for inference')
    parser.add_argument('--output-folder', type=str, default='./output/', help='Folder to save inference results')
    return parser.parse_args()

def infer(args):

    device = torch.device(args.device)
    mpc_model = MPC(weight_path=args.weight_path, device=device)

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Resize((512, 512)),
    ])

    with torch.no_grad():
        for img in os.listdir(args.infer_folder):
            img_path = os.path.join(args.infer_folder, img)
            print(f"Processing image: {img_path}")
            if not os.path.isfile(img_path):
                continue
            images = Image.open(img_path).convert('RGB')
            images = image_transform(images).unsqueeze(0).to(device)
            images = images.to(device)
            Mo = mpc_model.process(images)
            Mo = convert(Mo)
            print(f"Processed image shape: {Mo.shape}")
            output_path = os.path.join(args.output_folder, img)
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            cv2.imwrite(output_path, thresholding(Mo[0]))

if __name__ == '__main__':
    args = args_parser()
    os.makedirs(args.output_folder, exist_ok=True)
    infer(args)
Doc    print("Inference completed and results saved to:", args.output_folder)
            
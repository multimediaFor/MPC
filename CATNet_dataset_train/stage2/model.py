import warnings

warnings.filterwarnings("ignore")
import torch.nn as nn
from decoder_head import Decoder
from HRFormer.hrt_backbone import get_hrformer
import torch
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.encoder = get_hrformer()
        self.decoder = Decoder()

    def forward(self, inputs):
        x = self.encoder(inputs)
        out = self.decoder(x)
        return out

# import torch
# import torch.nn as nn
# from torchsummary import summary
# from fvcore.nn import FlopCountAnalysis, parameter_count
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyModel().to(device)
# x = torch.randn(1, 3, 512, 512).to(device)
# # 计算FLOPs, 跟输入图像大小相关
# flops = FlopCountAnalysis(model, x)
# print(f"GFLOPs: {flops.total() / 1e9:.3f}")
# x = torch.randn(1, 3, 1024, 1024).to(device)
# # 计算FLOPs, 跟输入图像大小相关
# flops = FlopCountAnalysis(model, x)
# print(f"GFLOPs: {flops.total() / 1e9:.3f}")
#
# # 计算参数量
# params = parameter_count(model)
# print(f"Parameters (M): {params[''] / 1e6:.3f}")

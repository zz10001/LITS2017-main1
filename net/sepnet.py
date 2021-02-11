import torch
from torch import nn
import torch.nn.functional as F

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand):
        super(InvertedResidual, self).__init__()
        self.expand=expand
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 0, dilation=expand, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, 1, bias=False),
        )

    def forward(self, x):
        x_pad = fixed_padding(x, 3, self.expand)
        y= self.conv(x_pad)
        return y

class block_down(nn.Module):

    def __init__(self, inp_channel, out_channel, expand):
        super(block_down, self).__init__()
        self.deepwise1 = InvertedResidual(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual(inp_channel, out_channel, expand)
        self.resnet= nn.Conv2d(inp_channel, out_channel, 1, 1, 0, 1, bias=False)

    def forward(self, input):
        resnet=self.resnet(input)
        x = self.deepwise1(input)
        x= self.deepwise2(x)
        out=torch.add(resnet,x)
        return out


class block_up(nn.Module):

    def __init__(self, inp_channel, out_channel, expand):
        super(block_up, self).__init__()
        self.up = nn.ConvTranspose2d(inp_channel, out_channel, 2, stride=2)
        self.deepwise1 = InvertedResidual(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual(inp_channel, out_channel, expand)
        self.resnet = nn.Conv2d(inp_channel, out_channel, 1, 1, 0, 1, bias=False)

    def forward(self, x, y):
        x = self.up(x)
        x1 = torch.cat([x, y], dim=1)
        x = self.deepwise1(x1)
        x = self.deepwise2(x)
        resnet=self.resnet(x1)
        out=torch.add(resnet,x)

        return out


class U_net(nn.Module):

    def __init__(self, args):
        super(U_net, self).__init__()
        self.args = args
        class_num = 2
        self.inp = nn.Conv2d(3, 64, 1)
        self.block2 = block_down(64, 128, expand=1)
        self.block3 = block_down(128, 256, expand=2)
        self.block4 = block_down(256, 512, expand=2)
        self.block5 = block_down(512, 1024, expand=1)
        self.block6 = block_up(1024, 512, expand=1)
        self.block7 = block_up(512, 256, expand=1)
        self.block8 = block_up(256, 128, expand=2)
        self.block9 = block_up(128, 64, expand=2)
        self.out = nn.Conv2d(64, class_num, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1_use = self.inp(x)
        x1 = self.maxpool(x1_use)
        x2_use = self.block2(x1)
        x2 = self.maxpool(x2_use)
        x3_use = self.block3(x2)
        x3 = self.maxpool(x3_use)
        x4_use = self.block4(x3)
        x4 = self.maxpool(x4_use)
        x5 = self.block5(x4)

        x6 = self.block6(x5, x4_use)
        x7 = self.block7(x6, x3_use)
        x8 = self.block8(x7, x2_use)
        x9 = self.block9(x8, x1_use)
        out= self.out(x9)
        return out


# if __name__ == "__main__":
#     test_input = torch.rand(1, 3, 480, 640).to("cuda")
#     print("input_size:", test_input.size())
#     model = U_net(3)
#     model.cuda()
#     ouput = model(test_input)
#     print("output_size:", ouput.size())
#     params=list(model.named_parameters())
#     k=0
#     for name,param in params:
#         print(name)
#         if param.requires_grad:
#             l=1
#             for i in param.size():
#                 l*=i
#             k=k+l
#         print(l)
#     print("模型总的参数量是："+str(k))
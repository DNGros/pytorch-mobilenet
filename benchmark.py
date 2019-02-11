# https://github.com/marvis/pytorch-mobilenet/blob/master/benchmark.py

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import timeit

class MobileNet(nn.Module):
    def __init__(self, max_group_size=None):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, 
                    groups=min(inp, max_group_size or 9e9), bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def speed(model, name, use_cuda, batch_size = 1):
    t0 = time.time()
    input = torch.rand(batch_size,3,224,224)
    if use_cuda:
        input = input.cuda()
    with torch.no_grad():
        t = timeit.Timer(lambda: model(input))
    print(f"{name:15s} {t.timeit(number=10):0.5f}s")

if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    models = [
        ("resnet18", models.resnet18),
        ("alexnet", models.alexnet),
        ("vgg16", models.vgg16),
        ("squeezenet1_0", models.squeezenet1_0),
        ("mobilenet", MobileNet),
        ("mobilenet one group", lambda: MobileNet(max_group_size=1)),
        ("mobilenet four group", lambda: MobileNet(max_group_size=4))
    ]

    print(f"Pytorch v{torch.__version__}")
    print(f"GPU name {torch.cuda.get_device_name(None)}")

    for use_cuda in (True, False):
        print(f"-----CUDA {use_cuda}-----")
        for batch_size in (1, 4, 32):
            print(f"--batch size = {batch_size}")
            for model_name, init_func in models:
                model = init_func()
                if use_cuda:
                    model = model.cuda()
                speed(model, model_name, use_cuda, batch_size=batch_size)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, mode, in_channels, hidden_channels, out_channels, kernel_size, stride, padding, activation):
        super(Block, self).__init__()
        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(hidden_channels),
            activation,
            nn.Conv2d(hidden_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            activation
        )
        self.up_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding),
            activation,
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, stride, padding),
            activation,
            nn.BatchNorm2d(out_channels),

        )
        self.mode = mode
        
    def forward(self, x):
        if self.mode == 'down':
            return self.down_block(x)
        elif self.mode == 'up':
            return self.up_block(x)
        
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.D_block1 = Block(mode='down', in_channels=in_channels, hidden_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.D_block2 = Block(mode='down', in_channels=64, hidden_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.D_block3 = Block(mode='down', in_channels=128, hidden_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.D_block4 = Block(mode='down', in_channels=256, hidden_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.D_block5 = Block(mode='down', in_channels=512, hidden_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        
        self.U_block1 = Block(mode='up', in_channels=1024, hidden_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.U_block2 = Block(mode='up', in_channels=512, hidden_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.U_block3 = Block(mode='up', in_channels=256, hidden_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))
        self.U_block4 = Block(mode='up', in_channels=128, hidden_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(64, out_channels, 3, 1, 1)       
        self.maxpool = nn.MaxPool2d(2)
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2, 0, bias=True)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2, 0, bias=True)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2, 0, bias=True)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=True)
        
    def forward(self, x):
        # print(f"input : {x.shape}")
        x = self.D_block1(x)
        residual_1 = x
        x = self.maxpool(x)
        x = self.D_block2(x)
        residual_2 = x
        x = self.maxpool(x)
        x = self.D_block3(x)
        residual_3 = x
        x = self.maxpool(x)
        x = self.D_block4(x)
        residual_4 = x
        x = self.maxpool(x)
        x = self.D_block5(x)
        
        x = torch.concat((residual_4, self.upconv1(x)), dim=1)
        x = self.U_block1(x)
        x = torch.concat((residual_3, self.upconv2(x)), dim=1)
        x = self.U_block2(x)
        x = torch.concat((residual_2, self.upconv3(x)), dim=1)
        x = self.U_block3(x)
        x = torch.concat((residual_1, self.upconv4(x)), dim=1)
        x = self.U_block4(x)
        x = self.conv1(x)
        # print(f"output : {x.shape}")
        return x
        
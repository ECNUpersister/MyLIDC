from model.unet.unet_components import *
from model.utils.SCSE import SCSEModule


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels,
                              64)  # 输入x.shape=[64,1,64,64] 这一步做了两次卷积 conv1=（in_channel=1，mid_channel=64） bn relu conv2=(in_channel=64,out_channel=64) bn relu
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.attention1 = SCSEModule(256, 16)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x.shape=[64,1,64,64] 第一个64是batch_size，第二个1是in_channel，最后的64*64是图像尺寸
        x1 = self.inc(x)  # x1.shape=[64,64,64,64]
        x2 = self.down1(x1)  # x2.shape=[64,128,32,32]
        x3 = self.down2(x2)  # x3.shape=[64,256,16,16]
        x4 = self.down3(x3)  # x4.shape=[64,512,8,8]
        x5 = self.down4(x4)  # x5.shape=[64,512,4,4]
        x = self.up1(x5, x4)  # x.shape=[64,256,8,8]
        x = self.attention1(x)
        x = self.up2(x, x3)  # x.shape=[64,128,16,16]
        x = self.up3(x, x2)  # x.shape=[64,64,32,32]
        x = self.up4(x, x1)  # x.shape=[64,64,64,64]
        logits = self.outc(x)  # logits.shape=[64,1,64,64]
        return logits

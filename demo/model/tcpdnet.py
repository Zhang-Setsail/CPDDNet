import torch.nn as nn
from . import register_model
from .base_model import BaseModel
import torch
from .base_unet import UNet_Base
from . import bilinear

@register_model('TCPDNet')
class TCPDNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.color_unet = UNet_Base(nf=config['nf'], in_channels=3, out_channels=3)
        self.polar_unet = UNet_Base(nf=config['nf'], in_channels=4, out_channels=4)

        self.colors = None
        self.polars = None

    def forward(self, x):

        sub_bayer_x = bilinear.generate_sub_bayer(x)

        self.colors = []
        self.polars = []

        for i in range(4):
            sub_color_x = bilinear.Color_Bilinear(sub_bayer_x[::,i,::,::])
            sub_color_x = self.color_unet(sub_color_x)
            self.colors.append(sub_color_x)

        colors = torch.stack(self.colors, dim=1)

        pixel_shuffle = bilinear.generate_polar_mosaic(colors)

        for i in range(3):
            sub_polar = bilinear.Polar_Bilinear(pixel_shuffle[::,i,::,::,::])
            sub_polar = self.polar_unet(sub_polar)
            self.polars.append(sub_polar)

        out = torch.stack(self.polars, dim=2)

        return out, colors
    
if __name__ == "__main__":
    config = {}
    device = torch.device("mps")
    model = TCPDNet(config).to(device)
    model.eval()
    model.color_unet.load_state_dict(torch.load("../pretrainedModel/tcpd_color_unet_model.pth"))
    model.polar_unet.load_state_dict(torch.load("../pretrainedModel/tcpd_polar_unet_model.pth"))
    print(model)
    # x = torch.randn(1, 1, 768, 1024).to(device)
    import numpy as np
    x = np.load("./test_raw_data.npy").astype(np.float32)
    x = torch.from_numpy(x).to(device).unsqueeze(0).unsqueeze(0)
    print(x.shape)
    out = model(x)
    print(out.shape)

    import cv2
    prgb_0 = out.squeeze(0)[0,::,::,::].permute(1,2,0).cpu().detach().numpy()
    cv2.imshow("prgb_0", prgb_0[::,::,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    torch.save(model.state_dict(), "../pretrainedModel/tcpd_model.pth")



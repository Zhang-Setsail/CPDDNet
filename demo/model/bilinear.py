import torch
import torch.nn.functional as F
import torch.nn as nn

def sparse_sub_bayer(sub_bayer):
    r_sub_bayer = torch.zeros_like(sub_bayer[:,0,:,:])
    r_sub_bayer[:, 0::2, 0::2] = sub_bayer[:,0,0::2,0::2]
    g_sub_bayer = torch.zeros_like(sub_bayer[:,0,:,:])
    g_sub_bayer[:, 1::2, 0::2] = sub_bayer[:,0,1::2,0::2]
    g_sub_bayer[:, 0::2, 1::2] = sub_bayer[:,0,0::2,1::2]
    b_sub_bayer = torch.zeros_like(sub_bayer[:,0,:,:])
    b_sub_bayer[:, 1::2, 1::2] = sub_bayer[:,0,1::2,1::2]
    return torch.stack([r_sub_bayer, g_sub_bayer, b_sub_bayer], dim=1)

def bilinear_interp(x, is_green=False):
    # dim info of x: (batch, c, h, w)
    if is_green:
        w = torch.tensor([[[[0, 1/4, 0],
                            [1/4, 1, 1/4],
                            [0, 1/4, 0]]]], dtype=x.dtype).to(device=x.device)
    else:
        w = torch.tensor([[[[1/4, 1/2, 1/4],
                            [1/2, 1, 1/2],
                            [1/4, 1/2, 1/4]]]], dtype=x.dtype).to(device=x.device)
    w.detach_()
    c = x.shape[1]
    w = w.repeat(c,1,1,1)
    x = F.pad(x, (1, 1, 1, 1), "reflect")
    #x = F.pad(x, (1, 1, 1, 1), "constant")
    return F.conv2d(x, w, bias=None, groups=c)


def Color_Bilinear(sub_bayer):
    sparsed_sub_bayer = sparse_sub_bayer(sub_bayer)
    sub_color = torch.zeros_like(sparsed_sub_bayer)
    sub_color[:,[0,2],:,:] = bilinear_interp(sparsed_sub_bayer[:,[0,2],:,:])
    sub_color[:,1,:,:] = bilinear_interp(sparsed_sub_bayer[:,1,:,:].unsqueeze(1), is_green=True).squeeze(1)
    return sub_color

def sparse_polar_mosaic(polar_mosaic):
    sparse_polar_mosaic_90 = torch.zeros_like(polar_mosaic[:,0,:,:])
    sparse_polar_mosaic_90[:,0::2,0::2] = polar_mosaic[:,0,0::2,0::2]
    sparse_polar_mosaic_45 = torch.zeros_like(polar_mosaic[:,0,:,:])
    sparse_polar_mosaic_45[:,0::2,1::2] = polar_mosaic[:,0,0::2,1::2]
    sparse_polar_mosaic_135 = torch.zeros_like(polar_mosaic[:,0,:,:])
    sparse_polar_mosaic_135[:,1::2,0::2] = polar_mosaic[:,0,1::2,0::2]
    sparse_polar_mosaic_0 = torch.zeros_like(polar_mosaic[:,0,:,:])
    sparse_polar_mosaic_0[:,1::2,1::2] = polar_mosaic[:,0,1::2,1::2]
    sparse_polar_mosaic = torch.stack([sparse_polar_mosaic_0, sparse_polar_mosaic_45, sparse_polar_mosaic_90, sparse_polar_mosaic_135], dim=1)
    return sparse_polar_mosaic

def Polar_Bilinear(polar_mosaic):
    sparsed_polar_mosaic = sparse_polar_mosaic(polar_mosaic)
    full_polar = bilinear_interp(sparsed_polar_mosaic)
    return full_polar

def generate_sub_bayer(raw_CPFA):
    # dim info of raw_CPFA: (batch, 1, h, w)
    tl = raw_CPFA[:,:,0::2,0::2]
    tr = raw_CPFA[:,:,0::2,1::2] 
    bl = raw_CPFA[:,:,1::2,0::2] 
    br = raw_CPFA[:,:,1::2,1::2] 
    return torch.stack([tl, tr, bl, br], axis=1)

def generate_polar_mosaic(sub_color):
    # sub_color: (batch, 4, 3, h/2, w/2)
    sub_color = torch.transpose(sub_color, 1, 2) #(batch, 3, 4, h/2, w/2)
    polar_mosaic = nn.PixelShuffle(2)(sub_color) #(batch, 3, 1, h, w)
    return polar_mosaic
import torch.nn as nn
import torch
import numpy as np

def scale(x, old_min, old_max, new_min, new_max):
    return (x - old_min)*(new_max - new_min)/(old_max - old_min) + new_min

def compute_stokes(x):
    # dim info of x: (batch, polar, rgb,  h, w)
    # dim info here: polar = 0 => 0, polar = 1 => 45, polar = 2 => 90, polar = 3 => 135
    s = torch.stack([torch.sum(x, axis=1)/2,
                   x[:,0,:,:,:]-x[:,2,:,:,:],
                   x[:,1,:,:,:]-x[:,3,:,:,:]],
                    axis=1)

    # now let us scale the Stokes parameters
    scaled_s = torch.stack([scale(s[:,0,:,:,:], 0, 2, 0, 1),
                          scale(s[:,1,:,:,:], -1, 1, 0, 1),
                          scale(s[:,2,:,:,:], -1, 1, 0, 1)],
                           axis=1)
    return s, scaled_s

def compute_dop(s, d = 0):
    # dim info of s: (batch, S, rgb, h, w)
    dop = torch.div(torch.sqrt(torch.square(s[:,1,:,:,:]) + torch.square(s[:,2,:,:,:]) + d), s[:,0,:,:,:])

    # now let us scale the DOP
    #dop = scale(dop, 0, 2, 0, 1)
    # we do not need to scale the DOP here because we divide by 2 instead of 4
    # reference: https://github.com/pjlapray/Polarimetric_Spectral_Database/issues/2
    dop[dop < 0] = 0
    dop[dop > 1] = 1
    dop[torch.isnan(dop)] = 0
    dop[torch.isinf(dop)] = 0

    return dop

def compute_aop(s):
    # dim info of s: (batch, S, rgb, h, w)

    aop = 0.5*torch.atan2(s[:,2,:,:,:], s[:,1,:,:,:])

    # torch.atan2 function ranges from -pi to pi
    # try: torch.rad2deg(torch.atan2(torch.tensor(-0.000000001), torch.tensor(-0.999999)))
    # and: torch.rad2deg(torch.atan2(torch.tensor(0.000000001), torch.tensor(-0.999999)))
    # ref: https://www.mathworks.com/help/matlab/ref/atan2.html

    # now let us scale the AOP
    aop = scale(aop, np.deg2rad(-90), np.deg2rad(90), 0, 1)

    return aop


'''
CPSNR is a Pytorch version of:
https://github.com/ymonno/EARI-Polarization-Demosaicking/blob/master/Functions/TIP_RI/imcpsnr.m
'''

def CPSNR(x, y, peak=1, b=15):
    # dim info of x: (batch, rgb, h, w)
    # dim info of y: (batch, rgb, h, w)
    # b: a scalar
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction="none")
    batch_psnr = mse(x, y).detach()
    cpsnr = torch.mean(10*torch.log10(peak**2/(torch.mean(batch_psnr, (1, 2, 3)) + 1e-32)))
    return cpsnr

'''
angle_error is a Pytorch version of:
https://github.com/ymonno/EARI-Polarization-Demosaicking/blob/master/Functions/angleerror_AOLP.m
'''

def angle_error(x, y, b=15):
    # dim info of x: (batch, rgb/polar, h, w)
    # dim info of y: (batch, rgb/polar, h, w)
    # b: a scalar
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction="none")
    batch_error,_ = torch.min(torch.stack([mse(x, y).detach(), mse(x-1,y).detach(), mse(x+1,y).detach()], axis=0), 0)
    angle_error = torch.mean(torch.sqrt(torch.mean(batch_error, (1, 2, 3)))*180)
    return angle_error

def RMSE(x, y, b=15):
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction='none')
    rmse = torch.mean(torch.sqrt(torch.mean(mse(x*255, y*255), (1,2,3))))
    return rmse
    

def eval_all(y, gt):
    y_s, y_scaled_s = compute_stokes(y)
    gt_s, gt_scaled_s = compute_stokes(gt)

    s0_cpsnr = CPSNR(y_scaled_s[:,0,:,:,:], gt_scaled_s[:,0,:,:,:])
    s1_cpsnr = CPSNR(y_scaled_s[:,1,:,:,:], gt_scaled_s[:,1,:,:,:])
    s2_cpsnr = CPSNR(y_scaled_s[:,2,:,:,:], gt_scaled_s[:,2,:,:,:])

    i0_cpsnr = CPSNR(y[:,0,:,:,:], gt[:,0,:,:,:])
    i45_cpsnr = CPSNR(y[:,1,:,:,:], gt[:,1,:,:,:])
    i90_cpsnr = CPSNR(y[:,2,:,:,:], gt[:,2,:,:,:])
    i135_cpsnr = CPSNR(y[:,3,:,:,:], gt[:,3,:,:,:])

    y_dop = compute_dop(y_s)
    gt_dop = compute_dop(gt_s)
    dop_cpsnr = CPSNR(y_dop, gt_dop)
    
    y_aop = compute_aop(y_s)
    gt_aop = compute_aop(gt_s)
    aop_angle_error = angle_error(y_aop, gt_aop)

    s0_rmse = RMSE(y_scaled_s[:,0,:,:,:], gt_scaled_s[:,0,:,:,:])
    s1_rmse = RMSE(y_scaled_s[:,1,:,:,:], gt_scaled_s[:,1,:,:,:])
    s2_rmse = RMSE(y_scaled_s[:,2,:,:,:], gt_scaled_s[:,2,:,:,:])

    i0_rmse = RMSE(y[:,0,:,:,:], gt[:,0,:,:,:])
    i45_rmse = RMSE(y[:,1,:,:,:], gt[:,1,:,:,:])
    i90_rmse = RMSE(y[:,2,:,:,:], gt[:,2,:,:,:])
    i135_rmse = RMSE(y[:,3,:,:,:], gt[:,3,:,:,:])
    
    dop_rmse = RMSE(y_dop, gt_dop)

    avg_cpsnr = (i0_cpsnr + i45_cpsnr + i90_cpsnr + i135_cpsnr + dop_cpsnr + s0_cpsnr + s1_cpsnr + s2_cpsnr)/8
    
    return { 'CPSNR/I0': i0_cpsnr,
             'CPSNR/I45': i45_cpsnr,
             'CPSNR/I90': i90_cpsnr,
             'CPSNR/I135': i135_cpsnr,
             'CPSNR/S0': s0_cpsnr,
             'CPSNR/S1': s1_cpsnr,
             'CPSNR/S2': s2_cpsnr,
             'CPSNR/DoP': dop_cpsnr,
             'CPSNR/Average': avg_cpsnr,
             'Error/AoP (angle error)': aop_angle_error }, y_dop, y_aop


if __name__ == "__main__":
    gt = torch.randn(1, 4, 3, 256, 256)
    gt_noise = gt + torch.randn_like(gt)*0.1
    print(eval_all(gt_noise, gt))
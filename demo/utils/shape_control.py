import torch
import numpy as np

# reference: https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799
def rgb_to_ycbr(x):
    # dim info of x: (batch, polar, rgb, h, w)
    y = torch.zeros_like(x)
    y[:,:,0,:,:] = x[:,:,0,:,:]*65.481 + x[:,:,1,:,:]*128.553 + x[:,:,2,:,:]*24.966 + 16
    y[:,:,1,:,:] = -x[:,:,0,:,:]*37.797 - x[:,:,1,:,:]*74.203 + x[:,:,2,:,:]*112 + 128
    y[:,:,2,:,:] = x[:,:,0,:,:]*112 - x[:,:,1,:,:]*93.786 - x[:,:,2,:,:]*18.214 + 128
    return y

def dense_raw_to_channel_separate_torch(raw_data: torch.Tensor) -> torch.Tensor:
    channel_separate_data = torch.zeros(16, raw_data.shape[0]//4, raw_data.shape[1]//4, device=raw_data.device)
    channel_separate_data[0,:,:] = raw_data[0::4,0::4]
    channel_separate_data[1,:,:] = raw_data[0::4,1::4]
    channel_separate_data[2,:,:] = raw_data[0::4,2::4]
    channel_separate_data[3,:,:] = raw_data[0::4,3::4]
    channel_separate_data[4,:,:] = raw_data[1::4,0::4]
    channel_separate_data[5,:,:] = raw_data[1::4,1::4]
    channel_separate_data[6,:,:] = raw_data[1::4,2::4]
    channel_separate_data[7,:,:] = raw_data[1::4,3::4]
    channel_separate_data[8,:,:] = raw_data[2::4,0::4]
    channel_separate_data[9,:,:] = raw_data[2::4,1::4]
    channel_separate_data[10,:,:] = raw_data[2::4,2::4]
    channel_separate_data[11,:,:] = raw_data[2::4,3::4]
    channel_separate_data[12,:,:] = raw_data[3::4,0::4]
    channel_separate_data[13,:,:] = raw_data[3::4,1::4]
    channel_separate_data[14,:,:] = raw_data[3::4,2::4]
    channel_separate_data[15,:,:] = raw_data[3::4,3::4]
    return channel_separate_data

def channel_separate_to_dense_raw_torch(channel_separate_data: torch.Tensor) -> torch.Tensor:
    dense_raw_data = torch.zeros(channel_separate_data.shape[0], channel_separate_data.shape[2]*4, channel_separate_data.shape[3]*4, device=channel_separate_data.device)
    dense_raw_data[::,0::4,0::4] = channel_separate_data[::,0,:,:]
    dense_raw_data[::,0::4,1::4] = channel_separate_data[::,1,:,:]
    dense_raw_data[::,0::4,2::4] = channel_separate_data[::,2,:,:]
    dense_raw_data[::,0::4,3::4] = channel_separate_data[::,3,:,:]
    dense_raw_data[::,1::4,0::4] = channel_separate_data[::,4,:,:]
    dense_raw_data[::,1::4,1::4] = channel_separate_data[::,5,:,:]
    dense_raw_data[::,1::4,2::4] = channel_separate_data[::,6,:,:]
    dense_raw_data[::,1::4,3::4] = channel_separate_data[::,7,:,:]
    dense_raw_data[::,2::4,0::4] = channel_separate_data[::,8,:,:]
    dense_raw_data[::,2::4,1::4] = channel_separate_data[::,9,:,:]
    dense_raw_data[::,2::4,2::4] = channel_separate_data[::,10,:,:]
    dense_raw_data[::,2::4,3::4] = channel_separate_data[::,11,:,:]
    dense_raw_data[::,3::4,0::4] = channel_separate_data[::,12,:,:]
    dense_raw_data[::,3::4,1::4] = channel_separate_data[::,13,:,:]
    dense_raw_data[::,3::4,2::4] = channel_separate_data[::,14,:,:]
    dense_raw_data[::,3::4,3::4] = channel_separate_data[::,15,:,:]
    return dense_raw_data

def dense_raw_to_channel_separate(raw_data: np.array) -> np.array:
    channel_separate_data = np.zeros((16, raw_data.shape[0]//4, raw_data.shape[1]//4))
    channel_separate_data[0,:,:] = raw_data[0::4,0::4]
    channel_separate_data[1,:,:] = raw_data[0::4,1::4]
    channel_separate_data[2,:,:] = raw_data[0::4,2::4]
    channel_separate_data[3,:,:] = raw_data[0::4,3::4]
    channel_separate_data[4,:,:] = raw_data[1::4,0::4]
    channel_separate_data[5,:,:] = raw_data[1::4,1::4]
    channel_separate_data[6,:,:] = raw_data[1::4,2::4]
    channel_separate_data[7,:,:] = raw_data[1::4,3::4]
    channel_separate_data[8,:,:] = raw_data[2::4,0::4]
    channel_separate_data[9,:,:] = raw_data[2::4,1::4]
    channel_separate_data[10,:,:] = raw_data[2::4,2::4]
    channel_separate_data[11,:,:] = raw_data[2::4,3::4]
    channel_separate_data[12,:,:] = raw_data[3::4,0::4]
    channel_separate_data[13,:,:] = raw_data[3::4,1::4]
    channel_separate_data[14,:,:] = raw_data[3::4,2::4]
    channel_separate_data[15,:,:] = raw_data[3::4,3::4]
    return channel_separate_data

def channel_separate_to_dense_raw(channel_separate_data: np.array) -> np.array:
    dense_raw_data = np.zeros((channel_separate_data.shape[1]*4, channel_separate_data.shape[2]*4))
    dense_raw_data[0::4,0::4] = channel_separate_data[0,:,:]
    dense_raw_data[0::4,1::4] = channel_separate_data[1,:,:]
    dense_raw_data[0::4,2::4] = channel_separate_data[2,:,:]
    dense_raw_data[0::4,3::4] = channel_separate_data[3,:,:]
    dense_raw_data[1::4,0::4] = channel_separate_data[4,:,:]
    dense_raw_data[1::4,1::4] = channel_separate_data[5,:,:]
    dense_raw_data[1::4,2::4] = channel_separate_data[6,:,:]
    dense_raw_data[1::4,3::4] = channel_separate_data[7,:,:]
    dense_raw_data[2::4,0::4] = channel_separate_data[8,:,:]
    dense_raw_data[2::4,1::4] = channel_separate_data[9,:,:]
    dense_raw_data[2::4,2::4] = channel_separate_data[10,:,:]
    dense_raw_data[2::4,3::4] = channel_separate_data[11,:,:]
    dense_raw_data[3::4,0::4] = channel_separate_data[12,:,:]
    dense_raw_data[3::4,1::4] = channel_separate_data[13,:,:]
    dense_raw_data[3::4,2::4] = channel_separate_data[14,:,:]
    dense_raw_data[3::4,3::4] = channel_separate_data[15,:,:]
    return dense_raw_data
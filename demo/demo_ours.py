input_list = ['data/FinalDataHigh/input_test/Scene36_1.npy',
              'data/FinalDataHigh/input_test/Scene37_1.npy',
              'data/FinalDataHigh/input_test/Scene38_1.npy',
              'data/FinalDataHigh/input_test/Scene39_1.npy',
              'data/FinalDataHigh/input_test/Scene40_1.npy']

label_list = [['data/FinalDataHigh/label/Scene36_0.png', 'data/FinalDataHigh/label/Scene36_45.png', 'data/FinalDataHigh/label/Scene36_90.png', 'data/FinalDataHigh/label/Scene36_135.png'],
              ['data/FinalDataHigh/label/Scene37_0.png', 'data/FinalDataHigh/label/Scene37_45.png', 'data/FinalDataHigh/label/Scene37_90.png', 'data/FinalDataHigh/label/Scene37_135.png'],
              ['data/FinalDataHigh/label/Scene38_0.png', 'data/FinalDataHigh/label/Scene38_45.png', 'data/FinalDataHigh/label/Scene38_90.png', 'data/FinalDataHigh/label/Scene38_135.png'],
              ['data/FinalDataHigh/label/Scene39_0.png', 'data/FinalDataHigh/label/Scene39_45.png', 'data/FinalDataHigh/label/Scene39_90.png', 'data/FinalDataHigh/label/Scene39_135.png'],
              ['data/FinalDataHigh/label/Scene40_0.png', 'data/FinalDataHigh/label/Scene40_45.png', 'data/FinalDataHigh/label/Scene40_90.png', 'data/FinalDataHigh/label/Scene40_135.png']]

pretrained_dn_model_path = 'pretrainedModel/Dn_model_raw2raw.pth'
pretrained_dm_color_model_path = 'pretrainedModel/fusion_tcpd_color_unet_model.pth'
pretrained_dm_polar_model_path = 'pretrainedModel/finetune_tcpd_polar_unet_model.pth'

save_dir = 'results'

config = {
    "name": "PIDNDM",
    "dn_model": {
        "in_nc": 16,
        "out_nc": 16,
        "nf": 64
    },
    "dm_model": {
        "nf": 64
    }
}   

import numpy as np
import os
import torch
import model
from utils import shape_control
import cv2
from utils.stokes import process_images_stokes

model = model.Model(config)

for input_path, label_path_list in zip(input_list, label_list):
    input_x = np.load(input_path)
    input_x = torch.from_numpy(input_x).float()

    label_x_0 = cv2.imread(label_path_list[0])[:,:,::-1].copy()
    label_x_45 = cv2.imread(label_path_list[1])[:,:,::-1].copy()
    label_x_90 = cv2.imread(label_path_list[2])[:,:,::-1].copy()
    label_x_135 = cv2.imread(label_path_list[3])[:,:,::-1].copy()

    save_path = os.path.join(save_dir, os.path.basename(input_path).split('.')[0])

    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, 'label_0.png'), label_x_0[...,::-1])
    cv2.imwrite(os.path.join(save_path, 'label_45.png'), label_x_45[...,::-1])
    cv2.imwrite(os.path.join(save_path, 'label_90.png'), label_x_90[...,::-1])
    cv2.imwrite(os.path.join(save_path, 'label_135.png'), label_x_135[...,::-1])

    label_s0, label_s1, label_s2, label_DoLP, label_AoP = process_images_stokes(label_x_0, label_x_45, label_x_90, label_x_135)
    
    # Convert from dense raw format to channel-separated format
    input_x = shape_control.dense_raw_to_channel_separate_torch(input_x)
    input_x = input_x.unsqueeze(0)  # Add batch dimension
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('mps')
    input_x = input_x.to(device)
    
    # Load DN model from training checkpoint
    dn_checkpoint = torch.load(pretrained_dn_model_path, map_location=device)
    model.dn_model.load_state_dict(dn_checkpoint['model_state_dict'])
    
    # Load DM models (these are direct state dicts)
    model.color_unet.load_state_dict(torch.load(pretrained_dm_color_model_path, map_location=device))
    model.polar_unet.load_state_dict(torch.load(pretrained_dm_polar_model_path, map_location=device))

    model.to(device)
    model.eval()
    with torch.no_grad():
        output_x, colors = model(input_x)
        output_x = output_x.cpu().numpy()

        output_x_0 = output_x[0,0,:,:,:].clip(0, 1).transpose(1, 2, 0)*255
        output_x_45 = output_x[0,1,:,:,:].clip(0, 1).transpose(1, 2, 0)*255
        output_x_90 = output_x[0,2,:,:,:].clip(0, 1).transpose(1, 2, 0)*255
        output_x_135 = output_x[0,3,:,:,:].clip(0, 1).transpose(1, 2, 0)*255

        cv2.imwrite(os.path.join(save_path, 'output_0.png'), output_x_0[...,::-1])
        cv2.imwrite(os.path.join(save_path, 'output_45.png'), output_x_45[...,::-1])
        cv2.imwrite(os.path.join(save_path, 'output_90.png'), output_x_90[...,::-1])
        cv2.imwrite(os.path.join(save_path, 'output_135.png'), output_x_135[...,::-1])

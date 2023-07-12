#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 01:15:34 2023

@author: marcorax93
"""

import h5py
import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt
import cv2, torch,torchvision
import torch.nn as nn
import torch.optim as optim




#%% Define the ANN

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, padding=(2,2), kernel_size=(5, 5), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=64, out_channels=28, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(4, 4), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(4, 4), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=16, padding=(1,1), kernel_size=(4, 4), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=(4, 4), stride=(2,2), bias=False),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=13, padding=(1,1), kernel_size=(3, 3), bias=False),
    nn.LeakyReLU()
    )


# model = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=16, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=16, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.ConvTranspose2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.ConvTranspose2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=64, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.ConvTranspose2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=32, out_channels=16, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU(),
#     nn.ConvTranspose2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
#     nn.LeakyReLU(),
#     nn.Conv2d(in_channels=16, out_channels=13, padding=(1,1), kernel_size=(3, 3)),
#     nn.LeakyReLU()
#     )

loss_fn = nn.MSELoss()  
optimizer = optim.RMSprop(model.parameters(), lr=8e-5)

n_epochs = 1
batch_size = 32


#%% Train the ANN
model.to('cuda') 
torch_save_f = "torch_save/train/"


for epoch in range(n_epochs):
    for subj in range(1,2):
        images_all=[]
        label_heatmaps_all=[]
        for sess in range(1,9):
            for mov in range(1,34):
                if isfile(join(torch_save_f,'S{}_session{}_mov{}_7500events_label'.format(subj,sess,mov))):
                    images = torch.load(torch_save_f+"S{}_session{}_mov{}_7500events".format(subj,sess,mov))
                    label_heatmaps = torch.load(torch_save_f+"S{}_session{}_mov{}_7500events_label".format(subj,sess,mov))
                    images_all.append(images)
                    label_heatmaps_all.append(label_heatmaps)
        images_all = torch.concat(images_all)
        label_heatmaps_all = torch.concat(label_heatmaps_all)
        label_heatmaps_all = torch.cat([label_heatmaps_all[:,0],label_heatmaps_all[:,1]])
        label_heatmaps_all = label_heatmaps_all.permute(0,3,1,2)
        images_all = torch.cat([images_all[:,:,:,0],images_all[:,:,:,1]])
        
        
        for i in range(0, len(images_all), batch_size):
            Xbatch = images_all[i:i+batch_size]
            Xbatch = Xbatch[:,None].to('cuda')
            y_pred = model(Xbatch)
            ybatch = label_heatmaps_all[i:i+batch_size].to('cuda')
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_py = y_pred.cpu().detach().numpy()
            ybatch_py = ybatch.cpu().detach().numpy()
            images_all_py = images_all[i:i+batch_size].cpu().detach().numpy()
            
        del images_all, label_heatmaps_all, Xbatch, y_pred, ybatch
    print(f'Finished epoch {epoch}, latest loss {loss}')
    
# Plot some results
plt.figure()
plt.imshow(np.sum(y_pred_py[14,:,:,:], 0))
plt.figure()
plt.imshow(np.sum(ybatch_py[14,:,:,:], 0))


plt.figure()
plt.imshow(images_all_py[14,:,:])

#%% Test Set
# model.to('cuda') 

torch_save_f = "torch_save/test/"


for subj in range(13,14):
    images_all=[] 
    label_heatmaps_all=[]
    for sess in range(1,9):
        for mov in range(1,34):
            if isfile(join(torch_save_f,'S{}_session{}_mov{}_7500events_label'.format(subj,sess,mov))):
                images = torch.load(torch_save_f+"S{}_session{}_mov{}_7500events".format(subj,sess,mov))
                label_heatmaps = torch.load(torch_save_f+"S{}_session{}_mov{}_7500events_label".format(subj,sess,mov))
                images_all.append(images)
                label_heatmaps_all.append(label_heatmaps)
    images_all = torch.concat(images_all)
    label_heatmaps_all = torch.concat(label_heatmaps_all)
    label_heatmaps_all = torch.cat([label_heatmaps_all[:,0],label_heatmaps_all[:,1]])
    label_heatmaps_all = label_heatmaps_all.permute(0,3,1,2)
    images_all = torch.cat([images_all[:,:,:,0],images_all[:,:,:,1]])
    
    
    for i in range(0, len(images_all), batch_size):
        Xbatch = images_all[i:i+batch_size]
        Xbatch = Xbatch[:,None].to('cuda')
        y_pred = model(Xbatch)
        ybatch = label_heatmaps_all[i:i+batch_size].to('cuda')
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_py = y_pred.cpu().detach().numpy()
        ybatch_py = ybatch.cpu().detach().numpy()
        images_all_py = images_all[i:i+batch_size].cpu().detach().numpy()
        
    del images_all, label_heatmaps_all, Xbatch, y_pred, ybatch
print(f'Finished epoch {epoch}, latest loss {loss}')
    
# Plot some results
plt.figure()
plt.imshow(np.sum(y_pred_py[2,:,:,:], 0))
plt.figure()
plt.imshow(np.sum(ybatch_py[2,:,:,:], 0))


plt.figure()
plt.imshow(images_all_py[2,:,:])

#%% Save Results
torch_save_f = "torch_save/"

# torch.save(images_all, torch_save_f+"input_images")
torch.save(model, torch_save_f+"model_100epochs_complete_training")
# torch.save(label_heatmaps_all, torch_save_f+"heat_maps_labels")
# torch.save(output_heatmap, torch_save_f+"pred_heat_maps")


#%% Load the tensors
torch_save_f = "torch_save/"

# images_all = torch.load(torch_save_f+"input_images")
model = torch.load(torch_save_f+"model_100epochs_complete_training")
# label_heatmaps_all = torch.load(torch_save_f+"heat_maps_labels")


#%% Original visual

# path of files generated using matlab 
path_ = 'h5/h5_dataset_7500_events/346x260/'
# camera projection matrices path
P_mat_dir = 'DHP19/P_matrices/'

image_h, image_w, num_joints = 260, 346, 13 # depend on how accumulated frames are generated in Matlab

t  = 19 # timestep of image to plot
subj, sess, mov = 2, 4, 6
decay_maps_flag = True # True to blur heatmaps
ch_idx = 2 # 0 to 3. This is the order of channels in .aedat/.h5

if ch_idx==1:
    P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
elif ch_idx==3:
    P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
elif ch_idx==2:
    P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
elif ch_idx==0:
    P_mat_cam = np.load(join(P_mat_dir,'P4.npy'))
    
    
vicon_xyz_all = load_file_(join(path_, 'S{}_session{}_mov{}_7500events_label.h5'.format(subj,sess,mov)))
images_all = load_file_(join(path_, 'S{}_session{}_mov{}_7500events.h5'.format(subj,sess,mov)))
vicon_xyz = vicon_xyz_all[t]
image = images_all[t, :, :, ch_idx]

# use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1,13])], axis=0)
coord_pix_all_cam2_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog/coord_pix_all_cam2_homog[-1]
print(np.shape(coord_pix_all_cam2_homog))
u = coord_pix_all_cam2_homog_norm[0]
v = image_h - coord_pix_all_cam2_homog_norm[1] # flip v coordinate to match the image direction

# mask is used to make sure that pixel positions are in frame range.
mask = np.ones(u.shape).astype(np.float32)
mask[u>image_w] = 0
mask[u<=0] = 0
mask[v>image_h] = 0
mask[v<=0] = 0

# pixel coordinates
u = u.astype(np.int32)
v = v.astype(np.int32)

# initialize the heatmaps
label_heatmaps = np.zeros((image_h, image_w, num_joints))

k = 2 # constant used to better visualize the joints when not using decay

for fmidx,pair in enumerate(zip(v,u, mask)):
    if decay_maps_flag:
        if pair[2]==1: # write joint position only when projection within frame boundaries
            label_heatmaps[pair[0],pair[1], fmidx] = 1
            label_heatmaps[:,:,fmidx] = decay_heatmap(label_heatmaps[:,:,fmidx])
    else:
        if pair[2]==1: # write joint position only when projection within frame boundaries
            label_heatmaps[(pair[0]-k):(pair[0]+k+1),(pair[1]-k):(pair[1]+k+1), fmidx] = 1

plt.figure()
plt.imshow(image, cmap='gray')
plt.imshow(np.sum(label_heatmaps, axis=-1), alpha=.5)
plt.show()



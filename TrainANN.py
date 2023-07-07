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

def load_file_(filepath):
    if filepath.endswith('.h5'):
        with h5py.File(filepath, 'r') as f_:
            data = (f_[list(f_.keys())[0]])[()]
    else:
        raise ValueError('.h5 required format.')
    return data

def decay_heatmap(heatmap, sigma2=4):
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma2)
    heatmap /= np.max(heatmap) # to keep the max to 1
    return heatmap


# path of files generated using matlab 
path_ = 'h5/h5_dataset_7500_events/346x260/'
# camera projection matrices path
P_mat_dir = 'DHP19/P_matrices/'

image_h, image_w, num_joints = 260, 346, 13 # depend on how accumulated frames are generated in Matlab
cut_image_h, cut_image_w, num_joints = 256, 256, 13 # depend on how accumulated frames are generated in Matlab
cut_x1 = (image_w-cut_image_w)//2
cut_x2 = cut_x1+cut_image_w
cut_y1 = (image_h-cut_image_h)//2
cut_y2 = cut_y1+cut_image_h
speck_downsample = 1/2
n_channels=2
cameras=[2,3]

decay_maps_flag = True # True to blur heatmaps
P_mat_cam = np.zeros([4,3,4])


P_mat_cam[1,:,:] = np.load(join(P_mat_dir,'P1.npy'))
P_mat_cam[3,:,:] = np.load(join(P_mat_dir,'P2.npy'))
P_mat_cam[2,:,:] = np.load(join(P_mat_dir,'P3.npy'))
P_mat_cam[0,:,:] = np.load(join(P_mat_dir,'P4.npy'))

P_mat_cam = P_mat_cam[cameras,:,:]

label_heatmaps_all = []
images_all = []


#12 subject training. test s13-17.
for subj in range(2,3):
    # for sess in range(1,6):
    for sess in range(1,3):
        for mov in range(1,34):
            if isfile(join(path_, 'S{}_session{}_mov{}_7500events_label.h5'.format(subj,sess,mov))):
                vicon_xyz = load_file_(join(path_, 'S{}_session{}_mov{}_7500events_label.h5'.format(subj,sess,mov)))
                images = load_file_(join(path_, 'S{}_session{}_mov{}_7500events.h5'.format(subj,sess,mov)))
                images = images[:,cut_y1:cut_y2,cut_x1:cut_x2,cameras]
                n_t = len(images)
                
                # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
                vicon_xyz_homog = np.ones([np.shape(vicon_xyz)[0],np.shape(vicon_xyz)[1]+1,np.shape(vicon_xyz)[2]])
                vicon_xyz_homog[:,:-1,:] = np.array(vicon_xyz)
                coord_pix_all_cam2_homog = np.einsum('kij,ljm->lkim',P_mat_cam, vicon_xyz_homog)
                coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog/coord_pix_all_cam2_homog[:,:,[-1],:]
                u = coord_pix_all_cam2_homog_norm[:,:,0]
                v = image_h - coord_pix_all_cam2_homog_norm[:,:,1] # flip v coordinate to match the image direction
                
                # pixel coordinates
                u = u.astype(np.int32)
                v = v.astype(np.int32)
                                
                # mask is used to make sure that pixel positions are in frame range.
                mask = np.ones(u.shape).astype(np.float32)
                mask[u>=image_w] = 0
                mask[u<=0] = 0
                mask[v>=image_h] = 0
                mask[v<=0] = 0
                
                
                
                # initialize the heatmaps
                label_heatmaps = np.zeros((n_t, n_channels, image_h, image_w, num_joints))
                
                k = 2 # constant used to better visualize the joints when not using decay
                
                for t in range(n_t):
                    for ch in range(n_channels):
                        for fmidx,pair in enumerate(zip(v[t,ch],u[t,ch], mask[t,ch])):
                            if decay_maps_flag:
                                if pair[2]==1: # write joint position only when projection within frame boundaries
                                    label_heatmaps[t,ch,pair[0],pair[1], fmidx] = 1
                                    label_heatmaps[t,ch,:,:,fmidx] = decay_heatmap(label_heatmaps[t,ch,:,:,fmidx])
                            else:
                                if pair[2]==1: # write joint position only when projection within frame boundaries
                                    label_heatmaps[t,ch,(pair[0]-k):(pair[0]+k+1),(pair[1]-k):(pair[1]+k+1), fmidx] = 1
                 
                label_heatmaps = label_heatmaps[:,:,cut_y1:cut_y2,cut_x1:cut_x2,:]
                
                
                if subj==2 and sess==4 and mov==6:
                    plt.figure()
                    plt.imshow(images[19,:,:,0], cmap='gray')
                    plt.imshow(np.sum(label_heatmaps[19,0], axis=-1), alpha=.5)
                    plt.show()
                                  
                images = list(images)
                
                for t in range(n_t):
                    images[t] = cv2.resize(images[t],(128,128))
                
                images = np.array(images)
                
                label_heatmaps_all.append(label_heatmaps)
                images_all.append(images)

#%% Preparing the tensors

label_heatmaps_all = torch.Tensor(np.concatenate(label_heatmaps_all))
images_all = torch.Tensor(np.concatenate(images_all))
label_heatmaps_all = torch.cat([label_heatmaps_all[:,0],label_heatmaps_all[:,1]])
label_heatmaps_all = label_heatmaps_all.permute(0,3,1,2)
images_all = torch.cat([images_all[:,:,:,0],images_all[:,:,:,1]])

#%% Training the ANN

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, padding=(2,2), kernel_size=(5, 5), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=64, out_channels=28, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=28, out_channels=28, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=28, out_channels=16, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=(3, 3)),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=(4, 4), stride=(2,2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=13, padding=(1,1), kernel_size=(3, 3)),
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

n_epochs = 1000
batch_size = 16

model.to('cuda') 

for epoch in range(n_epochs):
    for i in range(0, len(images_all), batch_size):
        Xbatch = images_all[i:i+batch_size]
        Xbatch = Xbatch[:,None].to('cuda')
        y_pred = model(Xbatch)
        ybatch = label_heatmaps_all[i:i+batch_size].to('cuda')
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
    
# Plot some results
plt.figure()
plt.imshow(np.sum(y_pred[14,:,:,:].cpu().detach().numpy(), 0))
plt.figure()
plt.imshow(np.sum(ybatch[14,:,:,:].cpu().detach().numpy(), 0))


plt.figure()
plt.imshow(images_all[19,:,:].cpu().detach().numpy())

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
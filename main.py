# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:00:18 2019

@author: Julien
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def load (file_name):
    img = nib.load(file_name)
    img_data = img.get_data()
    return img_data
    
def view4D(img_data,axe,b_plot=False,time=False):
    img_processed = []
    if not time :
        time = round(len(img_data[0][0][0])/2)
    if axe =='transversal':
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='coronal' or axe=='frontal':
        for i in range(len(img_data[0])):
            slice = img_data[:, i, :,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='sagital' or axe=='median':
        for i in range(len(img_data)):
            slice = img_data[i, :, :,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
                
    return img_processed

def plot_slice(i,slice,subtitle=""):
    plt.figure(i)
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show
    plt.suptitle(subtitle)

# =============================================================================
# data= load('Data/fmri.nii/fmri.nii')
# np_imgs = view4D('Data/fmri.nii/fmri.nii','transversal', False)
# np_img = np_imgs[0]
# plot_slice(1,np_img,"Image source")
# =============================================================================

img=nib.load('Data/fmri.nii/fmri.nii')
img_data = img.get_data()
img_header = img.header

def fft_voxel(img,x,y,z,plot=False):
    img_data = img.get_data()
    img_header = img.header
    voxel=img_data[x][y][z][:]
    #f = np.fft.fft(voxel)
    if plot :
        t = np.arange(85)
        # affichage du signal
        plt.subplot(211)
        plt.plot(t,voxel)
        
        # calcul de A
        A = np.fft.fft(voxel)
        # visualisation de A
        plt.subplot(212)
        plt.plot(np.real(A))
        plt.ylabel("partie reelle")
        
        plt.show()
    return

#fft_voxel(img,40,40,30,True)

def eliminate_non_brain(img_data):
    for t in range(0,85):
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i,t]
            median = np.median(slice)
            indexes = slice<=median
            img_data[:, :, i,t][indexes]=0
    return img_data

# Plot difference before and after
# =============================================================================
# img_data = load('Data/fmri.nii/fmri.nii')
# np_imgs =view4D(img_data,'transversal', False,10)
# np_img = np_imgs[0]
# plot_slice(1,np_img,"Image source")
# 
# img_data = eliminate_non_brain(img_data)
# np_imgs = view4D(img_data,'transversal', False,10)
# np_img = np_imgs[0]
# plot_slice(2,np_img,"Image après élimination")
# =============================================================================

print(img_header)

acq_num=85

# Select a random voxel by getting one random x- and y-coorinate
x_voxel = np.random.randint(64)
y_voxel = np.random.randint(64)
z_voxel = np.random.randint(50)

# Create the plot
fig, ax = plt.subplots(1,1,figsize=(15, 5))
ax.plot(img_data[x_voxel, y_voxel,z_voxel,:], lw=3)
ax.set_xlim(0, acq_num-1)
ax.set_xlabel('time [volumes]', fontsize=20)
ax.set_ylabel('signal strength', fontsize=20)
ax.set_title("Courbe d'intensité d'un voxel aléatoire", fontsize=25)
ax.tick_params(labelsize=12)
plt.show()

# Average all volumes
mean_data = img_data.mean(axis=2)

# Create the design matrix
constant = np.ones(acq_num)
with open("Data/ideal.txt") as f:
    content = f.readlines()
predicted_response= [float(x.strip()) for x in content] 

design_matrix = np.array((constant, predicted_response))

# Create the plots
fig, ax = plt.subplots(2,1, figsize=(15, 5))
ax[0].plot(design_matrix[0], lw=3)
ax[0].set_xlim(0, acq_num-1)
ax[0].set_ylim(0, 1.5)
ax[0].set_title('constant', fontsize=25)
ax[0].set_xticks([])
ax[0].set_yticks([0,1])
ax[0].tick_params(labelsize=12)
ax[0].tick_params(labelsize=12)

ax[1].plot(design_matrix[1], lw=3)
ax[1].set_xlim(0, acq_num-1)
ax[1].set_ylim(0, 1.5)
ax[1].set_title('expected response', fontsize=25)
ax[1].set_yticks([0,1])
ax[1].set_xlabel('time [volumes]', fontsize=20)
ax[1].tick_params(labelsize=12)
ax[1].tick_params(labelsize=12)

fig.subplots_adjust(wspace=0, hspace=0.5)
plt.show()

# Calculate the correlation coefficients - for each voxel
c_max=0
x_max=0
y_max=0
z_max=0
for x in range(64):
    for y in range(64):
        for z in range(50):
            real_response=img_data[x,y,z,:]
            c = np.corrcoef(predicted_response, real_response)[1:,0]
            if len(c)>0 and np.abs(c[0]) > c_max :
                c_max=c
                x_max=x
                y_max=y
                z_max=z

print(x_max)
print(y_max)
print(z_max)
# Find the voxel with the highest correlation coefficient
strongest_correlated = img_data[x_max,y_max,z_max,:]

# Create the plots
fig, ax = plt.subplots(1,1,figsize=(15, 5))
ax.plot(strongest_correlated, lw=3)
ax.plot(design_matrix[1,:], lw=3)
ax.set_xlim(0, acq_num-1)
ax.set_xlabel('time [volumes]', fontsize=20)
ax.tick_params(labelsize=12)
plt.show()
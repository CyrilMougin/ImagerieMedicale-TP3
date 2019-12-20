# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:00:18 2019

@author: Julien
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

img=nib.load('Data/fmri.nii')
img_data = img.get_data()
img_header = img.header

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
ax.set_title("Intensity of a random voxel", fontsize=25)
ax.tick_params(labelsize=12)
plt.show()

# Average all volumes
mean_data = img_data.mean(axis=3)

# Sample rate and desired cutoff frequencies (in Hz).
fs = 1000
lowcut = 0.1
highcut = 0.3

# Generate the time vector properly
t = np.linspace(0, 255, 85, endpoint=False)

 # Cut-off frequency of the filter
w = [lowcut, highcut]
b, a = butter(13, w, 'bandpass')


# Create the design matrix
constant = np.ones(acq_num)
with open("Data/ideal.txt") as f:
    content = f.readlines()
predicted_response= [float(x.strip()) for x in content] 

design_matrix = np.array((constant, predicted_response))

fig.subplots_adjust(wspace=0, hspace=0.5)
plt.show()

img_correlation=np.zeros((64,64,50))
# Calculate the correlation coefficients - for each voxel
c_max=0
x_max=0
y_max=0
z_max=0

median = np.median(img_data)
for z in range(50):
    for x in range(64):
        for y in range(64):
            if np.mean(img_data[x, y, z, :]) > median :
                real_response=filtfilt(b, a, img_data[x, y, z, :])
                c = np.corrcoef(predicted_response, real_response)[1:,0]
                if c[0] > 0.35:     
                    img_correlation[x,y,z]=c[0]
                else:
                    img_correlation[x,y,z]=0
                if c[0] > c_max :
                    c_max=c
                    x_max=x
                    y_max=y
                    z_max=z

# Find the voxel with the highest correlation coefficient
strongest_correlated = img_data[x_max,y_max,z_max,:]

# Define the min-max scaling function
def scale(data):
    return (data - data.min()) / (data.max() - data.min())

# Scale the voxel with the highest correlation
strongest_correlated_scaled = scale(img_data[x_max,y_max,z_max,:])

# Create the plots
fig, (ax) = plt.subplots(1,1,figsize=(15, 5))
ax.plot(strongest_correlated_scaled, label='voxel timecourse', lw=3)
ax.plot(design_matrix[1,:], label='design matrix', lw=3)
ax.set_xlim(0, acq_num-1)
ax.set_ylim(-0.25, 1.5)
ax.set_xlabel('time [volumes]', fontsize=20)
ax.set_ylabel('scaled response', fontsize=20)
ax.tick_params(labelsize=12)
ax.legend()
plt.show()

#img_correlation=medfilt(img_correlation,kernel_size=2)
img_correlation=gaussian_filter(img_correlation, sigma=1)
mask=img_correlation>0.15

l=list()
for z in range(50):
    for x in range(64):
        for y in range(64):
            if mask[x,y,z]:
                l.append(np.array([x,y,z]))
            else:
                img_correlation[x,y,z]=0
               
nb_cluster=15
kmeans = KMeans(n_clusters=nb_cluster, max_iter=600, random_state=0, algorithm="full").fit(np.array(l))

for index,element in enumerate(l):
    img_correlation[element[0],element[1],element[2]]=kmeans.labels_[index]*10+100

projection=np.amax(img_correlation,axis=2)
fig, ax = plt.subplots(1,1,figsize=(18, 6))
ax.imshow(mean_data[:,:,20], cmap='gray')
ax.imshow(projection, cmap='jet')
ax.set_title('Projection 2D', fontsize=25)
ax.set_yticks([])
ax.set_xticks([])
ax.set_yticks([])   
plt.show()

simulation_period=50


array_processed = img_correlation
img_mask=nib.Nifti1Image(array_processed,affine=img.affine)
nib.save(img_mask, 'Data/fmri_mask.nii')



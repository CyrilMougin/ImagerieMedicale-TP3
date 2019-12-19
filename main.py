# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:00:18 2019

@author: Julien
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

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


# =============================================================================
# def eliminate_non_brain(img_data):
#     for t in range(0,85):
#         for i in range(len(img_data[0][0])):
#             slice = img_data[:, :, i,t]
#             median = np.median(slice)
#             indexes = slice<=median
#             img_data[:, :, i,t][indexes]=0
#     return img_data
# 
# 
# img_data = eliminate_non_brain(img_data)
# =============================================================================
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
                if c[0] > 0.3:     
                    img_correlation[x,y,z]=c[0]
                else:
                    img_correlation[x,y,z]=np.nan
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

# affichage slice par slice des correlations
for i in range(len(img_data[0][0])):
    fig, ax = plt.subplots(1,1,figsize=(18, 6))
    ax.imshow(mean_data[:,:,i], cmap='gray')
    ax.imshow(img_correlation[:,:,i], cmap='afmhot')
    ax.set_title('thresholded map (overlay)', fontsize=25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()
    
# Look at activation for 2 different voxels:
signal1 = img_data[x_max, y_max, z_max, :]
signal2 = img_data[0, 0, 0, :]
time = np.arange(0, 340, 4)
plt.plot(time, signal1, 'b')
plt.plot(time, signal2, 'r')
plt.legend(['Activation', 'No activation'], loc='best')
plt.show()
# Compute fft for a voxel:
# input signal is 85 long, 2**7 = 128 seems to be enough
fft1=np.fft.fft(signal1, n=85)
fft2=np.fft.fft(signal2, n=85)
freq_signal1 = np.squeeze(fft1)
freq_signal2 = np.squeeze(fft2)
timestep = 3
freq = np.squeeze(np.fft.fftfreq(85, d=timestep))
plt.plot(freq[:42], np.absolute(freq_signal1[:42]), 'b')
plt.plot(freq[:42], np.absolute(freq_signal2[:42]), 'r')
plt.xlim((0, 0.13))
plt.ylim((-100, 1000))
plt.legend(['Activation', 'No activation'], loc='best')
plt.xticks(np.arange(0, max(freq[:85]), 0.02))
plt.show()

simulation_period=50

# =============================================================================
# from scipy.signal import butter, lfilter, freqz
# 
# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
# 
# 
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
# 
# # Sample rate and desired cutoff frequencies (in Hz).
# fs = 85
# lowcut = 0.005
# highcut = 0.3
# 
# # Plot the frequency response for a few different orders.
# plt.figure(100)
# plt.clf()
# for order in [3, 6, 9]:
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     w, h = freqz(b, a, worN=2000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
# 
# plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#          '--', label='sqrt(0.5)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')
# 
# # Filter a noisy signal.
# t = np.linspace(0, 255, 85, endpoint=False)
# a = 0.02
# 
# y = butter_bandpass_filter(img_data[x_max, y_max, z_max, :], lowcut, highcut, fs, order=6)
# plt.plot(t, y, label='Filtered signal')
# plt.xlabel('time (seconds)')
# plt.hlines([-a, a], 0, 255, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')
# 
# plt.show()
# =============================================================================

img_mask=nib.Nifti1Image(mean_data+img_correlation,affine=img.affine) #header=nii.get_header(), affine=nii.get_affine()
plot_slice(100,img_mask.get_data()[:, :, 10],"Image source")
nib.save(img_mask, 'Data/fmri_mask.nii')



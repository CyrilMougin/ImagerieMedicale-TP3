import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load data
    datas = nib.load("Data/fmri.nii").get_fdata()
    print(datas.shape)
    # Look at activation for 2 different voxels:
    signal1 = datas[50, 48, 25, :]
    signal2 = datas[0, 0, 0, :]
    time = np.arange(0, 340, 4)
    plt.plot(time, signal1, 'b')
    plt.plot(time, signal2, 'r')
    plt.legend(['Activation', 'No activation'], loc='best')
    plt.show()
    # Compute fft for a voxel:
    # input signal is 85 long, 2**7 = 128 seems to be enough
    freq_signal1 = np.squeeze(np.fft.fft(signal1, n=2**7))
    freq_signal2 = np.squeeze(np.fft.fft(signal2, n=2**7))
    timestep = 3.864
    freq = np.squeeze(np.fft.fftfreq(2**7, d=timestep))
    plt.plot(freq[:63], np.absolute(freq_signal1[:63]), 'b')
    plt.plot(freq[:63], np.absolute(freq_signal2[:63]), 'r')
    plt.xlim((0, 0.13))
    plt.ylim((-100, 500))
    plt.legend(['Activation', 'No activation'], loc='best')
    plt.xticks(np.arange(0, max(freq[:63]), 0.02))
    plt.show()

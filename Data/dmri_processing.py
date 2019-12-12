import numpy as np
import nibabel as nib

#
# LOAD DATA and b-values, b-vectors
# (nibabel will do this for you for the Nifti)
# (b-values and b-vectors are text files)

dmri = nib.load("dmri.nii")

dmri_data =dmri.get_data()

b=np.loadtxt(fname = "gradient_directions_b-values.txt")

b_vectors=b[1:,0:3]
b_value=b[1,3]

#calcul de matrice B
B=np.zeros((b_vectors.shape[0],6))
B[:,0]=b_vectors[:,0]*b_vectors[:,0]
B[:,1]=b_vectors[:,0]*b_vectors[:,1]
B[:,2]=b_vectors[:,0]*b_vectors[:,2]
B[:,3]=b_vectors[:,1]*b_vectors[:,1]
B[:,4]=b_vectors[:,1]*b_vectors[:,2]
B[:,5]=b_vectors[:,2    ]*b_vectors[:,2]





ensemble_D=np.zeros((dmri_data.shape[0],dmri_data.shape[1],dmri_data.shape[2]))

for x in range (dmri_data.shape[0]) :
    print(x)
    for y in range (dmri_data.shape[1]) :
        for z in range(dmri_data.shape[2]) :
            S0=dmri_data[x,y,z,0]
            S=np.array(dmri_data[x,y,z,1:])
            X=S/S0
            X=np.log(X)
            X=-X/b_value
            
            D=np.linalg.inv(np.dot(B.T,B)).dot(B.T).dot(X)
            ensemble_D[x,y,z]=D



#
# Compute tensors and FA/ADC
#  - tensor fit
#  - SVD  -> eigen values -> FA/AD
#

#
# Tracking deterministe sur la direction principale -> eigen vector1
#


#
# Save streamlines in a .trk format or .tck format to visualize in MI-Brain
#


#
# The DIPY gallery should help you a lot:
#  http://nipy.org/dipy/examples_index.html
#

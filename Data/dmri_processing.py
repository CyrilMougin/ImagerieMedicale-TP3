import numpy as np
import nibabel as nib

from dipy.viz import window, actor
import dipy.data as dpd
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
import math

#
# LOAD DATA and b-values, b-vectors
# (nibabel will do this for you for the Nifti)
# (b-values and b-vectors are text files)

dmri = nib.load("dmri.nii",mmap=False)

dmri_data =dmri.get_data()

# =============================================================================
# estimation des tenseurs
# =============================================================================
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



ensemble_D=X = np.empty(shape=[dmri_data.shape[0], dmri_data.shape[1],dmri_data.shape[2],6])


for x in range (dmri_data.shape[0]) :
    print(x)
    for y in range (dmri_data.shape[1]) :
        for z in range(dmri_data.shape[2]) :
            S0=dmri_data[x,y,z,0]
            if (S0>60) :
                S=np.array(dmri_data[x,y,z,1:])
                X=S/S0
                X=np.log(X)
                X=-X/b_value
                
                D=np.linalg.inv(np.dot(B.T,B)).dot(B.T).dot(X)
                ensemble_D[x,y,z]=D
            else :
                ensemble_D[x,y,z]=np.array([0, 0, 0, 0, 0, 0])

#enregister les tenseurs
img = nib.Nifti1Image(ensemble_D, np.eye(4))
img.set_data_dtype(np.float32)
header_info = img.header
header_info['pixdim'][1:5]  = dmri.header['pixdim'][1:5]
x = nib.Nifti1Image(ensemble_D, np.eye(4), header_info)
nib.save(img, filename="tensors.nii")


# =============================================================================
# SVD+ carte FA/ADC
# =============================================================================
evals=np.empty(shape=[dmri_data.shape[0], dmri_data.shape[1],dmri_data.shape[2],3])
evecs=np.empty(shape=[dmri_data.shape[0], dmri_data.shape[1],dmri_data.shape[2],3,3])

fa=np.zeros(shape=[dmri_data.shape[0], dmri_data.shape[1],dmri_data.shape[2]])
adc=np.empty(shape=[dmri_data.shape[0], dmri_data.shape[1],dmri_data.shape[2]])


for x in range (dmri_data.shape[0]) :
    print(x)
    for y in range (dmri_data.shape[1]) :
        for z in range(dmri_data.shape[2]) :
            img=ensemble_D[x,y,z]
            D=np.zeros((3,3))
            D[0,:]=img[0:3]
            D[1,1:]=img[3:5]
            D[1,0]=img[1]
            D[2,0]=img[2]
            D[2,1]=img[4]
            D[2,2]=img[5]
            if (dmri_data[x,y,z,0]<60) :
                if (D[0,0]!=-np.inf and D[0,0]!=np.inf  and math.isnan(D[0,0])==False) :
                    u,s,vh= np.linalg.svd(D, full_matrices=False)
                    
                    evals[x,y,z]=s
                    evecs[x,y,z]=vh
                fa[x,y,z]=0
                adc[x,y,z]=0
            else :
                if (D[0,0]!=-np.inf and D[0,0]!=np.inf  and math.isnan(D[0,0])==False) :
                    u,s,vh= np.linalg.svd(D, full_matrices=False)
                    
                    evals[x,y,z]=s
                    evecs[x,y,z]=vh
                    fa[x,y,z]=np.sqrt(1/2)*np.sqrt(np.power(s[0]-s[1],2)+np.power(s[1]-s[2],2)+np.power(s[2]-s[0],2))/np.sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2])
                    adc[x,y,z]=(s[0]+s[1]+s[2])/3
                

def afficher_tenseurs(fa_,evec,eva) :
    cfa = dti.color_fa(fa_, evec)
    sphere = dpd.default_sphere
    ren = window.Renderer()
    ren.add(actor.tensor_slicer(eva, evec, scalar_colors=cfa, sphere=sphere,
                                scale=0.5))
    window.record(ren, out_path='tensor.png', size=(1200, 1200))

afficher_tenseurs(fa, evecs,evals)

def enregistrer_cartes(fa_, adc_) :
    img = nib.Nifti1Image(fa_, np.eye(4))
    img.set_data_dtype(np.float32)
    nib.save(img, filename="fa.nii")
    
    img = nib.Nifti1Image(adc_, np.eye(4))
    img.set_data_dtype(np.float32)
    nib.save(img, filename="adc.nii")
    
enregistrer_cartes(fa, adc)

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

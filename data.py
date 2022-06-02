#Maria Williams
#6/1/22: loading and manipulating nifti files
#This was a helpful resource: https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html

!pip install nibabel
import nibabel as nib

imgfile = nib.load('ds107_sub001_highres.nii')
img = img.get_data()
print(img.shape)
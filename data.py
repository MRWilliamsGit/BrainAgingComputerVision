#Maria Williams
#6/1/22: loading and manipulating dcm files

import pydicom as dicom
import matplotlib.pylab as plt

img = dicom.dcmread('testimg.dcm')
plt.imshow(img.pixel_array, cmap=plt.cm.bone)
plt.savefig('testimg')
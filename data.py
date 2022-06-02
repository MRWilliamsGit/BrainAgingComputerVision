#Maria Williams
#6/2/22: loading and manipulating dcm files

import pydicom as dicom
import matplotlib.pylab as plt

#load and convert dcm to jpg
from pydicom import dcmread
from pydicom.pixel_data_handlers import convert_color_space

image_path = 'BrainAgingComputerVision/testimg1.dcm'
img = dicom.dcmread(image_path)
new_image = img.pixel_array.astype(float)

#convert to jpg
scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
scaled_image = np.uint8(scaled_image)
final_image = Image.fromarray(scaled_image)
final_image.show()
final_image.save('testimgdone.jpg')
print(final_image.size)
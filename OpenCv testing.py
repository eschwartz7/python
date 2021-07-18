import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from ImageCollection import exposure_list
import glob
from subprocess import call

images = glob.glob('*.jpg')
pixel_count = 0
intensity_count = 0
exposure_values = [int(image.replace('.jpg', '')) for image in images]
print("Exposures: ", exposure_values)
print("Image List: ", images)
intensity_values = []

for image in images:
    #Read each image and crop a 50x50 patch of pixels
    img = cv2.imread(image, 0)

    cropped_img = img[1000:1050, 1500:1550]
    rows, cols = cropped_img.shape

    #Iterate through each pixel and find its intensity value
    for i in range(rows):
        for j in range(cols):
            intensity = img[i,j]
            #print("Intensity", intensity)
            intensity_count += intensity
            pixel_count += 1

    #Find average intensity
    avg_intensity = intensity_count / pixel_count
    intensity_values.append(avg_intensity)
    print("Image Name: ", image, "Average Intensity: ", avg_intensity)

cv2.destroyAllWindows()

#Plot Exposure vs Intensity Graph
plt.plot(exposure_values, intensity_values)
plt.title("Intensity vs Exposure Graph")
plt.xlabel("Exposure Time (us)")
plt.ylabel("Pixel Intensity")
plt.show()


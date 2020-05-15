#import required packages 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from os import path



#load input image from Sample images file folder or set new path for new input image 

#--create a counter that upticks when a sequentially named image is found: 
#--all images must be named img1 -> imgX where X is the total number of images
#--this counter tracks how many images we are matching stored in 'filecount'
i=1
pathPresence = True
while pathPresence == True:
    pathPresence = path.exists('./Documents/project_spring_2020_burkekindel_pkg/Sample_images/img%d.jpg' %i )
    print(pathPresence)
    i+=1
    print(i)

print(i-2)
filecount = i-2

i=1

#--create a dictionary to hold our identified number of images so they can be easily looped through
#--length of this dictionary will serve as our new filecount as well
imgDict = {}
while i <= filecount:

    input_img = cv2.imread('./Documents/project_spring_2020_burkekindel_pkg/Sample_images/img%d.jpg' %i)

  #scale input image and set threshold
    scale_percent = 8 #percent of original image size
    width = int(input_img.shape[1]* scale_percent/100)
    height = int(input_img.shape[0]* scale_percent/100)
    dim = (width, height)
    input_img_resized = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)
  #  input_img_gs = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2GRAY)
    thresh, input_img_bw = cv2.threshold(input_img_resized,5,250,cv2.THRESH_BINARY)
    imgDict['%d' %i] = input_img_bw
    i+=1
    
    
    import math
import os

#load reference image file and set output file name
template_img_dir = './Documents/project_spring_2020_burkekindel_pkg/Reference_images'
result_grid_filename = './Documents/project_spring_2020_burkekindel_pkg/grid.jpg'
result_figsize_resolution = 40 # 1 = 100px

#create list variable that lists files in reference images, and length variable for counting iterations 
images_list = sorted(os.listdir(template_img_dir))
images_count = len(images_list)
print('Images: ', images_list)
print('Images count: ', images_count)

# Calculate the grid size:
grid_size = math.ceil(math.sqrt(images_count))

# Create plt plot:
fig, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))


current_file_number = 0

# create an image grid of reference images that will be compared to the input image. 
for image_filename in images_list:
    x_position = current_file_number % grid_size
    y_position = current_file_number // grid_size
    plt_image = plt.imread(template_img_dir + '/' + images_list[current_file_number])
    axes[x_position, y_position].imshow(plt_image)
    
    axes[0,0].set_title('Bregma: -0.46', fontsize=50)
    axes[0,1].set_title('Bregma: -0.58', fontsize=50)
    axes[0,2].set_title('Bregma: -0.70', fontsize=50)
    axes[1,0].set_title('Bregma: -0.82', fontsize=50)
    axes[1,1].set_title('Bregma: -0.94', fontsize=50)
    axes[1,2].set_title('Bregma: -1.00', fontsize=50)
    axes[2,0].set_title('Bregma: -1.06', fontsize=50)
    axes[2,1].set_title('Bregma: -1.12', fontsize=50)
    axes[2,2].set_title('Bregma: -1.32', fontsize=50)
    
    print((current_file_number + 1), '/', images_count, ': ', image_filename)

    current_file_number += 1

plt.subplots_adjust(left=0.0, right=.9, bottom=0.0, top=.9)
plt.savefig(result_grid_filename)

#save template grid for future reference 
template_grid = cv2.imread('./Documents/project_spring_2020_burkekindel_pkg/grid.jpg')

#resize template grid
scale_percent = 25 #percent of original image size
temp_width = int(template_grid.shape[1] * scale_percent/100)
temp_height = int(template_grid.shape[0]* scale_percent/100)
temp_dim = (temp_width, temp_height)
template_grid= cv2.resize(template_grid, temp_dim, interpolation = cv2.INTER_AREA)

#match template grid to sample/input image and frame reference image with most similarity 

#--Loop runs the length of imgDict
#--Creates a copy of template_grid each time to draw our rectangle on
#--Spits out image named as the input was minus 'img' 
i=1
while i <= len(imgDict):
    tempGrid = template_grid.copy()
    #--it is very dumb that the thing we compare our images too is called the image and the images we compare are called the template
    res = cv2.matchTemplate(tempGrid, imgDict['%d'%i], cv2.TM_CCOEFF)
    cv2.imwrite('./Documents/project_spring_2020_burkekindel_pkg/Processed_images/res%d.jpg' %i, res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1]+height)


    cv2.rectangle(tempGrid, top_left, bottom_right, 255, 2)

    #save framed grid to Processed images file './Processed_images/___.jpg', 
    #before running code set file name that corresponds to original input image 
    output_filepath = './Documents/project_spring_2020_burkekindel_pkg/Processed_images/%d.jpg'%i
    status = cv2.imwrite(output_filepath, tempGrid)

    #double check that file has saved 
    print('New image has been saved: ', status)
    i+=1
    print(i)
# project_spring_2020

[![CircleCI](https://circleci.com/gh/biof309/project_spring_2020/tree/master.svg?style=shield)](https://circleci.com/gh/biof309/project_spring_2020/tree/master)
Julia Burke
Morgan Kindel 
Final Project Outline
2/26/2020

Histological techniques provide us with important information about the brain, such as protein levels, which brain regions neurons are projecting to, and cell-type, at a high spatial resolution. These techniques involve sectioning the brain into many slices, often less than 50 micrometers in thickness, mounting the slices onto a slide, and imaging each slice, one at a time. Several microscopes on the NIH campus, such as the light sheet microscope, collect brain images in tiles since the total area is too large to fit in the frame of view. In order to view this data as a single composite image, the individual tiles must be stitched together. Several parameters can be changed to affect the output, such as tile overlap, image averaging, or style of blend. Using the image name and metadata, a python program can be utilized to stitch the image. Often, one of the biggest flaws with stitching programs is the appearance of lines where each tile was integrated. To combat this, one can perform image deconvolution to create a cleaner picture. This process uses an algorithm to reverse any distortion that occurs during image collection. This is performed using a point spread function; a function that describes the distortion based on the path through the instrument. Once this is determined, the inverse can be used to undo the distortion. 
After an individual image is processed and can be compared with those of other experimental groups (i.e. two or more brains), it is crucial to keep the comparison consistent—that is, matching the slices from one brain to those of another. This is often done using a common reference point on the skull, called the bregma (a slice that is directly below the bregma has bregma coordinates of 0, a slice is 0.50 mm after the bregma is ‘-0.50’, and a slice that is 0.50 mm before the bregma is ‘0.50’. Mismatched coordinates and slices result in inconsistent findings, or incorrect conclusions. For example, one region of the brain, the paraventricular thalamus (PVT) is located in the middle of the brain and is spread over many slices. As slices move from the front of the brain to the back, the shape of the PVT changes in subtle, yet important ways. The most common way of accurately tracking these shape changes is to mount every brain slice in order; and manually calculate bregma coordinates. This is extremely time-consuming, especially with larger experimental groups and/or additional tissue processing (i.e. immunohistochemistry), and many researchers instead end up estimating brain region/coordinates, by visually matching sample images with those in a reference brain atlas. Although this method works well when identifying approximate coordinates for distinct brain regions (such as the prefrontal cortex and the hindbrain), using it to distinguish between proximal slices, such as those that contain PVT, lacks accuracy and precision, resulting in selecting an incorrect region of interest (ROI).
  To address this issue, our script will 1) stitch the individual image 2) identify the bregma coordinates for images 3) order those images, and 4) outline the PVT based on these coordinates. After stitching, the script will compare features (based on contrast of brain vs. slide) of each image with those of a reference atlas to output a bregma coordinate and verify these matches by using the slice thickness and interval. In the second part, the images will be sorted based on these coordinates. In the last part, the range of images with coordinates that include the PVT will independently processed; the script will include PVT outline shapes that correspond to each coordinate, and these shapes will be added to the images and saved as an additional file.  


#Check if two images the same size are the same 

import cv2
import numpy as np

Atlas = cv2.imread("INSERT HERE")

New = cv2.imread("INSERT HERE")

# Check if 2 images are equals
if Atlas.shape == New.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(Atlas, New)
    b, g, r = cv2.split(difference)


    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely Equal")
		
cv2.imshow("Atlas", atlas)
cv2.imshow("New", new)
cv2.waitKey(0)
cv2.destroyAllWindows()







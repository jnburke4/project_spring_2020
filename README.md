# project_spring_2020

[![CircleCI](https://circleci.com/gh/biof309/project_spring_2020/tree/master.svg?style=shield)](https://circleci.com/gh/biof309/project_spring_2020/tree/master)

Julia Burke
Morgan Kindel 
Final Project Outline
2/26/2020
=======
Histological techniques provide us with important information about the brain, such as protein levels, which brain regions neurons are projecting to, and cell-type, at a high spatial resolution. These techniques involve sectioning the brain into many slices, often less than 50 micrometers in thickness, mounting the slices onto a slide, and imaging each slice, one at a time.
To compare brains from different experimental groups and different brain regions among each other, a common reference point on the skull, called the bregma is used to determine loction (a slice that is directly below the bregma has bregma coordinates of 0, a slice is 0.50 mm after the bregma is ‘-0.50’, and a slice that is 0.50 mm before the bregma is ‘0.50’.) Mismatched coordinates and slices result in inconsistent findings, or incorrect conclusions. For example, one region of the brain, the paraventricular thalamus (PVT) is located in the middle of the brain and is spread over many slices. As slices move from the front of the brain to the back, the shape of the PVT changes in subtle, yet important ways. The most common way of accurately tracking these shape changes is to mount every brain slice in order; and manually calculate bregma coordinates. Many researchers end up estimating brain region/coordinates, by visually matching sample images with those in a reference brain atlas. Although this method works well when identifying approximate coordinates for distinct brain regions (such as the prefrontal cortex and the hindbrain), using it to distinguish between proximal slices, such as those that contain PVT, lacks accuracy and precision, resulting in selecting an incorrect region of interest (ROI).
  To address this issue, our script will identify the bregma coordinates for images by matching an input image to a refrence image. 
  
  Our code compares an input image to refrence images; it is set up so that it cycles through each of the images in the input, and saves a new image with the matched refrence in a box. 

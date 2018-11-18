# Shape-preservingLoss
The implementation of the MICCAI paper entitled "A deep model with shape-preserving loss for gland instance segmentation". The shape-preserving loss is implemented based on Caffe.

The loss requires four inputs:

1. logits by the model
2. target label/ground truth
3. IDMask:
   Given a target segmentation map, the contour is extracted and divided into small segments indexing from -1 to -N. For each contour        segment indexing -i, all pixels located with the searching range of the contour segment are assigned with the value i. All background      pixels are assigned with the value 0.
4. distance matrix:
   For each pixel in the target label, we will calculate its distances to the frist and the second closest gland instances. Based on the      distances, we will assign a weight to each pixel in the loss, in order to deal with close gland instances. 
   
The corresponding MATLAB codes are used to generate both IDMask and distance matrix for each segmentation map. The main file is GenerateIDMask.m

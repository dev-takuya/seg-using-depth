# surgical-instrument-segmentation-using-estimated-depth-from-monocular-laparoscopic-image
It is necessary to extract surgical instruments from laparoscopic images in order to improve the safety of laparoscopic surgery using a surgery support system. It is reported that the segmentation accuracy can be improved by using color and depth information. In this paper, we propose a U-Net based image segmentation network using the estimated depth by cGAN as well as color for improving the accuracy. We conducted experiments using 4-fold cross validation with 1,800 images in the MICCAI challenge dataset, and confirmed that the proposed method achieved the average IoU of 88% and the average Dice coefficient of 93%. The proposed method reduced the excessive extraction and improved the extraction accuracy by using the estimated depth information as well as color information.

## Requirements
### Library
```
Python==3.7.3
TensorFlow==1.14.0
Keras==2.2.4
```
### Dataset
MICCAI2017 Dataset: [MICCAI 2017 Robotic Instrument Segmentation Sub-Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

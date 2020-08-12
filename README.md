# Concrete Crack Segmentation

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


A __novel semantic segmentation architecture for crack detection.__Semantic segmentation is the process of classifying each pixel of an image into distinct classes using deep learning.
Classical image processing methods demands heavy feature engineering,as well as they are not that precise,when it comes to manual exraction of relavant features in real life scenarios.
Hence,We tried to make a  modified U-net based architecture , and replaced all the convolutional layers with residual blocks(inspired from ResNet architecture) in both encoder and decoder region
for evaluation of our model we have used Dice Loss as our objetive function and F1 score as a metric.Other than that,for better convergence and optimisation, a learning rate schedular and AMSGRAD optimiser was utilised


## Table of contents
* [Contributors](#Contributors)
* [Introduction](#Introduction)
* [Dataset & Preprocessing](#Dataset-And-Preprocessing)
* [Model Architecture](#Network-Architecture)
* [Loss Function & Optimizer](#Loss-Function-And-Optimizer)
* [Learning Rate](#Learning-Rate)
* [Result](#Result)
* [Citations](#Citations)

### Contributors:
This project is created by the joint efforts of
* [Subham Singh](https://github.com/Subham2901)
* [Sandeep Ghosh](https://github.com/Sandeep2017)
* [Amit Maity](https://github.com/Neel1097)

### Introduction:
A crack is a linear fracture in concrete which extends partly or completely through the member. A crack occurs, when the tensile stresses of a concrete exceeds the tensile capacity due to various natural and man-made reasons. The utility of concrete can be seen in almost everywhere, from buildings, bridges, to other structures. Thus, when a crack occurs in a concrete slab, it can be an indication of some major structural problems in the whole architecture, which possess a potential risk of some serious accident. There lie some existing ways to detect cracks, starting from visual inspection and monitoring to other non-destructive techniques (NDT) which uses various image processing techniques to segment cracks, but due to unavoidable noise in images, the segmentation of the cracks from its background, isn’t that precise. In this paper, we have tried to design a simple deep learning algorithm, which holds the potential to detect cracks of any thickness. As a result, it bypasses the need of manual feature extraction, by learning the essential features necessary for segmenting the crack from its background by classifying each pixel as a crack or not.    

## Image Showing Crack On Concrete
<img src='/Images/intro/intro1.jfif'>

### Dataset And Preprocessing:
We evaluated our method on different dataset. They are as follows:
* 	Crack500  – The dataset consists of 500 images and corresponding masks of size (3264x2448). We used 80% data for training and 20% for validation.
* 	DeepCrack  – The dataset consists of 537 manual annotated images. This dataset was also splitted the same way as above.
#### Image Augmentation:
We augmented our data on the fly using the [Albumentations library](https://albumentations.ai/). We applied random flips and rotations with random changes in lighting by increasing/decreasing contrast, gamma & brightness. We also applied random distortions like elastic distortion, grid distortion and optical distortion

### Network Architecture:
 The base network architecture used here is based on U-Net, which was originally designed for segmentation of microscopic cells with limited number of annotated data. This is highly correlated with the task of crack segmentation. Therefore, this architecture is ideally suited for this work. Further, we have replaced all the Conv2d blocks with residual blocks(inspired form ResNet) such that we can make our model much deeper as well as can resolve the issue of the vanishing gradients. We have used the residual blocks in both downsampling(encoder) as well as upsampling(decorder) blocks.
 #### The skip Block & the BLock :
 ![](https://github.com/Subham2901/Concrete_Crack_Segmentation/blob/master/Images/MOdel/model%20image%20final2.png)
 #### The Complete Model Architecture :
 ![](https://github.com/Subham2901/Concrete_Crack_Segmentation/blob/master/Images/MOdel/model%20image%20final1.png)
 

### Loss Function And Optimizer:
##### * Loss Function:
We have passed the network’s output layer through a sigmoid function such that each pixel in the output layer would have a range between [0,1], which determines the probability that a crack is present in each pixel or not. As a result, we have noticed that the non-crack pixels outnumber the crack pixels by a huge ratio. Hence, there was a huge class imbalance. Thus, we chose the dice coefficient loss as it directly optimizes the dice score. And moreover, it’s directly equivalents to the F1 score too
##### * Optimizer:
The optimizer that we have used here to optimize our model is ADAM and Beyond. Which uses a new exponential moving average AMSGRAD. The AMSGRAD uses a smaller learning rate in comparison to ADAM. In case of ADAM the decrement or decay of learning rate is not guaranteed where as AMSGRAD  uses smaller learning rates , it maintains the maximum of  all the learning rates until the present time step and uses that maximum value for normalizing the running average of the gradient instead of learning rate in ADAM or RMSPROP. Thus, it converges better than ADAM or RMSPROP
### Learning Rate:
The learning rate we have used here is not constant throughout the training of the data, instead we have used a learning rate schedular, which increases/decreases the learning rate gradually after every fixed set of epochs such that  we can attain the optimum convergence by the end of our training of the data.

### Result:
__We have tested our model architecture on two datasets namely Crack500 & DeepCrack__
##### And we have achieved the following results:
##### ![](https://github.com/Subham2901/Concrete_Crack_Segmentation/blob/master/Images/Result/Result.JPG)
#### Prediction Images:
__CRACK500__
#### ![](https://github.com/Subham2901/Concrete_Crack_Segmentation/blob/master/Images/crack500/finalimage.JPG)
__DeepCrack__
#### ![](https://github.com/Subham2901/Concrete_Crack_Segmentation/blob/master/Images/deepcrack/final.JPG)
### Citations:
# Please cite the following papers while using the follwing datasets:
* __CRACK500__
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}' 

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}

* DEEPCRACK
```
@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}
```



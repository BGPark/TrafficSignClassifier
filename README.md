## Traffic Sign Recognition Program

Overview
---
This software classify traffic signs. The main purpose of this project is to learn and understand CNN(Convolutional Neural network) as a part of deep learning. Please check [my notebook](https://github.com/BGPark/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb) for more detail. 

The Steps
---
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Dataset
In this project I train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried out the model on images of German traffic signs that can find on the web.

## Dataset Normalize and Uniformed Train Data
In the pre-processing step, I proceeded to convert the image RGB to a real number ranging from -1 to 1.((color - 128)/256) It's probably one of the lightest ways to handle it. In the case of traffic signs, the outline is also important, but the color values are also considered important, so grayscale processing is not done.
And the distribution of images for each class in train dataset is not balanced. I uniformed the number of each class image used in one epoch for more equal learning in network learning.

## Train Data Augment
As the same data is repeated and the test set is configured, it is manipulated to receive the same image at the time of learning through the random effect. This processing is used only for learning and not for evaluation or testing.

## Model Design
My traffic sign classifier design is as below:

Layer | Desc.
------------ | -------------
Input | 32x32x3
Conv | 1x1x8 filter, 1x1 stride, VALID
relu | 
Conv  | 5x5x32 filter, 1x1 stride, VALID
relu| 
maxpool | 2x2, VALID
Conv | 5x5x64 filter, 1x1 stride VALID
relu|
maxpool | 2x2, VALID
Conv | 3x3x128 filter, 1x1 stride, VALID
relu|
flatten|
dense | 1152x400
dropout|
dense | 400x43
softmax| 

# Model Training
Here is final configuration for training.
* EPOCHS = 200
* BATCH_SIZE = 128
* Adam Optimizer with learning rate : 0.001
* Hyper parameter Initilizer : mu=0, sigma=0.1

## German traffic sign Performance
Dataset | Accuracy
----------------|-----------------
Validation|Over 97%
Test|94.4%

## Web sign image Performance
Dataset | Accuracy
----------------|-----------------
10 Web Images| 90%

## Conclusion
The main challenge of this project was to make CNN a better way to understand CNN than to image processing and to make the best predictions that CNN could have, rather than raising the predictability rate through direct transformations on images.

The first attempt was to apply only normalization to the original images to fit ML. I tested it with changing the network and increasing Epoch. As a result of the process, overfitting occurred and the prediction rate was too low. In particular, for classes with a small number of samples, there were classes whose accuracy was less than 50%.

 As a second approach, we assume that the number of learning of the class is insufficient, and applied uniformed. However, the results were somewhat improved but did not show a satisfactory level of predictability.
 
Third, the method of transforming the images used in learning into a random method was applied as the approach, and then the evaluation data and the real data were processed in a manner of not applying the image transformation.

Finally, the simultaneous uniformed and random effects yielded a good level of accuracy.

However, there are a lot of ways to get higher accuracy, and a better solution would require more computing power, so I'll go on to the next step.

## References
* [LeNet by Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* [Traffic Sign Recognition with Multi-Scale Convolutional Networks by Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

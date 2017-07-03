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
I create traffic sign classifier network based on LeNet4. However the original LeNet is desinged for Grayscale color map. So, I extend it for RGB maps. After simply increasing the color map and proceeding with the learning, overfitting occurred. So I modified the network and decided to add a dropout, which worked great.

My network design is as below:

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
softmax| 43 classes

# Model Training
Here is final configuration for training.
* EPOCHS = 200
* BATCH_SIZE = 128
* Adam Optimizer with learning rate : 0.001
* Hyper parameter Initilizer : mu=0, sigma=0.1

I used train data to learn by using 1.'randomly mixed', 2.'randomly augmented', and 3.'uniformed total number per class' data. However, the verification set and the test set did not add any ordering or effecting transformations. ( Therefore, the tensor accuracy and loss looks more shaking. You can check the final score in my [notebook log](https://render.githubusercontent.com/view/ipynb?commit=9b919c890ee45e7520adff80cd414be349017273&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f42475061726b2f547261666669635369676e436c61737369666965722f396239313963383930656534356537353230616466663830636434313462653334393031373237332f547261666669635f5369676e5f436c61737369666965722e6970796e62&nwo=BGPark%2FTrafficSignClassifier&path=Traffic_Sign_Classifier.ipynb&repository_id=96008020&repository_type=Repository#Train,-Validate-and-Test-the-Model))
( The graph of the validation set has the effect of successive class placement, so that when there are a small number of classes in one batch, it can be seen as a large number of generations that can not distinguish the class correctly. )

![Accuracy & loss](https://github.com/BGPark/TrafficSignClassifier/blob/master/loss_n_accuracy.png)


## German traffic sign Performance
Dataset | Accuracy
----------------|-----------------
Validation dataset Accuracy | 0.9617
Test dataset Accuracy | 0.9460

## Web sign image Performance
Dataset | Accuracy
----------------|-----------------
15 Web Images| 85.71%

Here are the five with the lowest confidence in 15 new test samples.

I have never learned the "No Stop" sign, however my model are very confident of "Stop". If there is a sign that is not included in learning, it may cause a great issue. And, I could see that the accuracy of the sign, which contains a somewhat complicated figuer in the middle of sign, is lowered. I think it would be okay to reinforce the network or to increase the size of the images used for learning, or it may be better to proceed with sufficient learning.


### No Stop - doesn't exist in the training set
![No Stop - doesn't exist in the training set](https://github.com/BGPark/TrafficSignClassifier/blob/master/signs/class%2099.jpg)
* class 14 : score(99.981%) Stop
* class 18 : score(0.019%) General caution
* class 25 : score(0.000%) Road work
* class 12 : score(0.000%) Priority road
* class 13 : score(0.000%) Yield

### Double curve
![Double curve](https://github.com/BGPark/TrafficSignClassifier/blob/master/signs/class%2021.jpg)
* class 25 : score(73.560%) Road work
* class 28 : score(26.397%) Children crossing
* class 18 : score(0.028%) General caution
* class 26 : score(0.006%) Traffic signals
* class 20 : score(0.005%) Dangerous curve to the right

### Right-of-way at the next intersection
![Right-of-way at the next intersection](https://github.com/BGPark/TrafficSignClassifier/blob/master/signs/class%2011.jpg)
* class 11 : score(51.790%) Right-of-way at the next intersection
* class 27 : score(48.210%) Pedestrians
* class 18 : score(0.000%) General caution
* class 28 : score(0.000%) Children crossing
* class 25 : score(0.000%) Road work

### Road narrows on the right
![Road narrows on the right](https://github.com/BGPark/TrafficSignClassifier/blob/master/signs/class%2024.jpg)
* class 24 : score(97.695%) Road narrows on the right
* class 25 : score(2.305%) Road work
* class 29 : score(0.000%) Bicycles crossing
* class 28 : score(0.000%) Children crossing
* class 31 : score(0.000%) Wild animals crossing

### Speed 20km
![Speed 20km](https://github.com/BGPark/TrafficSignClassifier/blob/master/signs/class%200.jpg)
* class  0 : score(93.394%) Speed limit (20km/h)
* class  4 : score(5.022%) Speed limit (70km/h)
* class  8 : score(1.270%) Speed limit (120km/h)
* class  1 : score(0.313%) Speed limit (30km/h)
* class  2 : score(0.001%) Speed limit (50km/h)



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

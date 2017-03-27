# Traffic Sign Classification

This project consists of an elaborated neural network to classify German Traffic Signs. It uses a custom variation of LeNet Neural Network extended with two extra Convolution Layers to identify and classify Traffic Signs from the German Traffic Sign Data set. The script itself can work with image of any size, as long as it can be converted to 32x32 pixels without irreversibly deformed it.

The main goals of this project is to develop a neural network that gets as input a raw image of a German Traffic Sign and outputs a label (between 0 and 41) that correctly classifies it. Figure 1 depicts an example of traffic sign image. 

![alt text][image0]


[//]: # (Image References)

[image0]: ./pre/original.png "Traffic Sign Image Example"
[image1]: ./layers/classes.png "Classes Distribution"
[image2]: ./pre/grayscale.jpg "Grayscaling Operation"
[image3]: ./pre/standarize.jpg "Standarize Operation"
[image4]: ./pre/all.jpg "All Pre-Processing Operations"
[image5]: ./preprocessing.png "Pre-Processing Image Before Neural Network"
[image6]: ./pre/pre-processing-functions.png "Pre-Processing Functions"
[image7]: ./augment-data.png "Augment Image before Pipeline"
[image13]: ./pre/pre-augmented.png "Augmented Image"
[image14]: ./pre/pre-processed.png "Pre-Processed Image"


[image8]: ./german_signs/original/50km.jpg "50 km/h"
[image9]: ./german_signs/original/60km.jpg "60 km/h"
[image10]: ./german_signs/original/children.jpg "Children Crossing"
[image11]: ./german_signs/original/right.jpg "Keep Right"
[image12]: ./german_signs/original/stop.jpg "Stop Sign"


## Access

The source code for this project is available at [project code](https://github.com/otomata/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary and Exploratory Visualization

The data set provided by the German Neuroinformatik Institude consists of more than 50.000 images divided into 42 classes. All images are loaded using the pickle package and pre-analyzed using the helper functions from the numpy and matplotlib packages. The summary statistic of the Traffic Signs data set are presented below:

*Size of Training set:    34799
*Size of Validation set:   
*Size of Test set:         12630
*Image Shape: 		   32x32x3 (RGB)
*Number of unique classes: 42

Figure 2 describes the class distribution for the training set. From this figure, it is possible to assert that the distribution of images is not uniform between the classes, we can see a maximum difference factor of 10 between two given classes  (~ 200 to ~ 2000). Even though this difference is not sufficient to consider this data set as distorted (skwed), some classes will have more chances to be classified correctly than others.

![alt text][image1]

The code for this part is contained in the code cell of the second setp of the IPython notebook. 

## Design and Test a Model Architecture

### Pre-Processing

Pre-processing the data set is an important step not only to reduce the training time but also to achieve more favorable accuracy rates. For this model, we have tried different image processing techniques offered by the TensorFlow library (tf.image). We decide to use only Tensor Flow functions into this step because they fit perfectly as part of the pipeline and are execute online with the neural network. Actually, the pre-processing functions are the first step of the pipeline, right before the images are feed into the network (see Figure 3).

![alt text][image5]

Among the several image processing functions the TensorFlor library offers, the following functions were selected to be tested for this project: contrast adjust, brightness adjust, image standardization(scales image to have zero mean and unit norm), gray scale transformation, hue adjust and glimpse extraction. Figure 4 presents the output of each of these functions over a given image from the data set.

![alt text][image6]

For the neural network developed for this project, only the Gray Scale and Standardized functions resulted in significant improvements. It is is important to point out that Gray Scale alone is enough to hold satisfactory results, the use of Standardized only helped to improve around %1 the accuracy of our classifier. The use of Contrast, Brightness and Hue did not show any compelling improvement in our tests. Figure 5 shows a final pre-processed image.

![alt text][image14]

The code for this part is contained in the code cell of the second setp of the IPython notebook.

### Data Augmentation

Data labeling is a key point for a supervised learning algorithm to be able to predict the correct outcome. However, labeling data for all possible scenarios can be expensive and time consuming. Data augmentation helps to enrich the labeled data set by "augmenting" the data with some operations that mimics different conditions. For instance, image can be augmented with different contrasts, brightness, lighting, etc.

For this project, we have created a new pipeline using the Tensor Flow library for image manipulation to augment the data set. Tensor Flow offers some functions that randomly change the image, reducing the chance of the same image to be processed twice by the Neural Network, which therefore reduces the chance of over fitting the neural network. This new pipeline is placed before the neural network and executed online with the optimizer for every batch.

Figure 6 shows the Data Augmentation Pipeline. This new pipeline changes the image contrast and brightness in random fashion (see Tenfor Flow documentation for tf.image.random_contrast and tf.image.random_brightness). The main objective is to train the network to deal with different contrast and brightness situations. In addition, the augment pipeline also uses the extract_glimpse that randomly "moves", or translate, the image around in order to improve the Neural Network robustness for Translation Invariance. Figure 7 shows an example of an augmented image. 

![alt text][image7]

![alt text][image13]

The code for this part is contained in the code cell of the second setp of the IPython notebook.

### Neural Network

The neural network implemented for this work is a custom extended version of LeNet network enhanced with two extra Convolution Layers. These extra layers helps the network to learn more complex object such as traffic signs. We have used the LeNet as a starting point and customized it to fit the Traffic Signs requirements. This process was done empirically by increasing the number of layers and testing to check if accuracy was improved. The classical LeNet achieved around 85% accuracy on validation test and every new layer increased the accuracy in roughly 5%, resulting in a final accuracy of 95%. Finally, there is an extra fully connected layer to adjust the number of neurons because of the increased number of shared weights (due to the extra convolution layers). Figure 8 presents the network architecture.


| Layer        		 	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten		| 1600 neurons        									|
| Dropout	    | Training probality of 50%      									|
| Fully connected		| output 400        									|
| Dropout	    | Training probality of 50%      									|
| Fully connected		| output 120        									|
| Fully connected		| output 82       									|
| Fully connected		| output 42       									|
| Softmax				|         									|

The code for the Network is located in the seventh cell of the ipython notebook.

### Training

The training algorithm employs the Adam optimizer algorithm with a custom adjustable learning rate. The training algorithm adapts the learning rate according with the validation set accuracy. If accuracy drops, learning rate decreases, if accuracy remains the same, learning rate increases. This adjustable learning rate helps to improve model convergence. In addition, the training algorithm keeps track of progress by saving the model every time it achieves a new threshold. Moreover, if the training accuracy diverges too much, the training algorithm restore the previous best accuracy state.

The rest of the algorithm follows a common approach for classification problem which is to minimize the cross entropy between the logits (softmax) and the labels hot encoding. The values for the number of EPOCHS (20), batch size (256) and initial learning rate (0.001) were defined empirically. 


The code for training the model is located in the eigth cell of the ipython notebook. 

### Results

The approach taken to develop this project was an iterative process with empirical validation. The first step was to test the classical LeNet network, which achieved an accuracy of 86% on the validation test. After that, we extended the network with two additional convolution layers because we realized that our network was not capable to take into account more complex objects such as Traffic signs. With the addition of these extra layers, the accuracy for the training set reached 98% but the validation set did not reach 92%. Finally, we extended the network with two additional dropouts after the fully connected layers 1 and 2, which helped the network to reach 95% accuracy for the validation set in less than 10 EPOCHS.

Even though this network has achieved 95% accuracy for the validation set, it did not perform well with the images downloaded from the Internet. We observed that our network was not robust enough to achieve translation invariance; our network classifies correctly only when the downloaded image are cropped at the center. This observation let us to extend our augmentation pipeline (see previous section about data augmentation) to also translate the images. As we mentioned previously, we use Tensor Flows functions to augment the data set in a random fashion and online with the training algorithm, i.e. the image is augmented only for the training and is not saved. We have set our augment pipeline to translate the images up to 10 pixel up/down/right/left.

Because of the internet images and the translation augmentation, our final training algorithm takes two rounds to train the model. It first trains the network over the original data set for 10 EPOCHS. Then, it trains another 10 EPOCHS using the augment data pipeline to change the images at random, for each training. During this new training round, the accuracy for the validation set decays because of the image translation from data augmentation. We have observed that using a translation of up to 10 pixel worsens the validation accuracy but improves the prediction of the Internet images. Values below 10 pixels yields better accuracy for the original validation set; the value 10 for the translation was the limit we found to reach the minimum accuracy for the validation set of 93% and also able to correctly predict the images from the web. 

Final Results:
* training set accuracy of   ~97%
* validation set accuracy of ~93% 
* test set accuracy of       ~92%

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

###Test a Model on New Images

Here are five German traffic signs downloaded from the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]


First, it is important to observe the relation between these images and the number of examples at the training data set. From the training data set, image 1 has 2010 examples, image 2 has 1260 examples, image 3 480 examples, image 4 1860 examples and image 5 690 examples. So, because of the unbalance character among these classes, images 3 and 5 are more difficult to classify.

The second observation is that the image format (size) and the place in the image where the traffic sigh is positioned are quite different from the training data set. The size difference can be easily circumvented by pre-processing the image. However, the position problem is not trivial to solve because the sign can be positioned at any place in the image. At first, we tried to hand code a crop pre-processing function but we believe that this is not right way to solve this problem because the sigh can be at any place. Finally, as we mentioned previously, we decide to augment our data set with a translation operation to move the signs around inside the images to overcome this problem.

In order to understand better the predictions from our model, we decided to test the classifier over three versions for every image from the web, they are: the original image, an automatic cropped version and a hand cropped version. Here are the results for the prediction for all of them:

| Image			        |	Original	|	Aut. Croped	| 	Hand Croped	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		|   									|
| 60 km/h      		| Stop sign   									|
| Children Crossing      		| Stop sign   									| 
| Keep right     		| Stop sign   									| 
| Stop Sign      		| Stop sign   									| 

The code for making these predictions is located in the tenth cell of the Ipython notebook.

For the original images, the classifier was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. For the automatically and hand croped, the classifier giver an accuracy of % and %, respectively. As we can see, even with augment data, our neural network is not translation invariant because the cropped images from the web yielded a better accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here, we present the top 5 probilities for the original images only. For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 50 km/h   									| 
| .20     				| 60 km/h 										|
| .05					| Children Crossing										|
| .04	      			|  Keep right			 				|
| .01				    |  Stop Sign							|


The code for making predictions is located in the 11th cell of the Ipython notebook.

# FINAL PROJECT OUTLINE

## Team 9

### Neural Networks to identify Dog Breeds :sparkles:

*Submitted by: Mohammad Anas, Ying Feng, Vicky Nomwesigwa, Deekshita Saikia*

### Problem Statement

People have been breeding dogs since prehistoric times. The earliest dog breeders used wolves to create domestic dogs. These dogs would be bred for specific tasks, like hunting, guarding, and herding. Nowadays, we see dogs in all shapes and sizes, and frequently without pedigrees to describe their breed or heritage. The identities of dogs with unknown or mixed-breed lineages are frequently guessed based on their physical appearance, but it is not known how accurate these visual breed assessments are.

Identifying a particular breed can therefore be a rigorous task. Through this project, we aim to build a neural network for classifying dog breeds based on over 20,000 input images of various physical traits.

### Motivation

Majority of dogs are often difficult to classify by simply looking, and breed identification is important when rescuing dogs, finding them forever homes, treating them, and various other furry situations. Breed prediction also helps veterinarians treat breed specific ailments for stray, unidentified dogs that need medical care. The hope is that classifying the breed a dog belongs to will help give more information about the dog’s personality, full-grown size, and health. Moreover, this algorithm can help us understand the behavior of different dog species and facilitate selection of certain breeds to work closely with humans for specific tasks. For example, German shepherds are renowned for their courage, confidence, loyalty and ability to learn commands. Hence, they are used for security purposes.

### Data

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotations from ImageNet for the task of fine-grained image categorization.The ImageNet project is a large visual database of about 14 million hand-annotated images widely used in visual object recognition software research. 

The dataset contains the following:
* Number of categories/breeds: 120
* Number of images: 20,580
* Annotations: Class labels, Bounding boxes

There are 150+ images available corresponding to each class label, which gives us enough data to train on. The original data source can be found [here](http://vision.stanford.edu/aditya86/ImageNetDogs/). 

### Methods

Primarily, we will be relying on convolutional neural networks (CNN) for the purpose of classifying dog breeds. CNNs are a class of deep learning neural networks which are a common choice for image classification. CNNs are fast and efficient as the number of trainable parameters do not increase exponentially. A CNN convolves learned features with input data and uses 2D convolutional layers. This means that this type of network is ideal for processing 2D images. Compared to other image classification algorithms, CNN does not have underlying assumptions of the underlying data and requires very little preprocessing. We would customize the CNN model with multiple building blocks such as convolution layers, pooling layers and connected layers and treating dogs’ images as input layers. The CNN model will adaptively learn spatial hierarchies of features through a backpropagation method and classify dogs into breeds.

### Experiments 

Once we have implemented our baseline CNN that is trained on the images in the dataset, we plan to further improve the performance of our model by regularization techniques like adding dropout layers. Moreover, we plan to apply image augmentation techniques to improve our model’s generalization performance. One image augmentation technique that we plan to focus on in particular is cropping. By looking at our dataset, we note that some images focus more on the surroundings rather than the dog itself. By cropping our images we believe that we can reduce noise in our dataset and improve our performance. Other image augmentation techniques like rotation might also be performed before building the model. The model built on these augmented images will then be compared to our baseline model to evaluate performance. 

For the purpose of model evaluation, we will be using the test dataset to measure generalization performance that is available as a separate dataset. Fortunately, we don’t have a class imbalance in our dataset and we can rely on metrics like F1 score to evaluate the performance of our model and supplemented with metrics such as Precision and Recall to further tune the model performance.

### Project Plan

A rough timeline of the project is as under:
* Data Wrangling and Pre-processing – 2 weeks
* Modeling – 1 week
* Assimilating results – 2 weeks
* Documentation – 1 week

### Roles 

* Writer () - Primarily responsible for writing the final report, and all other aspects of documentation.
* Programmer () - Primarily responsible for writing and maintaining reproducible codes
* Presenter () - Primarily responsible for preparing slides and presenting to a general audience
* Coordinator () - Primarily responsible for coordinating meetings and keep deliverables on schedule

### References 

Khosla, A., Jayadevaprakash, N., Yao, B., & Fei-Fei, L. (2011). Stanford Dogs Dataset. Stanford Dogs Dataset. http://vision.stanford.edu/aditya86/ImageNetDogs/
ImageNet. (2021). ImageNet. https://www.image-net.org/update-mar-11-2021.php
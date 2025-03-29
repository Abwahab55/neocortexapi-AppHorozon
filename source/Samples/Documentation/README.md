# Project Title: ML 24/25-01 Investigate Image Reconstruction by using Classifiers
[![Made with - C#](https://img.shields.io/badge/Made_with-C%23-2ea44f?style=for-the-badge&logo=C%23)](https://learn.microsoft.com/en-us/dotnet/csharp/)
![Built With - ❤️](https://img.shields.io/badge/Built_With-❤️-2ea44f?style=for-the-badge&logo=Love)

### An experiment to demonstrate how the reconstruction method recreates the images by using Classifiers in C#.


* [Problem Statement](#Problem-Statement)
* [Introduction](#Introduction)
* [The Overview of the project](#The-Overview-of-the-project)
* [Spatial Pooler](#Spatial-Pooler)
* [Phases of SP](#Phases-of-SP)
* [Sparse Distributed Representation (SDR)](#Sparse-Distributed-Representation-(SDR))
* [HTM Classifier](#HTM-Classifier)
* [KNN Classifier](#KNN-Classifier)
* [Methodology](#Methodology)
  
# Problem Statement: 

This process provides a simple but effective way to add missing data. These methods, when used together, may improve reconstructing pictures from sparse representations while still maintaining computational feasibility. In this work, we investigate whether it is feasible to reconstruct pictures from SDRs using HTM and KNN classifiers. This work intends to improve HTM's spatial learning capacity by using the IClassifier interface, thereby exploiting KNN's power in similarity-based categorization. By means of this method, we want to create a classification-based reconstruction framework able to undo the SDR encoding process and restore the original input pictures. 

 
# Introduction:
 
In this project, an experiment which is performed to makes us indulge with the knowledge of Sparse Distributed Representations (SDRs),classifiers and an integral component in the neocortexapi.Reconstructing or approximating an image from incomplete, noisy, or partial data is the basic challenge in image reconstruction in computer vision, machine learning, and artificial intelligence.The goal of this work is to create a classification-based way to rebuild pictures from SDR-encoded data that takes advantage of the best parts of HTM and KNN. This will create a new framework that strikes a good balance between how quickly the reconstruction is done and how well it looks. Image reconstruction has use in many important areas.Image reconstruction improves object identification, feature classification, and pattern recognition under demanding situations.This work offers a computationally efficient and interpreted method applicable to many fields by solving the difficulties related to conventional deep learning models. The results of this study could provide fresh avenues for creating strong, scalable image reconstruction methods catered to practical applications.


# The Overview of the project

![Copy of Conversation tree example (2)](https://github.com/user-attachments/assets/b7d66182-0b3b-4289-a612-bb1f1ec07c9f)
Figure 1 Image Reconstruction Using Classifiers





# Spatial Pooler
In the HTM framework, the Spatial Pooler (SP) is a component responsible for creating sparse distributed representations (SDR) of input data. The primary goal of the SP is to transform input patterns into a stable and sparse representation that subsequent stages of the neural network can easily use.
In the HTM framework, the Spatial Pooler (SP) is a component responsible for creating sparse distributed representations (SDR) of input data. The primary goal of the SP is to transform input patterns into stable and sparse representations that subsequent stages of the neural network can easily use.
In the HTM framework, the Spatial Pooler (SP) is a component responsible for creating sparse distributed representations (SDR) of input data. The primary goal of the SP is to transform input patterns into stable and sparse representations that subsequent stages of the neural network can easily use.


# Phase of SP 
The SP has three phases: overlap, inhibition, and learning.  Numerous columns exist inside a stored procedure.  Every column has a distinct arrangement of proximal synapses linked by a proximal dendritic segment.  Each proximal synapse provisionally links to a singular column from the input, so that each column in the SP corresponds to a distinct property inside the input.  The activity level of the input column serves as the synaptic input, where an active column is represented as "1" and an inactive column as "0".  The persistence value of the synapse is assessed to ascertain its connectivity.  If the persistence value is at least equal to the linked threshold, the synapse is considered connected; otherwise, it is deemed disconnected.  The persistence values are scalars inside the closed interval [0,1] 
(Source:https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00111/full )


# Sparse Distributed Representation (SDR)
Recent neuroscience research indicates that the brain utilizes Sparse Distributed Representations for information processing. This applies universally to all animals, ranging from mice to humans.  These SDRs are essential for enhancing comprehension of the brain's computational methodology.  SDRs represent the information the brain processes at a particular instant, with each active cell embodying a semantic facet of the overarching message.  Sparse denotes that only a limited number of the many (thousands of) neurons are concurrently active, as opposed to the conventional "dense" representation in computers, characterized by a few bits of 0s and 1s.  Distributed signifies that both the active cells and the importance of the pattern are dispersed across the depiction.  It makes the SDR robust against the failure of individual neurons and facilitates sub-sampling.  Each bit or neuron has a specific meaning; thus, if an identical bit is activated in two Sparse Distributed Representations (SDRs), it indicates semantic similarity.(Source: https://www.cortical.io/science/sparse-distributed-representations/?highlight=SDR)


# HTM Classifier
HTM Classifier is focused around the concept of Temporal Memory, The HTM classifier utilizes a spatial pooler for training to capture a set of images. The HTM classifier’s training was done with a spatial pooler to grab the images’ spatial features. It is then followed by a set training cycle of 20 repetitions in order to ensure that the model is able to adapt and comprehend meaningful representations of the images that were fed into it. To feed the model data, HTM uses columns and cells to form representations of the input data (SDR), which is referred to as a SDR.


# KNN Classifier
The KNN classifier is one of the simplest classifiers to use, but also one of the most effective at the same time in regards to classifying data, which is based off computational similarity to the closest training examples for that particular feature. In this instance, the training examples are the binarized images. The classifier “guesses” the class label based on the input SDRs and trains SDRs by calculating the similarity of the input SDRs and the training SDRs. 


# Methodology:
The implementation of this project is focused on incorporating machine learning methods such as Hierarchical Temporal Memory (HTM) and K-Nearest Neighbors (KNN) for image classification and reconstruction. It can be performed using pre-defined algorithms.
**Image Binarization**:
The first step towards achieving the goal of the project is to binarize the images located in the training folder.the training folder is located here:https://github.com/Abwahab55/neocortexapi-AppHorozon/tree/AppHorozon/source/Samples/NeoCortexApiSample/Sample  and Imagebinarizer from Dianet was used to binarize the images. Each image is adjusted to a standard dimension of 64x64 pixels. This maintains a specific degree of consistency among the images. In the binarization step, each pixel of an image is converted to either white or black. More technically, the pixel gets assigned 0 if it’s white and 1 if it’s black. The after effect is that a Sparse Distributed Representation (SDR) is achieved that can undergo further operations.
To complete the binarization, the ImageBinarizerSpatialPattern class was created. This class takes an image, transforms it to a monochrome version, and subsequently performs a binarization by assigning white and black values for pixels.

**SDR Generation**:
after binarization it goes through sp for a better organized sdr values, then This transformed image is stored as an SDR.
Both of the HTM (Hierarchical Temporal Memory) and KNN (K-Nearest Neighbors) classifiers are trained using the SDR data. These classifies are for measurement of the likeness of the original image to the reconstructed image. The steps in the training procedure of every classifier is given below:
# Classifier Training:
**HTM**:
HTM Classifier is focused around the concept of Temporal Memory, The HTM classifier utilizes a spatial pooler for training to capture a set of images. The HTM classifier’s training was done with a spatial pooler to grab the images’ spatial features. It is then followed by a set training cycle of 20 repetitions in order to ensure that the model is able to adapt and comprehend meaningful representations of the images that were fed into it. To feed the model data, HTM uses columns and cells to form representations of the input data (SDR), which is referred to as a SDR.

**KNN**:
The KNN classifier is one of the simplest classifiers to use, but also one of the most effective at the same time in regards to classifying data, which is based off computational similarity to the closest training examples for that particular feature. In this instance, the training examples are the binarized images. The classifier “guesses” the class label based on the input SDRs and trains SDRs by calculating the similarity of the input SDRs and the training SDRs.

**Reconstruction**:
Now that all classifiers have had their training done, the image reconstruction task is performed as with both the HTM and KNN classifiers. The process of reconstruction requires taking an input SDR, providing it to the classifiers, then recreating the input image guided by the learned representations. During this process, the SDR output by the classifier is checked against the original SDR to compare for likeness as a measure of which the classifier’s capabilities.
HTM utilizes its memory to generate a prediction of the SDR, based on the learned representations. After that, similarity measures help one to compare the rebuilt picture with the original. KNN contrasts training set stored SDRs with input SDR.  Based on these closest matches, it reconstructs the picture and notes the most comparable SDRs. 

**Similarity Evaluation**: 
The efficiency of both classifiers in image reconstruction is assessed by calculating the similarity between the original SDR and the rebuilt SDR using the following metrics: Jaccard Similarity is basically a similarity metric based on sets that computes the ratio of the intersection to the union of two sets.  It assesses the similarity between two SDRs by analyzing their non-zero components. The second similarity is Cosine Similarity metric which calculates the cosine of the angle between two vectors, and represent the SDRs in a high-dimensional space.  It is used to assess the similarity of the two SDRs about their orientation. Hamming Distance: A bitwise comparison metric that calculates the number of differing bits between two SDRs. It is used to measure the binary similarity between SDRs.

**Visualization**: 
For the purpose of visualizing the results of the similarity computations, a bar graph is used. We display the similarity scores of each picture, both HTM and KNN, to facilitate a clear comparison between the two classifiers. We construct the graph and then store it in the selected folder. Various colors are used to depict the HTM and KNN results in the bar graph, and labels are used to indicate the percentages of similarity between the two sets of results.
The last step of our project include a comparison of the similarity scores of the HTM and KNN classifiers. The assessment underlines the following elements: HTM typically captures the structural representations and temporal patterns of images, resulting in better similarity ratings. When the training set consists of similar images, KNN performs well; yet, it is less successful in generalizing acro/ss many patterns. This results in, often, lower similarity scores than HTM. The capacity of the HTM and KNN classifiers to correctly reconstruct input images was assessed. We used a comparison of original and reconstructed images for each classifier to produce graphs that displayed the respective classifier performance. 

for the output files the folder structure is as follows
neocortexapi\source\Samples\NeoCortexApiSample\bin\Debug\net8.0\
## Example Output
**Original vs Reconstructed Image**
**Original**: https://github.com/Abwahab55/neocortexapi-AppHorozon/blob/AppHorozon/source/MySEProject/Documentation/result/TestFiles.png

**Reconstructed using HTM**: https://github.com/Abwahab55/neocortexapi-AppHorozon/tree/AppHorozon/source/Samples/Documentation/result/TestFiles_HTM_Reconstructed.png

**Reconstructed using KNN**: https://github.com/Abwahab55/neocortexapi-AppHorozon/tree/AppHorozon/source/Samples/Documentation/result/TestFiles_KNN_Reconstructed.png

**Similarity comparison**: https://github.com/Abwahab55/neocortexapi-AppHorozon/tree/AppHorozon/source/Samples/Documentation/result/SimilarityComparison_Improved.png

## Running the Project

**Dependencies:**
- .NET 8
- NeoCortexApi
- Daenet ImageBinarizer
- MSTest (for Unit Testing)

**Run Instructions:**
1. Build the project in Visual Studio.
2. Place training images in the `Sample` folder.
3. Run the solution — outputs are generated in the `bin/Debug/net8.0` folder.

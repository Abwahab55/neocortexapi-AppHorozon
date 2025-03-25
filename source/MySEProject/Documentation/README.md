# Project Title: ML 24/25-01 Investigate Image Reconstruction by using Classifiers
[![Made with - C#](https://img.shields.io/badge/Made_with-C%23-2ea44f?style=for-the-badge&logo=C%23)](https://learn.microsoft.com/en-us/dotnet/csharp/)
![Built With - ❤️](https://img.shields.io/badge/Built_With-❤️-2ea44f?style=for-the-badge&logo=Love)

### An experiment to demonstrate how the reconstruction method recreates the images by using Classifiers in C#.


* [Overview](#Overview)
* [Problem Statement](#Problem-Statement)
* [Introduction](#Introduction)

* [Methodology](#Methodology)
# METHODOLGY
The implementation of this project is focused on incorporating machine learning methods such as Hierarchical Temporal Memory (HTM) and K-Nearest Neighbors (KNN) for image classification and reconstruction. It can be performed using pre-defined algorithms.
The first step towards achieving the goal of the project is to binarize the images located in the training folder. Each image is adjusted to a standard dimension of 28x28 pixels. This maintains a specific degree of consistency among the images. In the binarization step, each pixel of an image is converted to either white or black. More technically, the pixel gets assigned 0 if it’s white and 1 if it’s black. The after effect is that a Sparse Distributed Representation (SDR) is achieved that can undergo further operations.
To complete the binarization, the ImageBinarizerSpatialPattern class was created. This class takes an image, transforms it to a monochrome version, and subsequently performs a binarization by assigning white and black values for pixels. This transformed image is stored as an SDR.
Both of the HTM (Hierarchical Temporal Memory) and KNN (K-Nearest Neighbors) classifiers are trained using the SDR data. These classifies are for measurement of the likeness of the original image to the reconstructed image. The steps in the training procedure of every classifier is given below:


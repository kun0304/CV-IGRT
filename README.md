# Human-Level Comparable Control Volumes Mapping with An Unsupervised-Learning Model for CT-Guided Radiotherapy

Pytorch implementation for Control Volumes Mapping with An Unsupervised-Learning Model
The code was written by Xiaokun Liang.

## Applications
Patient Positioning in CT-Guided Radiation Therapy

## Prerequisites
Windows 10, python 3.7, NVIDIA GPU ( >= 6GB)

## Dependencies
The project depends on following libraries:
PyTorch 1.3.0
NumPy 1.17.1
SimpleITK 1.2.2

## Prepare data
Create a cv_dataset file with two sub-files (training and testing), and put your own data into these two files. The data folder is organized in the following way:

![](.\structure\tree.jpg)

The CSV file is the center coordinates of the selected control volumes. For example:
Skull	   178	104	 150
C2		   124	108	 84
C6		   156	112	 24
Mandible 24	  104	 96

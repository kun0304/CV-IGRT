# Human-Level Comparable Control Volumes Mapping with An Unsupervised-Learning Model for CT-Guided Radiotherapy

Pytorch implementation for Control Volumes Mapping with An Unsupervised-Learning Model.
The code was written by Xiaokun Liang.

## Applications
Patient Positioning in CT-Guided Radiation Therapy

## Prerequisites
Windows 10, python 3.7, NVIDIA GPU ( >= 12GB)

## Dependencies
The project depends on following libraries:  
PyTorch 1.3.0  
NumPy 1.17.1  
SimpleITK 1.2.2

## Prepare data
The example of the dataset is put in the cv_dataset file. You can put your own data into these two files by following the same format. 

## Usage
1. Setting the hyper-parameter of the network in the param.py file.

2. Train the model: 
```
python cv_training.py
```

3. Test the model:
```
python cv_test.py
```

## Questions
Please contact 'xk.liang@qq.com'

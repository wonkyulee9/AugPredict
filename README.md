# AugPredict
Object Detection with self-supervised Augmentation Parameter Prediction

This repo is the code used for the Master's thesis "Object Detection with task-specific self-supervised learning".

The code reproduces object detection training on different models pre-trained with self-supervised learning, with and without "Augmentation Parameter Prediction" proposed in the paper. Object detection tests are done on a small dataset (SVHN).

Parts of the code for contrastive learning is adapted from https://github.com/HobbitLong/SupContrast and parts of the code for object detection using Faster-RCNN is adapted from https://github.com/pytorch/vision/tree/master/references/detection.

# Pre-Training
main_supcon.py is used to pre-train a model with SimCLR/SupCon
main_ce.py is used for supervised cross entropy training
main_augpred.py is used for SimCLR + Augmentation Parameter Prediction

# Downstream
main_ce.py is used to evaluate classification performance
main_detection.py is used to train and evaluate object detection performance

Note: for detection dataloader, download SVHN full numbers dataset and place into ./datasets/SVHN_full then run svhn_detect_data.py

import numpy as np
import math
import torch
from torchvision import transforms
import time
import shutil
import sys, os



mean=np.array([0.4914, 0.4822, 0.4465])
std=np.array([0.2023, 0.1994, 0.2010])

max_val = np.array([ (1. - mean[0]) / std[0],
                     (1. - mean[1]) / std[1],
                     (1. - mean[2]) / std[2],
                    ])

min_val = np.array([ (0. - mean[0]) / std[0],
                     (0. - mean[1]) / std[1],
                     (0. - mean[2]) / std[2],
                   ])
                    

eps_size=np.array([  abs( (1. - mean[0]) / std[0] ) + abs( (0. - mean[0]) / std[0] ),
                     abs( (1. - mean[1]) / std[1] ) + abs( (0. - mean[1]) / std[1] ),
                     abs( (1. - mean[2]) / std[2] ) + abs( (0. - mean[2]) / std[2] ),
                  ])



def train_scale():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def train_zero_norm():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def test_scale():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def test_zero_norm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def unnormalize():
    return transforms.Normalize( (- mean / std).tolist(), (1.0 / std ).tolist() )

def inverse_normalize():
    u = [-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010]
    sigma = [1./0.2023, 1./0.1994, 1./0.2010]
    return transforms.Normalize( u, sigma )


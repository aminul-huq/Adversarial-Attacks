import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attacks import *
from tqdm import tqdm


def generateFGSM(test_loader,model,device,eps,criterion,t):
    adv_img = []
    l = []
    eps = 0.1
    
    print("Generating FGSM Adversarial Images")
    iterator = tqdm(test_loader,ncols=0, leave=False)
    for data,labels in iterator:
        x1 = FGSM(model,data,labels,device,eps,criterion)
        l.append(labels)
        x2 = x1.squeeze().cpu().detach().numpy()
        adv_img.append(x2)
            
    x3 = np.array(adv_img)
    
    if t == 0:
        x3 = x3.reshape(50000,3,32,32)
    else:
        x3 = x3.reshape(10000,3,32,32)
        
    l = np.array(l)
        
    features_test = torch.from_numpy(x3)
    targets_test = torch.from_numpy(l)

    new_dataset = torch.utils.data.TensorDataset(features_test,targets_test)
    new_data_loader = torch.utils.data.DataLoader(new_dataset,100,shuffle = False)
    
    
    return new_data_loader



def generateMIFGSM(test_loader,model,device,eps,momentum,max_iter,criterion):
    adv_img = []
    l = []
    eps = 0.1
    
    print("Generating MIFGSM Adversarial Images")
    iterator = tqdm(test_loader,ncols=0, leave=False)
    for data,labels in iterator:
        x1 = MIFGSM(model,data,labels,device,eps,momentum,max_iter,criterion)
        l.append(labels)
        x2 = x1.squeeze().cpu().detach().numpy()
        adv_img.append(x2)
            
    x3 = np.array(adv_img)
    x3 = x3.reshape(10000,3,32,32)
    l = np.array(l)
        
    features_test = torch.from_numpy(x3)
    targets_test = torch.from_numpy(l)

    new_dataset = torch.utils.data.TensorDataset(features_test,targets_test)
    new_data_loader = torch.utils.data.DataLoader(new_dataset,100,shuffle = False)
    
    
    return new_data_loader


def generateDeepFool(test_loader,model,device,num_classes=10, overshoot=0.02, max_iter=10):
    adv_img = []
    l = []
     
    print("Generating DeepFool Adversarial Images")
    iterator = tqdm(test_loader,ncols=0, leave=False)
    for data,labels in iterator:
        _,_,_,_,x1 = deepfool(model,data,device,num_classes=10, overshoot=0.02, max_iter=10)
        l.append(labels)
        x2 = x1.squeeze().cpu().detach().numpy()
        adv_img.append(x2)
            
    x3 = np.array(adv_img)
    x3 = x3.reshape(10000,3,32,32)
    l = np.array(l)
        
    features_test = torch.from_numpy(x3)
    targets_test = torch.from_numpy(l)

    new_dataset = torch.utils.data.TensorDataset(features_test,targets_test)
    new_data_loader = torch.utils.data.DataLoader(new_dataset,100,shuffle = False)
    
    
    return new_data_loader




def generatePGD(test_loader,model,device,eps,attack_iter,criterion,t):
    adv_img = []
    l = []
    eps = 0.1
    attack_iter = 20
    attack_lr = 0.01
    
    print("Generating PGD Adversarial Images")
    iterator = tqdm(test_loader,ncols=0, leave=False)
    for data,labels in iterator:
        x1 = PGD(model,data,labels,eps,attack_iter,attack_lr,criterion,device)
        l.append(labels)
        x2 = x1.squeeze().cpu().detach().numpy()
        adv_img.append(x2)
            
    x3 = np.array(adv_img)
    if t == 0:
        x3 = x3.reshape(50000,3,32,32)
    else:
        x3 = x3.reshape(10000,3,32,32)
        
    l = np.array(l)
        
    features_test = torch.from_numpy(x3)
    targets_test = torch.from_numpy(l)
    
    new_dataset = torch.utils.data.TensorDataset(features_test,targets_test)
    new_data_loader = torch.utils.data.DataLoader(new_dataset,100,shuffle = False)
    
    
    return new_data_loader    

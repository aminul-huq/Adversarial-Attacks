import argparse,os,torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset
from preprocess import *
from attacks import *
from models import *
from utils import *
from generate_attacks import *

device = "cuda"
torch.cuda.set_device(1)
dir = os.getcwd()
file_name = "/Models/model.pth.tar"
path = dir + file_name    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Adversarial Attacks')

    # model hyper-parameter variables
    parser.add_argument('--pretrain', default=0,metavar = 'pretrain',type=int,help='0 for new train 1 for pretrained')
    parser.add_argument('--attack', default=0,metavar = 'attack',type=int,help='0 for FGSM 1 for MIFGSM 2 for PGD 3 for DF 4 for all')
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=5, metavar='itr', type=int, help='Number of iterations')
    parser.add_argument('--momentum', default=0.9, metavar='momentum', type=float, help='Momentum Value')
    parser.add_argument('--eps', default=(8./255.), metavar='eps', type=float, help='Epsilon')
    parser.add_argument('--step_size', default=7, metavar='step_size', type=int, help='Attack steps') ## attack iteration
    parser.add_argument('--overshoot', default=0.2, metavar='overshoot', type=float, help='Overshoot value for DeepFool')
    parser.add_argument('--preprocess', default=0, metavar='preprocess', type=int, help='0 for standard scaling\
                        , 1 for zero_norm')
    
    

    args = parser.parse_args()
    
    ## NEED to FIX the validation issue
    if args.preprocess == 0:
        print("SCALING IMAGES [0-1]")
        train_set = datasets.CIFAR10(root = '/home/aminul/data',train = True, 
                           transform = train_scale(), download = False)
        test_set = datasets.CIFAR10(root = '/home/aminul/data',train = False, 
                          transform = test_scale(), download = False)
    else:
        print("NORMALIZING IMAGES WITH ZERO-MEAN")
        train_set = datasets.CIFAR10(root = '/home/aminul/data',train = True, 
                           transform = train_zero_norm(), download = False)
        test_set = datasets.CIFAR10(root = '/home/aminul/data',train = False, 
                          transform = test_zero_norm(), download = False)
        

    train_loader = DataLoader(train_set,100,True)
    test_loader = DataLoader(test_set,100,False)

    train_loader_1 = DataLoader(train_set,1,True)
    test_loader_1 = DataLoader(test_set,1,False)
    
    
    num_classes = 10
    learning_rate = args.lr
    num_epochs = args.itr
    eps = args.eps
    mode = args.attack
    max_itr = args.step_size
    overshoot = args.overshoot
    momentum = args.momentum
    pretrain = args.pretrain
    criterion = nn.CrossEntropyLoss()
    
    model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    opt_1 = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    
    
    if pretrain == 0:
        history = train(model,train_loader,num_epochs,opt_1,criterion,device)
        torch.save(model.state_dict(), path)
    else:
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
        model.load_state_dict(torch.load(path))
        model.eval()
    
    
    
    if mode == 0:
        
        adv_loader = generateFGSM(test_loader_1,model,device,eps,criterion,t=1)
        print("Testing Against FGSM")
        test(model,adv_loader,criterion,device)
        
    elif mode == 1:
        
        adv_loader = generateMIFGSM(test_loader_1,model,device,eps,momentum,max_itr,criterion)
        print("Testing Against MIFGSM")
        test(model,adv_loader,criterion,device)
        
    elif mode == 2:
        
        adv_loader = generatePGD(test_loader_1,model,device,eps,max_itr,criterion,t=1)
        print("Testing Against PGD")
        test(model,adv_loader,criterion,device)
    
        
    elif mode == 3:
        
        adv_loader = generateDeepFool(test_loader_1,model,device,num_classes=10, overshoot=0.02, max_iter=10)
        print("Testing Against DeepFool")
        test(model,adv_loader,criterion,device)
        
    elif mode == 4:
        
        adv_loader = generateFGSM(test_loader_1,model,device,eps,criterion,t=1)
        torch.save(train_loader, dir + '/Saved_Images/FGSM_dataloader.pth')
        print("Testing Against FGSM")
        test(model,adv_loader,criterion,device)
        
        adv_loader = generateMIFGSM(test_loader_1,model,device,eps,momentum,max_itr,criterion)
        torch.save(train_loader, dir + '/Saved_Images/MIFGSM_dataloader.pth')
        print("Testing Against MIFGSM")
        test(model,adv_loader,criterion,device)
        
        adv_loader = generatePGD(test_loader_1,model,device,eps,max_itr,criterion,t=1)
        torch.save(train_loader, dir + '/Saved_Images/PGD20_dataloader.pth')
        print("Testing Against PGD")
        test(model,adv_loader,criterion,device)
        
        adv_loader = generatePGD(test_loader_1,model,device,eps,100,criterion,t=1)
        torch.save(train_loader, dir + '/Saved_Images/PGD100_dataloader.pth')
        print("Testing Against PGD")
        test(model,adv_loader,criterion,device)
        
        
        
    print("Testing Against Original Data")    
    test(model,test_loader,criterion,device)
    
    
    
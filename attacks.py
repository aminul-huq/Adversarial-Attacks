import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

    
def FGSM(model,img,labels,device,eps,criterion):
    
    model,img,labels = model.to(device),img.to(device),labels.to(device)
    
    #img.requires_grad =True
    #pred = model(img)
    #loss = criterion(pred,labels)
    #loss.backward()
    #img_grad = img.grad.data
    #sign = img_grad.sign()
    #perturb_imgs = img + eps * sign
    #perturb_imgs = torch.clamp(perturb_imgs,0.,1.)
    
    
    x = img.clone().detach().requires_grad_(True).to(device)
    alpha = eps 

    pred = model(x)
    loss = criterion(pred,labels)
    loss.backward()
    noise = x.grad.data

    x.data = x.data + alpha * torch.sign(noise)
    x.data.clamp_(min=0.0, max=1.0)
    x.grad.zero_()
    
    return x



def MIFGSM(model,img,labels,device,eps,momentum,max_iter,criterion):
    
    model,img,labels = model.to(device),img.to(device),labels.to(device)
    x = img.clone().detach().requires_grad_(True).to(device)
    alpha = eps / 4
    g = torch.zeros(img.size(0), 1, 1, 1).to(device)
    
    for _ in range(max_iter):
        pred = model(x)
        loss = criterion(pred,labels)
        loss.backward()

        noise = x.grad.data
        g = momentum * g.data + noise/ torch.mean(torch.abs(noise),dim=(1,2,3),keepdim=True)
        noise = g.clone().detach()
        
        x.data = x.data + alpha * torch.sign(noise)
        x.data.clamp_(min=0.0, max=1.0)
        #x.grad.zero_()
    
    return x



def deepfool(net, image, device, num_classes=10, overshoot=0.02, max_iter=10):
    
    net,image = net.to(device),image.to(device)
    
    f_image = net(image).cpu().data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    #x = torch.tensor(pert_image[None, :],requires_grad=True).to(device)
    x = pert_image[None, :].clone().detach().requires_grad_(True).to(device)
    
    fs = net.forward(x[0])
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            
            #x.zero_grad()
            
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

        #x = torch.tensor(pert_image, requires_grad=True).to(device)
        x = pert_image.clone().detach().requires_grad_(True).to(device)
        
        fs = net.forward(x[0])
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image




def PGD(model,image,labels,eps,attack_steps,attack_lr,criterion,device,random_init = True, target = None, clamp = (0,1)):
    
    model,image,labels = model.to(device),image.to(device),labels.to(device)    
        
    x_adv = x.clone()
    
    if random_init:
        x_adv = x_adv + (torch.rand(image.size(),dtype = image.dtype, device = device) - 0.5) *2 * eps
        
    for i in range(attack_steps):
        
        x_adv.requires_grad = True
        
        model.zero_grad()
        logits = model(x_adv)
        
        if target is None:
            loss = criterion(logits,labels)
            loss.backward()
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv + attack_lr * grad
        
        else:
            assert target.size() == y.size()
            loss = F.cross_entropy(logits, target)
            loss.backward()
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv - attack_lr * grad
       
        x_adv = x + torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, *clamp)
        
        
    return x_adv

    


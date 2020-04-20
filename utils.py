import os,time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
    

def train(model,train_loader,num_epochs,opt,criterion,device):
    train_loss_list = []
    iteration_list = []
    train_accuracy_list = []
    for epoch in (range(num_epochs)):
        total_loss = 0
        total_correct = 0
        total_data = 0
        loss = 0
        
        model.train()
        
        iterator = tqdm(train_loader)
        for i, (data, labels) in enumerate(iterator):
            
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output,labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
                
            total_data += labels.size(0)
            total_loss += loss.item()
            _,p = torch.max(output.data,dim =1)
            total_correct += (p == labels).sum().item()
            
        print("Training: epoch: [{}/{}]  loss: [{:.2f}] Accuracy [{:.2f}] ".format(epoch+1,num_epochs,
                                                                          total_loss/len(train_loader),
                                                                           total_correct*100/total_data)) 
        
        train_loss_list.append(total_loss/len(train_loader))
        iteration_list.append(epoch)
        train_accuracy_list.append(total_correct*100/total_data)
        
        history = {
            'train_loss' : train_loss_list,
            'train_acc'  : train_accuracy_list,
        }
    return history


def train_eval(model,train_loader,val_loader,num_epochs,opt,criterion,device):
    train_loss_list = []
    iteration_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    
    iterator = tqdm(num_epochs)
    for epoch in range(iterator):
        total_loss = 0
        total_correct = 0
        total_data = 0
        loss = 0
        
        model.train()
        
        for i, (data, labels) in enumerate(train_loader):
            
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output,labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_data += labels.size(0)
            total_loss += loss.item()
            _,p = torch.max(output.data,dim =1)
            total_correct += (p == labels).sum().item()
        
        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_data = 0
        val_loss = 0
        for data, labels in val_loader:
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            val_loss = criterion(output,labels)

            val_total_data += labels.size(0)
            val_total_loss += val_loss.item()
            _,p = torch.max(output.data,dim =1)
            val_total_correct += (p == labels).sum().item()
        
        print("Training: epoch: [{}/{}] Loss: [{:.2f}] Accuracy [{:.2f}] Eval: Loss: [{:.2f}] Accuracy[{:.2f}]".
              format(epoch+1,num_epochs,total_loss/len(train_loader),total_correct*100/total_data,val_total_loss/len(val_loader),                                                                              val_total_correct*100/val_total_data )) 
        
        
        train_loss_list.append(total_loss/len(train_loader))
        iteration_list.append(epoch)
        train_accuracy_list.append(total_correct*100/total_data)
        val_loss_list.append(val_total_loss/len(val_loader))
        val_accuracy_list.append(val_total_correct*100/val_total_data)
        
        history = {
            'train_loss' : train_loss_list,
            'train_acc'  : train_accuracy_list,
            'val_loss' : val_loss_list,
            'val_acc'  : val_accuracy_list
        }
    return history


def test(model,test_loader,criterion,device):
    model.eval()
    with torch.no_grad():
        
        total_loss = 0
        total_correct = 0
        total_data = 0
        loss=0
        iterator = tqdm(test_loader)
        for data, labels in iterator:
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output,labels)

            total_data += labels.size(0)
            total_loss += loss.item()
            _,p = torch.max(output.data,dim =1)
            total_correct += (p == labels).sum().item()
        
        print("Testing: Loss: [{:.2f}] Accuracy [{:.2f}]".format(total_loss/len(test_loader),total_correct*100/total_data))

        
def visualize_image(img,title):        
    plt.imshow(img)
    plt.title(title)
    plt.show()

        
def visualize_train(history):        
    plt.plot(history['train_loss'])
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.savefig("Training_loss")
    plt.show()
    
    plt.plot(history['train_acc'],color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of iteration")
    plt.savefig("Training_Acc")
    plt.show()
    
def visualize_eval(history):
    plt.plot(history['val_loss'])
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.savefig("Validation_loss")
    plt.show()

    plt.plot(history['val_acc'],color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of iteration")
    plt.savefig("Validation_Acc")
    plt.show()





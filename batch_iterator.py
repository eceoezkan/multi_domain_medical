import numpy as np
import torch
from utils import compute_metrics

def BatchIterator(model, phase, Data_loader, criterion, optimizer, device, num_classes, data_size, batch_size):
    
    """
    Train function. 
    Args:
        model: model of the current epoch
        phase: phase to run (train or eval)
        Data_loader: current dataloader of the phase
        criterion: loss criterion
        optimizer: optimizer for the training
        device: device to run
        num_classes: number of classes
        data_size: dataset size of the phase
        batch_size: batch size
    Returns:
        running_loss: running loss of the phase
        acc: accuracy of the phase
        balanced_acc: balanced accuracy of the phase
        f1_acc: f1 accuracy of the phase
        cnf: confusion matrix of the phase
    """ 

    # initialization
    running_loss = 0.0
    running_corrects = 0.0
    total = 0

    preds_vec = np.zeros(data_size)
    labels_vec = np.zeros(data_size)
    probs_mat = np.zeros((data_size,num_classes))
    
    # set model to training mode
    if phase == 'train':
        model.train()   
    else:
        # set model to evaluate mode   
        model.eval()   

    # iterate over data
    for i, data in enumerate(Data_loader):
        # get the data
        imgs, labels, _ = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        # # phase dependent operations
        # if phase == "train":
        #     optimizer.zero_grad()
        #     model.train()
        #     outputs = model(imgs)
        # else:
        #     model.eval()
        #     with torch.no_grad():
        #         outputs = model(imgs)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            if imgs.shape[0] == 1:
                continue
            else:
                outputs = model(imgs)

        # predictions and labels
        _, preds = torch.max(outputs, 1)
        preds_vec[i*batch_size:(i+1)*batch_size] = preds.cpu().detach().numpy()
        labels_vec[i*batch_size:(i+1)*batch_size] = labels.cpu().detach().numpy()
        probs_mat[i*batch_size:(i+1)*batch_size,:] = outputs.cpu().detach().numpy()
        
        # loss
        loss = criterion(outputs, labels)

        # update weights
        if phase == 'train':
            loss.backward() 
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)
            optimizer.step()  

        # statistics    
        total += imgs.size(0)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += (preds == labels).sum().item()
        
    # get the metrics
    acc = 100*running_corrects / total
    balanced_acc, f1_acc, cnf = compute_metrics(preds_vec,labels_vec, probs_mat)    

    return running_loss, acc, balanced_acc, f1_acc, cnf  

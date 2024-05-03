import time
import csv
import pandas as pd
import numpy as np
import os
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from batch_iterator import BatchIterator
from utils import checkpoint, Saved_items
# from torchvision.models.vision_transformer import interpolate_embeddings

def train_model(dataloaders, num_classes, device, data_sizes, results_dir, config):
    
    """
    Train function. 
    Args:
        dataloaders: dataloaders for all phases
        num_classes: number of classes
        device: device to run
        data_sizes: dataset sizes for all phases
        results_dir: path to save
        config: configuration parameters
    Returns:
        model: trained model
        epoch_losses_train: train losses over epochs
        epoch_losses_val: val losses over epochs
        epoch_accs_train: train accuracies over epochs
        epoch_accs_val: val accuracies over epochs
        best_epoch: best epoch that yielded highest accuracys
    """

    # define model
    if config['model_type'] == 'DenseNet121':
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')# pretrained=config['model_pretrained'])
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)   
    elif config['model_type'] == 'ResNet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')# pretrained=config['model_pretrained']) # 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif config['model_type'] == 'ResNet34':
        model = models.resnet34(pretrained=config['model_pretrained']) # weights='ResNet34_Weights.DEFAULT')# 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  
    elif config['model_type'] == 'ResNet50':
        model = models.resnet50(pretrained=config['model_pretrained']) # weights='ResNet50_Weights.DEFAULT')# 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  
    # elif config['model_type'] == 'ViTb16':
    #     model = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
    #     patch_size = 7 # if input_size is (100, 100)
    #     model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)
    #     model.conv_proj = nn.Conv2d(in_channels=3, out_channels=model.conv_proj.out_channels, kernel_size=patch_size, stride=patch_size)
    #     model_state = interpolate_embeddings(config['img_resize'], patch_size, model.state_dict())
    #     model.load_state_dict(model_state)
    #     model.image_size, model.patch_size = config['img_resize'], patch_size
    else:
        print('Model not defined!')
        exit()

    # if torch.cuda.device_count() > 1:
    #     print('Using', torch.cuda.device_count(), 'GPUs')
    #     model = nn.DataParallel(model)

    model = model.to(device)
    
    # check if best model already exists
    num_epochs = config['num_epochs']
    checkpoint_path = results_dir + 'checkpoint_last' + str(num_epochs-1)
    if os.path.exists(checkpoint_path):
        print('Experiment already done, loading results..')
        
        # get train/val accuracies and losses
        log_path = results_dir + 'log_train'
        print('Log path: ' + log_path)
        
        df = pd.read_csv(log_path)
        epoch_losses_train = np.array(df.train_loss)
        epoch_losses_val = np.array(df.val_loss)
        epoch_accs_train = np.array(df.train_acc)
        epoch_accs_val = np.array(df.val_acc) 
    else:         
        print('Experiment not done yet, running..')
        # criterion
        if config['criterion_type'] == 'BCELoss':
            criterion = nn.BCELoss().to(device)
        elif config['criterion_type'] == 'CELoss':    
            criterion = nn.CrossEntropyLoss().to(device)
        
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
        # scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'])

        # initialize outputs
        epoch_losses_train = []
        epoch_losses_val = []
        epoch_accs_train = []
        epoch_accs_val = []
        best_acc = -1
        best_epoch = -1

        # time
        since = time.time()
    
        # iterate over epcohs
        for epoch in range(0, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
        
            # train 
            phase = 'train'
            running_loss, acc, epoch_acc_train, f1_acc, epoch_cnf = BatchIterator(model=model, 
                                                                   phase=phase, 
                                                                   Data_loader=dataloaders['train'], 
                                                                   criterion=criterion, 
                                                                   optimizer=optimizer, 
                                                                   device=device, 
                                                                   num_classes = num_classes, 
                                                                   data_size = data_sizes['train'], 
                                                                   batch_size = config['batch_size'])
            epoch_loss_train = running_loss / data_sizes['train']
            epoch_losses_train.append(epoch_loss_train)
            epoch_accs_train.append(epoch_acc_train)
            print('Train_losses:' + str(epoch_loss_train) + ' Train Acc.:' + str(epoch_acc_train))
        
            # scheduler
            scheduler.step()

            # validation
            phase = 'val'
            running_loss, acc, epoch_acc_val, f1_acc, epoch_cnf  = BatchIterator(model=model, 
                                                                  phase=phase, 
                                                                  Data_loader=dataloaders['val'], 
                                                                  criterion=criterion, 
                                                                  optimizer=optimizer, 
                                                                  device=device, 
                                                                  num_classes = num_classes, 
                                                                  data_size = data_sizes['val'], 
                                                                  batch_size = config['batch_size'])
            epoch_loss_val = running_loss / data_sizes['val']
            epoch_losses_val.append(epoch_loss_val)
            epoch_accs_val.append(epoch_acc_val)
            print('Validation_losses:' + str(epoch_loss_val) + ' Val Acc.:' + str(epoch_acc_val))

            # checkpoint model if has best val acc yet
            if epoch_acc_val > best_acc:
                best_loss = epoch_loss_val
                best_acc = epoch_acc_val
                best_epoch = epoch
                if os.path.exists(results_dir + 'checkpoint_last' + str(epoch-1)):
                    os.remove(results_dir + 'checkpoint_last' + str(epoch-1))
                checkpoint(results_dir, model, best_loss, best_acc, best_epoch, epoch, config['learning_rate'])
            else:    
                shutil.move(results_dir + 'checkpoint_last' + str(epoch-1),results_dir + 'checkpoint_last' + str(epoch))

            # log training and validation loss over each epoch
            with open(results_dir + 'log_train', 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                if (epoch == 0):
                    logwriter.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "LR"])
                logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val, epoch_acc_train, epoch_acc_val, config['learning_rate']])

        # end of training        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # get the best checkpoint
    print('Checkpoint path: ' + checkpoint_path)
    checkpoint_best = torch.load(checkpoint_path)
    model = checkpoint_best['model']
    best_epoch = checkpoint_best['best_epoch']
    print('Best epoch is ' + str(best_epoch))

    return model, epoch_losses_train, epoch_losses_val, epoch_accs_train, epoch_accs_val, best_epoch
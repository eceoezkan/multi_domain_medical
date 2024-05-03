import torch
import numpy as np
import random
import os
import json
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score

def set_seeds(seed):
    """
    Sets random seeds for torch, random and numpy. 
    Args:
        seed: random seed for reproducibility 
    Returns:
        None    
    """
    
    # back to random seeds
    random.seed(seed)
    np.random.seed(seed)

    # for cuda
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def compute_metrics(preds_vec, labels_vec, probs_mat):
    
    """
    Computes accuracy metrics.
    Args:
        preds_vec: predictions vector (mapped to classes)
        labels_vec: ground truth labels 
        probs_mat: probability map (output of softmax)
        
    Returns:
        balanced accuracy, (weighted average) f1 accuracy, confusion matrix  
    """

    # compute balanced accuracy
    balanced_acc = balanced_accuracy_score(labels_vec, preds_vec)

    # f1 score
    f1_acc = f1_score(labels_vec, preds_vec, average='weighted')

    # confision matrix
    cnf = confusion_matrix(labels_vec, preds_vec)

    return balanced_acc, f1_acc, cnf

def Saved_items(results_dir, epoch_losses_train, epoch_losses_val, epoch_accs_train, epoch_accs_val, time_elapsed, batch_size):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        results_dir: path to save
        epoch_losses_train: training losses over epochs
        epoch_losses_val: validation losses over epochs
        epoch_acc_train: training accuracy over epochs
        epoch_acc_val: validation accuracy over epochs
        time_elapsed: time elapsed so far during the training
        batch_size: batch size for training
    Returns:
        None
    """
    print('Saving outputs...')
    state2 = {
        'epoch_losses_train': epoch_losses_train,
        'epoch_losses_val': epoch_losses_val,
        'epoch_accs_train': epoch_accs_train,
        'epoch_accs_val': epoch_accs_val,
        'time_elapsed': time_elapsed,
        "batch_size": batch_size
    }
    torch.save(state2, results_dir + 'Saved_items')

def checkpoint(results_dir, model, best_loss, best_acc, best_epoch, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        results_dir: path to save
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        best_acc: best balanced accuracy achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """
    print('Saving checkpoint...')
    state = {
        'model': model,
        'best_loss': best_loss,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'LR': LR
    }
    torch.save(state, results_dir + 'checkpoint_last' + str(epoch))

def gatherMetrics(experiment_list, task):
    """
    Gathers all the metrics of the experiments.
    Args:
        experiment_list: output of the config function (config_experiments) or any list of experiments
        task: task to evaluate all the configs
    Returns:
        df: dataframe with all the configs and metrics for all experiments
    """
    results_path = 'xxx' # todo
    
    # get the train, val, test metrics
    df = pd.DataFrame(columns=['experiment_id', 'model_type', 'model_pretrained', 'batch_size', 'learning_rate', 'seed', 'sample_class', 'criterion_type', 'num_concepts_to_work', 'num_epochs', 'img_resize', 'img_pad', 'train_ratio',  'weight_decay', 'step_size', 'train_bal_acc', 'val_bal_acc', 'test_acc', 'test_bal_acc', 'test_f1_score','task1','task2','experiment_type','sampling_percentage','reduced_percentage'], index = range(0,len(experiment_list)))
  
    # iterate over experiments
    for i in range(len(experiment_list)):
        curr_exp = experiment_list[i]['experiment_id']
    
        with open(results_path + '/ROCO_ext/results/results_' + task + '/' + 'configs/' + str(i) + '.json') as config_file:
            config = json.load(config_file)
    
        # copy configuration
        df['experiment_id'][i] = i
        df['model_type'][i] = config['model_type'] 
        df['model_pretrained'][i] = config['model_pretrained'] 
        df['batch_size'][i] = config['batch_size']
        df['learning_rate'][i] = config['learning_rate']
        df['seed'][i] = config['seed']
        if 'abnormality' not in config['experiment_tpye']:
            df['sample_class'][i] = config['sample_class'] 
        df['criterion_type'][i] = config['criterion_type']
        if 'abnormality' not in config['experiment_tpye']:
            df['num_concepts_to_work'][i] = config['num_concepts_to_work']
        df['num_epochs'][i] = config['num_epochs']
        df['img_resize'][i] = config['img_resize']
        df['img_pad'][i] = config['img_pad']
        df['train_ratio'][i] = config['train_ratio']
        df['weight_decay'][i] = config['weight_decay']
        df['step_size'][i] = config['step_size']
        df['experiment_type'][i] = config['experiment_tpye']
        df['sampling_percentage'][i] = config['sampling_percentage']
        
        # check experiment type
        if 'reduced' in config['experiment_tpye']:
            df['task1'][i] = config['task1']
            df['task2'][i] = config['task2'] 
            df['reduced_percentage'][i] = config['reduced_percentage'] 
            
        if 'secondary' in config['experiment_tpye']:
            df['task2'][i] = config['id_task_to_run_secondary']     
    
        # check if exists
        curr_path = results_path + '/ROCO_ext/results/results_' + task + '/experiment_' + str(curr_exp) + '/' + str(curr_exp) + '_results.json'
        if os.path.exists(curr_path):

            with open(curr_path, 'r') as f:
                data = json.load(f)
                df['train_bal_acc'][i] = float(data['train_bal_acc'])
                df['val_bal_acc'][i] = float(data['val_bal_acc'])
                df['test_acc'][i] = float(data['test_acc']/100)
                df['test_bal_acc'][i] = float(data['test_bal_acc'])
                df['test_f1_score'][i] = float(data['test_f1_score'])

    # find the maximum 
    # idx = np.argmax(np.array(df['val_bal_acc']))
    # print('Best parameter set: ' + str(idx))
    # print('All metrics: ')
    # print(df.iloc[idx])   
    
    return df

def gatherMetrics_mnist(experiment_list, task):
    """
    Gathers all the metrics of the experiments.
    Args:
        experiment_list: output of the config function (config_experiments) or any list of experiments
        task: task to evaluate all the configs
    Returns:
        df: dataframe with all the configs and metrics for all experiments
    """
    results_path = 'xxx' # todo
    
    # get the train, val, test metrics
    df = pd.DataFrame(columns=['experiment_id', 'model_type', 'model_pretrained', 'batch_size', 'learning_rate', 'seed', 'criterion_type', 'num_epochs', 'img_resize', 'max_id', 'train_ratio',  'weight_decay', 'step_size', 'train_bal_acc', 'val_bal_acc', 'test_acc', 'test_bal_acc', 'test_f1_score', 'task1', 'task2', 'experiment_type', 'reduced_percentage', 'mean_organ', 'std_organ', 'mean_modality', 'std_modality','sampling_percentage','sampling_percentage_max','id_task_to_test_secondary'], index = range(0,len(experiment_list)))
  
    # iterate over experiments
    for i in range(len(experiment_list)):
        curr_exp = experiment_list[i]['experiment_id']
    
        with open(results_path + '/MNIST/results/results_' + task + '/' + 'configs/' + str(i) + '.json') as config_file:
            config = json.load(config_file)
    
        # copy configuration
        df['experiment_id'][i] = i
        df['model_type'][i] = config['model_type'] 
        df['model_pretrained'][i] = config['model_pretrained'] 
        df['batch_size'][i] = config['batch_size']
        df['learning_rate'][i] = config['learning_rate']
        df['seed'][i] = config['seed']
        df['criterion_type'][i] = config['criterion_type']
        df['num_epochs'][i] = config['num_epochs']
        df['img_resize'][i] = config['img_resize']
        df['max_id'][i] = config['max_id']
        df['train_ratio'][i] = config['train_ratio']
        df['weight_decay'][i] = config['weight_decay']
        df['step_size'][i] = config['step_size']
        df['experiment_type'][i] = config['experiment_tpye']
        
        # check experiment type
        if 'reduced' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
            df['task1'][i] = config['task1']
            df['task2'][i] = config['task2'] 
            df['reduced_percentage'][i] = config['reduced_percentage']
            
        # if config['experiment_tpye'] == 'nbrData_secondary' or config['experiment_tpye'] == 'nbrData_secondary_upsampled':
        if 'secondary' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
            df['task2'][i] = config['id_task_to_run_secondary'] 
        if 'secondary_test' in config['experiment_tpye']:
            df['id_task_to_test_secondary'][i] = config['id_task_to_test_secondary']     
            
        if 'nbrData' in config['experiment_tpye'] and 'ROCO' not in config['experiment_tpye']:
            df['mean_organ'][i] = config['mean_organ']
            df['std_organ'][i] = config['std_organ'] 
            df['mean_modality'][i] = config['mean_modality']
            df['std_modality'][i] = config['std_modality'] 
            
        if 'ROCO' in config['experiment_tpye']:
            df['sampling_percentage'][i] = config['sampling_percentage']
            df['sampling_percentage_max'][i] = config['sampling_percentage_max']
            
        if 'sampled' in config['experiment_tpye'] and 'sampling_percentage' in config:
            df['sampling_percentage'][i] = config['sampling_percentage']
    
        # check if exists
        curr_path = results_path + '/MNIST/results/results_' + task + '/experiment_' + str(curr_exp) + '/' + str(curr_exp) + '_results.json'
        # print(curr_path)
        if os.path.exists(curr_path):

            with open(curr_path, 'r') as f:
                data = json.load(f)
                # df['train_bal_acc'][i] = float(data['train_bal_acc'])
                # df['val_bal_acc'][i] = float(data['val_bal_acc'])
                df['test_acc'][i] = float(data['test_acc']/100)
                df['test_bal_acc'][i] = float(data['test_bal_acc'])
                df['test_f1_score'][i] = float(data['test_f1_score'])

    # find the maximum 
    # idx = np.argmax(np.array(df['val_bal_acc']))
    # print('Best parameter set: ' + str(idx))
    # print('All metrics: ')
    # print(df.iloc[idx])   
    
    return df

def gatherMetrics_medmnist(experiment_list, task):
    """
    Gathers all the metrics of the experiments.
    Args:
        experiment_list: output of the config function (config_experiments) or any list of experiments
        task: task to evaluate all the configs
    Returns:
        df: dataframe with all the configs and metrics for all experiments
    """
    results_path = 'xxx' # todo
    
    # get the train, val, test metrics
    df = pd.DataFrame(columns=['experiment_id', 'model_type', 'model_pretrained', 'batch_size', 'learning_rate', 'seed', 'criterion_type', 'num_epochs', 'img_resize','train_ratio',  'weight_decay', 'step_size', 'train_bal_acc', 'val_bal_acc', 'test_acc', 'test_bal_acc', 'test_f1_score','task1','task2','experiment_type','sampling_percentage','reduced_percentage','max_sample_train'], index = range(0,len(experiment_list)))
  
    # iterate over experiments
    for i in range(len(experiment_list)):
        curr_exp = experiment_list[i]['experiment_id']
        with open(results_path + '/MedMNIST/results/results_' + task + '/' + 'configs/' + str(i) + '.json') as config_file:
            config = json.load(config_file)
    
        # copy configuration
        df['experiment_id'][i] = i
        df['model_type'][i] = config['model_type'] 
        df['model_pretrained'][i] = config['model_pretrained'] 
        df['batch_size'][i] = config['batch_size']
        df['learning_rate'][i] = config['learning_rate']
        df['seed'][i] = config['seed'] 
        df['criterion_type'][i] = config['criterion_type']
        df['num_epochs'][i] = config['num_epochs']
        df['img_resize'][i] = config['img_resize']
        df['train_ratio'][i] = config['train_ratio']
        df['weight_decay'][i] = config['weight_decay']
        df['step_size'][i] = config['step_size']
        df['experiment_type'][i] = config['experiment_tpye']
        df['sampling_percentage'][i] = config['sampling_percentage']
        
        # check experiment type
        if 'reduced' in config['experiment_tpye']:
            df['task1'][i] = config['task1']
            df['task2'][i] = config['task2'] 
            df['reduced_percentage'][i] = config['reduced_percentage'] 
            
        if 'secondary' in config['experiment_tpye']:
            df['task2'][i] = config['id_task_to_run_secondary'] 
            
        if 'dist' in config['experiment_tpye'] and 'max_sample_train' in config:
            df['max_sample_train'][i] = config['max_sample_train']     
    
        # check if exists
        curr_path = results_path + '/MedMNIST/results/results_' + task + '/experiment_' + str(curr_exp) + '/' + str(curr_exp) + '_results.json'
        if os.path.exists(curr_path):

            with open(curr_path, 'r') as f:
                data = json.load(f)
                df['train_bal_acc'][i] = float(data['train_bal_acc'])
                df['val_bal_acc'][i] = float(data['val_bal_acc'])
                df['test_acc'][i] = float(data['test_acc']/100)
                df['test_bal_acc'][i] = float(data['test_bal_acc'])
                df['test_f1_score'][i] = float(data['test_f1_score'])

    # find the maximum 
    # idx = np.argmax(np.array(df['val_bal_acc']))
    # print('Best parameter set: ' + str(idx))
    # print('All metrics: ')
    # print(df.iloc[idx])  
    
    return df
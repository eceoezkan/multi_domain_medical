# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import argparse
import os
import json
from torchvision import transforms
from dataset_generator import ROCO_Dataset
from utils import set_seeds
from train import train_model
from prep_labels import getImgsLabels
from batch_iterator import BatchIterator
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

def main(): 
    
    # folders
    data_path = 'xxx' # todo
    results_path = 'xxx' # todo

    # parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=int, default='0',
                            help='id of json file')
    parser.add_argument('--task_to_run', type=str, default='Organ',
                            help='run to perform on task with experiment <id>')
    parser.add_argument('--gpu_id', type=str, default="",
                            help="gpu IDs")

    args = parser.parse_args()
    print('Running for the following experiment:' + str(args.experiment_id))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # data path
    data_dir = data_path + '/Data/ROCO_ext/' 
    results_dir_root = results_path + '/ROCO_ext/results/results_' + args.task_to_run + '/'
    if 'Organ_hyper' in args.task_to_run or 'Secondary' in args.task_to_run or 'Reduced' in args.task_to_run:
        args.task_to_run = 'Organ'
    concept_list = 'concept_names.csv'

    # directories 
    if not os.path.exists(results_dir_root):
        os.makedirs(results_dir_root)
    
    # load the config from json file 
    with open(results_dir_root + 'configs/' + str(args.experiment_id) + '.json') as config_file:
        config = json.load(config_file)
    
    # get the parameters
    experiment_id = config['experiment_id']
    batch_size = config['batch_size']
    seed = config['seed']
    sample_class = config['sample_class']
    if args.task_to_run == 'Organ':
        task_to_run = 0
    elif args.task_to_run == 'Modality':
        task_to_run = 1  
    if 'reduced' in config['experiment_tpye']:    
        id_task1_reduced = config['task1']
        id_task2_reduced = config['task2']    
    num_workers = 2  
    
    print('Running for the experiment with following parameters: ')
    print(config)

    # read the concepts list
    df_concepts_all = pd.read_csv(data_dir + concept_list ,sep='\t')  

    # read semantic group
    df_concepts_all_semantic = pd.read_csv(data_dir + 'MostCommonCUIs.csv',sep=',')
    df_concepts_all_semantic = df_concepts_all_semantic[0:config['num_concepts_to_work']]
    
    # classes task[0] primary task task[1] to filter
    task = ['Body Part, Organ, or Organ Component','Diagnostic Procedure'] # ['Diagnostic Procedure'] # 
    CLASS_NAMES = []
    CLASS_NAMES_exp = []
    for i in range(len(task)):
        idx = df_concepts_all_semantic['Semantic attribute'] == task[i]
    
        CLASS_NAMES.append(df_concepts_all_semantic[idx]['CUI'].to_numpy())
        CLASS_NAMES_exp.append(df_concepts_all_semantic[idx]['Explanation'].to_numpy())
        
    # get the removed class names
    if 'reduced' in config['experiment_tpye']: 
        task1_reduced = CLASS_NAMES[0][id_task1_reduced]
        task1_reduced_exp = CLASS_NAMES_exp[0][id_task1_reduced]
        task2_reduced = CLASS_NAMES[1][id_task2_reduced]
        task2_reduced_exp = CLASS_NAMES_exp[1][id_task2_reduced]
        print('Removed task1 for the following experiment: ' + task1_reduced_exp)
        print('Removed task2 for the following experiment: ' + task2_reduced_exp)
    else:
        task1_reduced = None
        task2_reduced = None
    
    # get every sample_class class
    if task[0] == 'Body Part, Organ, or Organ Component':
        CLASS_NAMES[0] = CLASS_NAMES[0][1::sample_class]
        CLASS_NAMES_exp[0] = CLASS_NAMES_exp[0][1::sample_class]
    elif task[1] == 'Body Part, Organ, or Organ Component':
        CLASS_NAMES[1] = CLASS_NAMES[1][1::sample_class]
        CLASS_NAMES_exp[1] = CLASS_NAMES_exp[1][1::sample_class]
    
    if 'reduced' in config['experiment_tpye']:  
        isin_task1 = task1_reduced in str(CLASS_NAMES[0])
        isin_task2 = task2_reduced in str(CLASS_NAMES[1])
    else: 
        isin_task1 = True
        isin_task2 = True
    
    # check if this experiment can be run (reduced task1/task2 in the task matrix)
    if isin_task1 & isin_task2:
        
        # update task name    
        task = ['Organ','Modality'] 
        
        # results_dir    
        results_dir = 'experiment_' + str(experiment_id) + '/'    
        results_dir = results_dir_root + results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # result file
        res_file = results_dir + str(experiment_id)+'_results.json'    
    
        # check if the experiment is already done
        if os.path.exists(res_file):
            print('Test evaluation already done!')
        else:         
            # images, labels and dirs per phase
            set_seeds(seed)
            train_idx,train_label,train_dir,val_idx,val_label,val_dir,test_idx,test_label,test_dir = getImgsLabels(CLASS_NAMES,CLASS_NAMES_exp,task, task_to_run, data_dir, config, task1_reduced, task2_reduced)
            # task to run
            if 'secondary' in config['experiment_tpye']:
                CLASS_NAMES_sec = CLASS_NAMES[1-task_to_run][config['id_task_to_run_secondary']]
                CLASS_NAMES_sec_exp = CLASS_NAMES_exp[1-task_to_run][config['id_task_to_run_secondary']]
            else:     
                CLASS_NAMES_sec = CLASS_NAMES[1-task_to_run]
                CLASS_NAMES_sec_exp = CLASS_NAMES_exp[1-task_to_run]
            CLASS_NAMES = CLASS_NAMES[task_to_run]
            CLASS_NAMES_exp = CLASS_NAMES_exp[task_to_run]
            task = task[task_to_run]
    
            # list of tasks
            print('Tasks to run: ')
            print('Primary: ' + CLASS_NAMES_exp)
            print('Secondary: ' + CLASS_NAMES_sec_exp)
        
            # # hist
            # plt.figure()
            # plt.hist(train_label,len(CLASS_NAMES))
            # plt.hist(val_label,len(CLASS_NAMES))
            # plt.hist(test_label,len(CLASS_NAMES))
            # plt.legend(['train','val','test'])
            # plt.savefig(results_dir + 'img_distribution_split'+ task +'.png', dpi=300, bbox_inches = "tight") 

            # gpu
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print('Using {} device'.format(device))

            # transform functions
            set_seeds(seed)
            transformList = []
            transformList.append(transforms.Resize(size=config['img_resize']))
            transformList.append(transforms.ToTensor())
            train_transform=transforms.Compose(transformList)

            transformList = []
            transformList.append(transforms.Resize(size=config['img_resize']))
            transformList.append(transforms.ToTensor())
            test_transform=transforms.Compose(transformList)

            # datasets
            image_datasets = {'train': ROCO_Dataset(img_list = train_idx, 
                                            label_list = train_label, 
                                            curr_dir = train_dir,
                                            transform=train_transform,
                                            phase = 'train',
                                            num_classes = len(CLASS_NAMES),
                                            config = config),
                      'val': ROCO_Dataset(img_list = val_idx, 
                                            label_list = val_label, 
                                            curr_dir = val_dir,
                                            transform = test_transform,
                                            phase = 'val',
                                            num_classes = len(CLASS_NAMES),
                                            config = config),
                      'test': ROCO_Dataset(img_list = test_idx, 
                                            label_list = test_label, 
                                            curr_dir = test_dir,
                                            transform = test_transform,
                                            phase = 'test',
                                            num_classes = len(CLASS_NAMES),
                                            config = config)
                     }

            # dataset sizes
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
            # dataloaders
            set_seeds(seed)
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=num_workers)
                   for x in ['train', 'val', 'test']} 

            # train model
            set_seeds(seed)
            model_ft, train_loss, val_loss, train_acc, val_acc, best_epoch = train_model(dataloaders,
                                                            num_classes = len(CLASS_NAMES),
                                                            device = device, 
                                                            data_sizes = dataset_sizes,
                                                            results_dir = results_dir,
                                                            config = config)
    
            # # plot train/val accuracry and loss
            # plt.rcParams['figure.figsize'] = [10, 5]
            # print('Saving train/val accuracy and loss figures..')
            # plt.subplot(1,2,1)
            # plt.plot(range(0, len(train_loss)), train_loss, label = 'Train')
            # plt.plot(range(0, len(train_loss)), val_loss, label = 'Val')
            # plt.xlabel('Epochs')
            # plt.title('Loss')
            # plt.legend()

            # plt.subplot(1,2,2)
            # plt.plot(range(0, len(train_loss)), train_acc, label = 'Train')
            # plt.plot(range(0, len(train_loss)), val_acc, label = 'Val')
            # plt.xlabel('Epochs')
            # plt.title('Balanced Accuracy')
            # plt.legend()
            # plt.savefig(results_dir + 'loss_acc_experiment_' + str(experiment_id) +'.png', dpi=300, bbox_inches = "tight")
    
            print('Training done!')
    
            print('Save the train/val metrics')
            result = {}
            result['train_bal_acc'] = train_acc[best_epoch]
            result['val_bal_acc'] = val_acc[best_epoch]
    
            # test split
            print('Starting test evaluation..')
    
            preds_vec = np.zeros(dataset_sizes['test'])
            labels_vec = np.zeros(dataset_sizes['test'])
            labels_vec_sec = np.zeros(dataset_sizes['test'])
            correct = 0
            total = 0
            cnt = 0
        
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for images, labels, _ in dataloaders['test']:
                    # calculate outputs by running images through the network
                    images = images.to(device)
                    labels_sec = labels[1]
                    labels = labels[0].to(device)
                    outputs = model_ft(images)
        
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs, 1)
                    preds_vec[cnt*batch_size:(cnt+1)*batch_size] = predicted.cpu().detach().numpy()
                    labels_vec[cnt*batch_size:(cnt+1)*batch_size] = labels.cpu().detach().numpy()
                    labels_vec_sec[cnt*batch_size:(cnt+1)*batch_size] = labels_sec
            
                    # accuracy
                    total += images.size(0)
                    correct += (predicted == labels).sum().item()   
                    cnt = cnt + 1

            # compute balanced accuracy
            acc = 100 * correct / total

            # compute balanced accuracy
            balanced_acc = balanced_accuracy_score(labels_vec, preds_vec)

            # f1 score
            f1_acc = f1_score(labels_vec, preds_vec,average='weighted')

            # print
            print('Number of test images: ' + str(dataset_sizes['test']))
            print('Accuracy: %' + str(acc))
            print('Balanced accuracy: %' + str(100*balanced_acc))
            print('F1-score: %' + str(100*f1_acc))

            # confision matrix
            cnf = confusion_matrix(labels_vec, preds_vec)
            cnf_norm = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
            # plt.figure()
            # plt.rcParams['figure.figsize'] = [10, 5]
            # plt.imshow(cnf_norm)
            # plt.colorbar()
            # plt.savefig(results_dir + 'cnf_experiment_' + str(experiment_id) +'.png', dpi=300, bbox_inches = "tight")
    
            print('Test evaluation done!')
    
            print('Save the test metrics')
            result['test_acc'] = acc
            result['test_bal_acc'] = balanced_acc
            result['test_f1_score'] = f1_acc
    
            # evaluate for each initial and secondary task
            if 'secondary' not in config['experiment_tpye']:
                eval_mat_sec = np.zeros([len(CLASS_NAMES),len(CLASS_NAMES_sec)])
                for i in range(len(CLASS_NAMES_sec)):
                    curr_preds = preds_vec[labels_vec_sec == i]
                    curr_labels = labels_vec[labels_vec_sec == i]
                    for j in range(len(CLASS_NAMES)):
                        eval_mat_sec[j,i] = ((curr_preds == curr_labels) * (curr_labels == j)).sum() / (curr_labels == j).sum()
                result['eval_mat_sec'] = eval_mat_sec.tolist()   
    
            # class-wise accuracy
            eval_mat = np.zeros([len(CLASS_NAMES),1])
            curr_preds = preds_vec
            curr_labels = labels_vec
            for j in range(len(CLASS_NAMES)):
                eval_mat[j,:] = ((curr_preds == curr_labels) * (curr_labels == j)).sum() / (curr_labels == j).sum()    
            result['eval_mat'] = eval_mat.tolist()
            result['cnf'] = cnf.tolist()
            
            with open(results_dir + str(experiment_id)+'_results.json', 'w') as json_file:
                json.dump(result, json_file)
        
            print('All done! Finishing.')
        
    else:
        print('Experiment wont run, because reduced setting is not part of the classes.')   
    
if __name__ == "__main__":
    main()
# import libraries
import numpy as np
import torch
import argparse
import os
import json
from torchvision import transforms
from utils import set_seeds
from prep_labels import getImgsLabels_medmnist
from dataset_generator import MedMNIST_Dataset
from train import train_model
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
    data_dir = data_path + '/Data/MedMNIST/'
    results_dir_root = results_path + '/MedMNIST/results/results_' + args.task_to_run + '/'
    if 'Organ' in args.task_to_run or 'Secondary' in args.task_to_run or 'Reduced' in args.task_to_run:
        args.task_to_run = 'Organ'

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
    if args.task_to_run == 'Organ':
        task_to_run = 0
    elif args.task_to_run == 'Views':
        task_to_run = 1  
    if 'reduced' in config['experiment_tpye']:    
        id_task1_reduced = config['task1']
        id_task2_reduced = config['task2']    
    num_workers = 2  
    
    print('Running for the experiment with following parameters: ')
    print(config)
    
    # hard-code MedMNIST has 3 views and 11 organs
    task = ['Organ','Views']
    CLASS_NAMES = []
    CLASS_NAMES_exp = []
    CLASS_NAMES_exp.append(['Bladder','Left femoral head','Right femoral head','Heart','Left kidney',
                        'Right kidney','Liver','Left lung','Right lung','Pancreas','Spleen'])
    CLASS_NAMES_exp.append(['Axial','Coronal','Sagittal'])
    CLASS_NAMES.append(range(0,len(CLASS_NAMES_exp[0])))
    CLASS_NAMES.append(range(0,len(CLASS_NAMES_exp[1])))
        
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
        train_imgs,train_label,val_imgs,val_label,test_imgs,test_label = getImgsLabels_medmnist(CLASS_NAMES, CLASS_NAMES_exp, task, task_to_run, data_dir, config, task1_reduced, task2_reduced)
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
        print('Primary: ' , CLASS_NAMES_exp)
        print('Secondary: ' , CLASS_NAMES_sec_exp)

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
        image_datasets = {'train': MedMNIST_Dataset(img_list = train_imgs, 
                                            label_list = train_label, 
                                            transform=train_transform,
                                            phase = 'train',
                                            num_classes = len(CLASS_NAMES),
                                            config = config),
                      'val': MedMNIST_Dataset(img_list = val_imgs, 
                                            label_list = val_label, 
                                            transform = test_transform,
                                            phase = 'val',
                                            num_classes = len(CLASS_NAMES),
                                            config = config),
                      'test': MedMNIST_Dataset(img_list = test_imgs, 
                                            label_list = test_label, 
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
    
if __name__ == "__main__":
    main()
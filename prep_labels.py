import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from utils import set_seeds

def getImgsLabels(CLASS_NAMES, CLASS_NAMES_exp, task, task_to_run, data_dir, config, task1_reduced, task2_reduced):
    
    """
    Function to prepare the labels. 
    Args:
        CLASS_NAMES: list of the classes (with concept id)
        CLASS_NAMES_exp: list of the classes with their concept explanations
        task: list of tasks
        task_to_run: id of the task to run (to choose from task list)
        data_dir: data directory
        config: configuration parameters
        task1_reduced: removed task 1 (row) from the task matrix (for experiment_type = 'reduced')
        task2_reduced: removed task 2 (column) from the task matrix (for experiment_type = 'reduced')
        
    Returns:
        train_idx: ids of the images for train
        train_label: labels of the images for train
        train_dir: directory of the images for train
        val_idx: ids of the images for val
        val_label: labels of the images for train
        val_dir: directory of the images for train
        test_idx: ids of the images for test
        test_label: labels of the images for train
        test_dir : directory of the images for train
    """
    
    # path for 
    task_ext = 'concept_detection_'
    imgs_ext = '_images/'
    path_dir = data_dir + task_ext

    # total number of classes for tasks
    num_classes = np.zeros(len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        num_classes[i] = len(CLASS_NAMES[i])
    num_classes = num_classes.astype(int)
    num_classes_total = int(sum(num_classes))
    
    # get the label vec of reduced setting
    if 'reduced' in config['experiment_tpye']:
        label_vec_reduced = [np.where(CLASS_NAMES[0]==task1_reduced)[0][0],np.where(CLASS_NAMES[1]==task2_reduced)[0][0]]
    
    # check if it is secondary setting
    if 'secondary' in config['experiment_tpye']:
        print('Secondary task is selected.')
        task_to_run_secondary = config['id_task_to_run_secondary']
        secondary_task = CLASS_NAMES[1][task_to_run_secondary]
        secondary_task_exp = CLASS_NAMES_exp[1][task_to_run_secondary]
        print('All secondary classes:' + CLASS_NAMES_exp[1])
        print('Secondary task to run: ' + secondary_task_exp)
    
    # prepare data
    phase = ['train', 'valid']

    # iterate over phase:
    for curr_phase in phase:
    
        # initialize array for task matrix
        if len(CLASS_NAMES) != 1:
            task_mat = np.zeros([num_classes[0],num_classes[1]])
    
        # read annotations
        annot_dir = path_dir + curr_phase + '.csv'    
        df = pd.read_csv(annot_dir,sep='\t')  
        print('Annotations loaded!')
        
        # initialize img ids
        img_idx = []
    
        # if multi-class
        # label_vec = np.empty((0,num_classes_total), dtype = float)
        label_vec = np.zeros((0,len(task)), dtype = float)

        # iterate lines
        for index, row in df.iterrows():
        
            # split the line
            x = row['cuis'].split(';')
            curr_label = []
        
            # iterate over task for getting the label
            for task_id in range(len(task)):
        
                # compare the concepts of the current row with the classes to find the label
                curr_concept_list = np.intersect1d(x,CLASS_NAMES[task_id])
        
                # if not only one concept per image, skip
                if len(curr_concept_list) == 0:
                    continue      
                elif len(curr_concept_list) > 1:
                    # print('More than one class, problem! Skipping.')
                    continue           
                
                idx = np.where(CLASS_NAMES[task_id]==curr_concept_list)[0][0]
                curr_label.append(idx)
            
            # check if both tasks filled
            if len(curr_label) != len(task):
                continue  
            
            # else: 
            # get the img and label
            img_idx.append(row['ID'])
            label_vec = np.vstack([label_vec, curr_label])  
    
        # check the task matrix
        if (len(CLASS_NAMES) != 1):   
            label_vec_init = label_vec[:,0]
            label_vec_sec = label_vec[:,1]
            # iterate over tasks
            for task_init in range(num_classes[0]):
                curr_label_vec_sec = label_vec_sec[label_vec_init == task_init]
                for task_sec in range(num_classes[1]):
                    curr_cnt = sum(curr_label_vec_sec == task_sec)
                    task_mat[task_init,task_sec] = curr_cnt
        else:
            print('Second task not defined!')
            continue
        
        # check which task to train
        label_vec = label_vec.astype(int)
        if 'secondary' in config['experiment_tpye']:
            # choose secondary task labels and img_idx
            indices = np.where(label_vec[:,1] == task_to_run_secondary)[0]
            img_idx = np.array(img_idx)[indices]
            label_vec = np.array(label_vec)[indices,:]
            
        # sample the data using train_test_split function
        if (curr_phase == 'train') & (config['sampling_percentage'] < 100):
            img_idx, _, label_vec, _ = train_test_split(img_idx, label_vec, test_size=1-(config['sampling_percentage']/100), random_state = config['seed'])

        # print
        print('Number of images total: '+ str(len(df)))
    
        if curr_phase == 'train':
            # train validation split from train 
            train_idx, val_idx, train_label, val_label = train_test_split(img_idx, label_vec, test_size=1-config['train_ratio'], random_state = config['seed'])
            train_dir = data_dir + curr_phase + imgs_ext
            val_dir = data_dir + curr_phase + imgs_ext
            
            # print
            print(len(train_label))
            
            if 'reduced' in config['experiment_tpye']:
                # train
                tmp = np.where(np.sum(train_label == label_vec_reduced,1) == 2)[0]
                idx = random.sample(range(len(tmp)), int(np.round(len(tmp)*config['reduced_percentage']/100)))
                train_idx =  np.delete(train_idx,tmp[idx])
                train_label =  np.delete(train_label,tmp[idx],axis=0)
                # train_idx = np.array(train_idx)[np.sum(train_label == label_vec_reduced,1) != 2]
                # train_label = np.array(train_label)[np.sum(train_label == label_vec_reduced,1) != 2]
                
                # val 
                tmp = np.where(np.sum(val_label == label_vec_reduced,1) == 2)[0]
                idx = random.sample(range(len(tmp)), int(np.round(len(tmp)*config['reduced_percentage']/100)))
                val_idx =  np.delete(val_idx,tmp[idx])
                val_label =  np.delete(val_label,tmp[idx],axis=0)
                # val_idx = np.array(val_idx)[np.sum(val_label == label_vec_reduced,1) != 2]
                # val_label = np.array(val_label)[np.sum(val_label == label_vec_reduced,1) != 2]
                
            # get only task relevant labels    
            train_label = train_label[:,task_to_run]   
            val_label = val_label[:,task_to_run]
            
            # print
            print(len(train_label))
        
        # use validation set as test set
        elif curr_phase == 'valid':
            test_idx = img_idx
            test_label = label_vec
            test_dir = data_dir + curr_phase + imgs_ext
            print('Number of images to use for test: ' + str(len(test_idx)))
            
            class_count = np.zeros(num_classes[task_to_run])
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(test_label[:,task_to_run]==i))
                print('Number of images of ' + CLASS_NAMES_exp[task_to_run][i] + ': ' + str(sum_check))  
        
        # checks
        if curr_phase == 'train':
            class_count = np.zeros(num_classes[task_to_run])
            print('Number of images to use for train: ' + str(len(train_idx)))
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(train_label==i))
                print('Number of images of ' + CLASS_NAMES_exp[task_to_run][i] + ': ' + str(sum_check))  
            class_count = np.zeros(num_classes[task_to_run])
            print('Number of images to use for val: ' + str(len(val_idx)))    
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(val_label==i))
                print('Number of images of ' + CLASS_NAMES_exp[task_to_run][i] + ': ' + str(sum_check))  
    
    return train_idx, train_label, train_dir, val_idx, val_label, val_dir, test_idx, test_label, test_dir 

def getImgsLabels_mnist(CLASS_NAMES, task, task_to_run, data_dir, config, task1_reduced, task2_reduced):
    
    """
    Function to prepare the labels. 
    Args:
        CLASS_NAMES: list of the classes (with concept id)
        task: list of tasks
        task_to_run: id of the task to run (to choose from task list)
        data_dir: data directory
        config: configuration parameters 
        task1_reduced: removed task 1 (row) from the task matrix (for experiment_type = 'reduced', 'reduced_secondary' or reduced_secondary_upsampled)
        task2_reduced: removed task 2 (column) from the task matrix (for experiment_type = 'reduced', 'reduced_secondary' or reduced_secondary_upsampled)
        
    Returns:
        train_imgs: train images stored in the array
        train_idx: ids of the images for train
        train_label: labels of the images for train
        train_dir: directory of the images for train
        val_imgs: validation images stored in the array
        val_idx: ids of the images for val
        val_label: labels of the images for train
        val_dir: directory of the images for train
        test_imgs: test images stored in the array
        test_idx: ids of the images for test
        test_label: labels of the images for train
        test_dir : directory of the images for train
    """
    # total number of classes for tasks
    num_classes = np.zeros(len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        num_classes[i] = len(CLASS_NAMES[i])
    num_classes = num_classes.astype(int)
    num_classes_total = int(sum(num_classes))
    
    # check if with ROCO numbers for samples
    if 'ROCO' in config['experiment_tpye']:
        number_imgs_mat_train_val = np.array([[826, 1747, 262,  45,  20],
                         [ 23, 1134, 101,   8,   3],
                         [492,  313,  14,  63,   6],
                         [251,  116, 116, 149,  14],
                         [ 70,   36,  59, 507,  28],
                         [311,  171,  39,  71,  21],
                         [219,   42,  24, 109,  74],
                         [ 35,    7,  23,  20, 373],
                         [275,   29,  51,  64,   8],
                         [ 23,    2,  24,  10,   3]])

        number_imgs_mat_test = np.array([[143, 45, 22,  2,  2],
                        [  4,  7,  7,  2,  1],
                        [ 79, 53,  0, 11,  2],
                        [ 37, 11,  6, 19,  2],
                        [ 10,  4,  3, 47,  0],
                        [ 19, 15,  3,  4,  1],
                        [ 24,  2,  3,  8,  4],
                        [  5,  0,  3,  2, 39],
                        [ 23,  1,  3,  5,  1],
                        [ 0 ,  1,  3,  0,  0]])
        
        # multiply with the percentage
        # idx = np.unravel_index(number_imgs_mat_train_val.argmax(), number_imgs_mat_train_val.shape)
        # number_imgs_mat_train_val[idx] = number_imgs_mat_train_val[idx]*config['sampling_percentage_max']
        hist_mean = np.mean(number_imgs_mat_train_val)
        check_mat = number_imgs_mat_train_val>(config['sampling_percentage_max']*hist_mean)
        number_imgs_mat_train_val[check_mat] = hist_mean # number_imgs_mat_train_val[check_mat]*config['sampling_percentage_max']
        number_imgs_mat_train_val = config['sampling_percentage']*number_imgs_mat_train_val
        # number_imgs_mat_test = config['sampling_percentage']*number_imgs_mat_test
    
    # get the label vec of reduced setting
    if 'reduced' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
        label_vec_reduced = [task1_reduced,task2_reduced]
    
    # check if it is secondary setting
    if 'secondary' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
        print('Secondary task is selected.')
        task_to_run_secondary = config['id_task_to_run_secondary']
        secondary_task = CLASS_NAMES[1][task_to_run_secondary]
        print('Secondary task to run: ' + str(secondary_task))
    
    # prepare data
    phase = ['train', 'test']

    # iterate over phase:
    for curr_phase in phase:

        # annotations, for seondary one can choose upsampled setting
        if curr_phase == 'train' and 'upsampled' not in config['experiment_tpye'] and 'ROCO' not in config['experiment_tpye']:
            annot_dir = data_dir + curr_phase + '_' + str(config['max_id']) + '.csv'
        elif curr_phase == 'train' and ('upsampled' in config['experiment_tpye'] or 'ROCO' in config['experiment_tpye']):
            annot_dir = data_dir + curr_phase + '_' + str(num_classes[1] * config['max_id']) + '.csv'
        else:     
            annot_dir = data_dir + curr_phase + '.csv'    
        print('Loading annotations: ' + annot_dir)    
        df = pd.read_csv(annot_dir,sep=';') 
        print('Annotations loaded!')
        
        # images, for seondary one can choose upsampled setting
        if curr_phase == 'train' and 'upsampled' not in config['experiment_tpye'] and 'ROCO' not in config['experiment_tpye']:
            imgs_dir = data_dir + curr_phase + '_' + str(config['max_id']) + '.npy'
        elif curr_phase == 'train' and ('upsampled' in config['experiment_tpye'] or 'ROCO' in config['experiment_tpye']):
            imgs_dir = data_dir + curr_phase + '_' + str(num_classes[1] * config['max_id']) + '.npy'
        else:     
            imgs_dir = data_dir + curr_phase + '.npy'
        print('Loading images array: ' + imgs_dir)    
        imgs = np.load(imgs_dir)
        print('Images array loaded!')
        
        # get img_idx and labels
        img_idx = np.array(df['Path'])
        label_vec = np.transpose(np.array([df['Organ'],df['Modality']]))
        label_vec = label_vec.astype(int)

        # print
        print('Number of images total: '+ str(len(img_idx)))
        
        # check if train
        if curr_phase == 'train':
        
            # get the distribution of datapoints
            if 'nbrData' in config['experiment_tpye'] and 'ROCO' not in config['experiment_tpye']:
                mean_organ = config['mean_organ']
                std_organ = config['std_organ']
                mean_modality = config['mean_modality']
                std_modality = config['std_modality']
            
                # get the current distribution
                set_seeds(seed=config['seed'])
                s_organ = np.random.normal(mean_organ, std_organ, 500000)
                set_seeds(seed=config['seed'])
                s_modality = np.random.normal(mean_modality, std_modality, 100000)
            
                # initialize
                hist_organ_mat = np.zeros([num_classes[0],num_classes[1]])
                hist_modality_mat = np.zeros([num_classes[0],num_classes[1]])
                
                # histogram
                hist_organ = np.histogram(s_organ, bins=range(0,num_classes[0]+1), density=True)
                hist_modality = np.histogram(s_modality, bins=range(0,num_classes[1]+1), density=True)
                
                # normalize
                hist_organ = hist_organ[0]/hist_organ[0].max()
                hist_modality = hist_modality[0]/hist_modality[0].max()
                
                # iterate 
                for i in range(len(CLASS_NAMES[1])):
                    hist_organ_mat[:,i] = hist_organ
                for i in range(len(CLASS_NAMES[0])):
                    hist_modality_mat[i,:] = hist_modality

                # number of datapoints
                train_mat = config['max_id']*config['train_ratio']*np.ones((num_classes[0],num_classes[1]))
                val_mat =  config['max_id']*(1-config['train_ratio'])*np.ones((num_classes[0],num_classes[1]))
                datapoints_train = np.round(train_mat*hist_organ_mat*hist_modality_mat)
                datapoints_val = np.round(val_mat*hist_organ_mat*hist_modality_mat)
                print('Total number of images for train: '+str(datapoints_train.sum()))
                print('Total number of images for val: '+str(datapoints_val.sum()))
                
                if 'upsampled' in config['experiment_tpye']:
                    # total number of datapoints for general model
                    total_num_datapoints_train = datapoints_train.sum()
                    total_num_datapoints_val = datapoints_val.sum()
                    
                    # total number of datapoints for secondary model
                    secondary_total_num_datapoints_train = datapoints_train[:,task_to_run_secondary].sum()
                    secondary_total_num_datapoints_val = datapoints_val[:,task_to_run_secondary].sum()
                    
                    # print
                    print('Train - Number of datapoints for general model: ' + str(total_num_datapoints_train))
                    print('Train - Number of datapoints for specialized model: ' + str(total_num_datapoints_train))
                    
                    # upsample the number of datapoints to match the total for general model
                    datapoints_train = np.round(np.tile(datapoints_train[:,task_to_run_secondary]*(total_num_datapoints_train/secondary_total_num_datapoints_train),(num_classes[1],1)).transpose())
                    datapoints_val = np.round(np.tile(datapoints_val[:,task_to_run_secondary]*(total_num_datapoints_val/secondary_total_num_datapoints_val),(num_classes[1],1)).transpose())
            
                # get only the relevant images
                # train
                check_vec = df['ImgID'] <  datapoints_train[df['Organ'],df['Modality']] 
                train_imgs = imgs[:,:,:,check_vec]
                train_idx = img_idx[check_vec]
                train_label = label_vec[check_vec]
                train_dir = data_dir + curr_phase + '/'
            
                # val
                check_vec = (df['ImgID'] < (datapoints_train.max() + datapoints_val[df['Organ'],df['Modality']])) & (df['ImgID'] >= (datapoints_train.max()))
                val_imgs = imgs[:,:,:,check_vec]
                val_idx = img_idx[check_vec]
                val_label = label_vec[check_vec]
                val_dir = data_dir + curr_phase + '/'
                
                # check which task to train for secondary setting
                # if config['experiment_tpye'] == 'nbrData_secondary' or config['experiment_tpye'] == 'nbrData_reduced_secondary' or config['experiment_tpye'] == 'nbrData_secondary_upsampled':
                if 'secondary' in config['experiment_tpye']:
            
                    # choose secondary task labels and img_idx
                    indices = np.where(train_label[:,1] == task_to_run_secondary)[0]
                    train_imgs = train_imgs[:,:,:,indices]
                    train_idx = np.array(train_idx)[indices]
                    train_label = np.array(train_label)[indices,:] 
                    indices = np.where(val_label[:,1] == task_to_run_secondary)[0]
                    val_imgs = val_imgs[:,:,:,indices]
                    val_idx = np.array(val_idx)[indices]
                    val_label = np.array(val_label)[indices,:] 
                    
            # get the medical distribution of datapoints
            elif 'ROCO' in config['experiment_tpye']:

                # number of datapoints
                train_mat = config['train_ratio']*np.ones((num_classes[0],num_classes[1]))
                val_mat =  (1-config['train_ratio'])*np.ones((num_classes[0],num_classes[1]))
                datapoints_train = np.round(train_mat*number_imgs_mat_train_val)
                datapoints_val = np.round(val_mat*number_imgs_mat_train_val)
                # print('Total number of images for train: '+str(datapoints_train.sum()))
                # print('Total number of images for val: '+str(datapoints_val.sum()))
            
                # get only the relevant images
                # train
                check_vec = df['ImgID'] <  datapoints_train[df['Organ'],df['Modality']] 
                train_imgs = imgs[:,:,:,check_vec]
                train_idx = img_idx[check_vec]
                train_label = label_vec[check_vec]
                train_dir = data_dir + curr_phase + '/'
            
                # val
                max_id = number_imgs_mat_train_val.max()
                check_vec = (df['ImgID'] < (max_id*config['train_ratio'] + datapoints_val[df['Organ'],df['Modality']])) & (df['ImgID'] >= (max_id*config['train_ratio']))
                val_imgs = imgs[:,:,:,check_vec]
                val_idx = img_idx[check_vec]
                val_label = label_vec[check_vec]
                val_dir = data_dir + curr_phase + '/'
                
                # check which task to train for secondary setting
                if 'secondary' in config['experiment_tpye']:
            
                    # choose secondary task labels and img_idx
                    indices = np.where(train_label[:,1] == task_to_run_secondary)[0]
                    train_imgs = train_imgs[:,:,:,indices]
                    train_idx = np.array(train_idx)[indices]
                    train_label = np.array(train_label)[indices,:] 
                    indices = np.where(val_label[:,1] == task_to_run_secondary)[0]
                    val_imgs = val_imgs[:,:,:,indices]
                    val_idx = np.array(val_idx)[indices]
                    val_label = np.array(val_label)[indices,:]   
                    
            # get the sampled version of datapoints
            elif 'sampled' in config['experiment_tpye']:
                # number of datapoints
                datapoints_train = (config['sampling_percentage']/100)*config['max_id']*config['train_ratio']*np.ones((num_classes[0],num_classes[1]))
                datapoints_val =  (config['sampling_percentage']/100)*config['max_id']*(1-config['train_ratio'])*np.ones((num_classes[0],num_classes[1]))
                print('Total number of images for train: '+str(datapoints_train.sum()))
                print('Total number of images for val: '+str(datapoints_val.sum()))
                
                if 'upsampled' in config['experiment_tpye']:
                    # total number of datapoints for general model
                    total_num_datapoints_train = datapoints_train.sum()
                    total_num_datapoints_val = datapoints_val.sum()
                    
                    # total number of datapoints for secondary model
                    secondary_total_num_datapoints_train = datapoints_train[:,task_to_run_secondary].sum()
                    secondary_total_num_datapoints_val = datapoints_val[:,task_to_run_secondary].sum()
                    
                    # print
                    print('Train - Number of datapoints for general model: ' + str(total_num_datapoints_train))
                    print('Train - Number of datapoints for specialized model: ' + str(total_num_datapoints_train))
                    
                    # upsample the number of datapoints to match the total for general model
                    datapoints_train = np.round(np.tile(datapoints_train[:,task_to_run_secondary]*(total_num_datapoints_train/secondary_total_num_datapoints_train),(num_classes[1],1)).transpose())
                    datapoints_val = np.round(np.tile(datapoints_val[:,task_to_run_secondary]*(total_num_datapoints_val/secondary_total_num_datapoints_val),(num_classes[1],1)).transpose())
            
                # get only the relevant images
                # train
                check_vec = df['ImgID'] <  datapoints_train[df['Organ'],df['Modality']] 
                train_imgs = imgs[:,:,:,check_vec]
                train_idx = img_idx[check_vec]
                train_label = label_vec[check_vec]
                train_dir = data_dir + curr_phase + '/'
            
                # val
                check_vec = (df['ImgID'] < (datapoints_train.max() + datapoints_val[df['Organ'],df['Modality']])) & (df['ImgID'] >= (datapoints_train.max()))
                val_imgs = imgs[:,:,:,check_vec]
                val_idx = img_idx[check_vec]
                val_label = label_vec[check_vec]
                val_dir = data_dir + curr_phase + '/'
                
                # check which task to train for secondary setting
                if 'secondary' in config['experiment_tpye']:
            
                    # choose secondary task labels and img_idx
                    indices = np.where(train_label[:,1] == task_to_run_secondary)[0]
                    train_imgs = train_imgs[:,:,:,indices]
                    train_idx = np.array(train_idx)[indices]
                    train_label = np.array(train_label)[indices,:] 
                    indices = np.where(val_label[:,1] == task_to_run_secondary)[0]
                    val_imgs = val_imgs[:,:,:,indices]
                    val_idx = np.array(val_idx)[indices]
                    val_label = np.array(val_label)[indices,:]         
            else:     
                # unique tuples
                imgIDs = df['ImgID'].unique()
                
                # check which task to train for secondary setting
                if 'secondary' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
            
                    # choose secondary task labels and img_idx
                    indices = np.where(label_vec[:,1] == task_to_run_secondary)[0]
                    imgs = imgs[:,:,:,indices]
                    img_idx = np.array(img_idx)[indices]
                    label_vec = np.array(label_vec)[indices,:]   
        
                # train validation split from train (with tuple ids - no random split, choose first train_ratio percentage)
                # train_tuple_idx, val_tuple_idx = train_test_split(imgIDs_, test_size=1 - config['train_ratio'], random_state = config['seed'])
                if 'secondary' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
                    idx = int(np.round(len(imgIDs)*config['train_ratio'])*num_classes[0])
                else:
                    idx = int(np.round(len(imgIDs)*config['train_ratio'])*num_classes[0]*num_classes[1])
            
                # train
                train_imgs = imgs[:,:,:,:idx]
                train_idx = img_idx[:idx]
                train_label = label_vec[:idx]
                train_dir = data_dir + curr_phase + '/'
            
                # val
                val_imgs = imgs[:,:,:,idx:]
                val_idx = img_idx[idx:]
                val_label = label_vec[idx:]
                val_dir = data_dir + curr_phase + '/' 
            
            # for reduced setting choose which ones to remove
            if 'reduced' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
                # train
                tmp = np.where(np.sum(train_label == label_vec_reduced,1) == 2)[0]
                idx = random.sample(range(len(tmp)), int(np.round(len(tmp)*config['reduced_percentage']/100)))
                train_imgs = np.delete(train_imgs,tmp[idx],axis=3)
                train_idx =  np.delete(train_idx,tmp[idx])
                train_label =  np.delete(train_label,tmp[idx],axis=0)
                
                # val
                tmp = np.where(np.sum(val_label == label_vec_reduced,1) == 2)[0]
                idx = random.sample(range(len(tmp)), int(np.round(len(tmp)*config['reduced_percentage']/100)))
                val_imgs = np.delete(val_imgs,tmp[idx],axis=3)
                val_idx =  np.delete(val_idx,tmp[idx])
                val_label =  np.delete(val_label,tmp[idx],axis=0)
                
            print(train_idx[0:20])
            print(train_label[0:20])
            print(val_idx[0:20])
            print(val_label[0:20])    
                 
            # get only task relevant labels    
            train_label = train_label[:,task_to_run]   
            val_label = val_label[:,task_to_run]
            
            # # print
            # print(train_idx[0:10])
            # print(train_label[0:10])
            # print(val_idx[0:10])
            # print(val_label[0:10])
            
            # for control expperiments
            if 'control' in config['experiment_tpye']:
                # iterate over labels to map 5->0, 6->1, 7->2, 8->3, 9->4
                for curr_label in range(5,10):
                    train_label[train_label == curr_label] = curr_label-5
                    val_label[val_label == curr_label] = curr_label-5
                    
                # change number of classes for tasks
                CLASS_NAMES[0] = range(0,5)
                num_classes[0] = len(CLASS_NAMES[0])
                num_classes = num_classes.astype(int)
                num_classes_total = int(sum(num_classes))  
        
        # use validation set as test set
        elif curr_phase == 'test':
            if 'ROCO' in config['experiment_tpye']:
                # get only the relevant images
                check_vec = df['ImgID'] <  number_imgs_mat_test[df['Organ'],df['Modality']] 
                imgs = imgs[:,:,:,check_vec]
                img_idx = img_idx[check_vec]
                label_vec = label_vec[check_vec]
            
            # check which task to train for secondary setting
            if 'secondary' in config['experiment_tpye'] or 'control' in config['experiment_tpye']:
                # choose secondary task labels and img_idx to test
                if 'secondary_test' in config['experiment_tpye']:
                    indices = np.where(label_vec[:,1] == config['id_task_to_test_secondary'])[0]
                    imgs = imgs[:,:,:,indices]
                    img_idx = np.array(img_idx)[indices]
                    label_vec = np.array(label_vec)[indices,:]  
                else: 
                    indices = np.where(label_vec[:,1] == task_to_run_secondary)[0]
                    imgs = imgs[:,:,:,indices]
                    img_idx = np.array(img_idx)[indices]
                    label_vec = np.array(label_vec)[indices,:]    
               
            test_imgs = imgs
            test_idx = img_idx
            test_label = label_vec
            test_dir = data_dir + curr_phase  + '/'
            
            # print
            print(test_idx[0:10])
            print(test_label[0:10])
            
            # for control expperiments
            if config['experiment_tpye'] == 'control':
                test_label = np.concatenate((test_label,np.transpose([test_label[:,task_to_run]])),axis=1)
                # iterate over labels to map 5->0, 6->1, 7->2, 8->3, 9->4
                for curr_label in range(5,10):
                    test_label[test_label[:,task_to_run] == curr_label,task_to_run] = curr_label-5
            print('Number of images to use for test: ' + str(len(test_idx)))
            
            class_count = np.zeros(num_classes[task_to_run])
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(test_label[:,task_to_run]==i))
                print('Number of images of ' + str(CLASS_NAMES[task_to_run][i]) + ': ' + str(sum_check))  
        
        # checks
        if curr_phase == 'train':
            class_count = np.zeros(num_classes[task_to_run])
            print('Number of images to use for train: ' + str(len(train_idx)))
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(train_label==i))
                print('Number of images of ' + str(CLASS_NAMES[task_to_run][i]) + ': ' + str(sum_check))  
            class_count = np.zeros(num_classes[task_to_run])
            print('Number of images to use for val: ' + str(len(val_idx)))    
            for i in range(num_classes[task_to_run]):
                sum_check = int(sum(val_label==i))
                print('Number of images of ' + str(CLASS_NAMES[task_to_run][i]) + ': ' + str(sum_check))  
    
    return train_imgs, train_idx, train_label, train_dir, val_imgs, val_idx, val_label, val_dir, test_imgs, test_idx, test_label, test_dir 

def getImgsLabels_medmnist(CLASS_NAMES, CLASS_NAMES_exp, task, task_to_run, data_dir, config, task1_reduced, task2_reduced):
    
    """
    Function to prepare the labels. 
    Args:
        CLASS_NAMES: list of the classes (with concept id)
        CLASS_NAMES_exp: list of the classes with their concept explanations
        task: list of tasks
        task_to_run: id of the task to run (to choose from task list)
        data_dir: data directory
        config: configuration parameters
        task1_reduced: removed task 1 (row) from the task matrix (for experiment_type = 'reduced')
        task2_reduced: removed task 2 (column) from the task matrix (for experiment_type = 'reduced')
        
    Returns:
        train_idx: ids of the images for train
        train_label: labels of the images for train
        train_dir: directory of the images for train
        val_idx: ids of the images for val
        val_label: labels of the images for train
        val_dir: directory of the images for train
        test_idx: ids of the images for test
        test_label: labels of the images for train
        test_dir : directory of the images for train
    """
    # total number of classes for tasks
    num_classes = np.zeros(len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        num_classes[i] = len(CLASS_NAMES[i])
    num_classes = num_classes.astype(int)
    num_classes_total = int(sum(num_classes))
    
    # get the label vec of reduced setting
    if 'reduced' in config['experiment_tpye']:
        label_vec_reduced = [task1_reduced,task2_reduced]
    
    # check if it is secondary setting
    if 'secondary' in config['experiment_tpye']:
        print('Secondary task is selected.')
        task_to_run_secondary = config['id_task_to_run_secondary']
        secondary_task = CLASS_NAMES[1][task_to_run_secondary]
        secondary_task_exp = CLASS_NAMES_exp[1][task_to_run_secondary]
        # print('All secondary classes:' + CLASS_NAMES_exp[1])
        print('Secondary task to run: ' + secondary_task_exp)
    
    # prepare data
    phase = ['train', 'val', 'test']

    # iterate over phase:
    for curr_phase in phase:
        
        # initialize imgs and label_vec
        print('Annotations loading..')
        imgs = np.empty((0,28,28))
        label_vec = np.empty((0,2))
    
        # initialize array for task matrix
        if len(CLASS_NAMES) != 1:
            task_mat = np.zeros([num_classes[0],num_classes[1]])
    
        # iterate over views 
        cnt = 0
        for i in ['a','c','s']:
            data_path = data_dir+'organ'+i+'mnist.npz'
            with np.load(data_path) as data:
                imgs = np.concatenate((imgs,data[curr_phase+'_images']),axis=0)
                labels = data[curr_phase+'_labels']
                labels = np.concatenate((labels,cnt*np.ones((len(labels),1))),axis=1)
                label_vec = np.concatenate((label_vec,labels),axis=0)
                cnt = cnt + 1  
        print('Annotations loaded!')
            
        # check if distribution experiment
        if ('dist' in config['experiment_tpye']) & (curr_phase == 'train'):
            print('Preparing distribution..')
            # iterate over each combination
            for i in range(len(CLASS_NAMES[0])):
                for j in range(len(CLASS_NAMES[1])):
                    curr_label_vec = [i,j]
                    tmp = np.where(np.sum(label_vec == curr_label_vec,1) == 2)[0]
                    # remove entries more than max sample size
                    if len(tmp) > config['max_sample_train']:
                        idx = random.sample(range(len(tmp)), len(tmp)-config['max_sample_train'])
                        imgs = np.delete(imgs,tmp[idx],axis=0)
                        label_vec =  np.delete(label_vec,tmp[idx],axis=0)    
            print('Distribution done for training..')              
            
        # check if distribution experiment
        if ('dist' in config['experiment_tpye']) & (curr_phase == 'val'):
            print('Preparing distribution..')
            # iterate over each combination
            for i in range(len(CLASS_NAMES[0])):
                for j in range(len(CLASS_NAMES[1])):
                    curr_label_vec = [i,j]
                    tmp = np.where(np.sum(label_vec == curr_label_vec,1) == 2)[0]
                    # remove entries more than max sample size
                    if len(tmp) > config['max_sample_val']:
                        idx = random.sample(range(len(tmp)), len(tmp)-config['max_sample_val'])
                        imgs = np.delete(imgs,tmp[idx],axis=0)
                        label_vec =  np.delete(label_vec,tmp[idx],axis=0)    
            print('Distribution done for validation..')                  
                
        # check the task matrix
        if (len(CLASS_NAMES) != 1):   
            label_vec_init = label_vec[:,0]
            label_vec_sec = label_vec[:,1]
            # iterate over tasks
            for task_init in range(num_classes[0]):
                curr_label_vec_sec = label_vec_sec[label_vec_init == task_init]
                for task_sec in range(num_classes[1]):
                    curr_cnt = sum(curr_label_vec_sec == task_sec)
                    task_mat[task_init,task_sec] = curr_cnt
            print(task_mat)        
            
        else:
            print('Second task not defined!')
            continue        
        
        # check which task to train
        label_vec = label_vec.astype(int)
        if 'secondary' in config['experiment_tpye']:
            # choose secondary task labels and img_idx
            indices = np.where(label_vec[:,1] == task_to_run_secondary)[0]
            imgs = np.array(imgs)[indices]
            label_vec = np.array(label_vec)[indices,:] 
            
        # sample the data using train_test_split function
        if (curr_phase != 'test') & (config['sampling_percentage'] < 100):
            imgs, _, label_vec, _ = train_test_split(imgs, label_vec, test_size=1-(config['sampling_percentage']/100), random_state = config['seed'])
        print(imgs.shape)    

        # print
        print('Number of images total after sampling: '+ str(len(imgs)))
    
        if curr_phase == 'train' or curr_phase == 'val':
            if 'reduced' in config['experiment_tpye']:
                # train
                tmp = np.where(np.sum(label_vec == label_vec_reduced,1) == 2)[0]
                idx = random.sample(range(len(tmp)), int(np.round(len(tmp)*config['reduced_percentage']/100)))
                imgs =  np.delete(imgs,tmp[idx],axis=0)
                label_vec =  np.delete(label_vec,tmp[idx],axis=0)
                
        # print
        print('Number of images after reducing: '+ str(len(imgs)))    
        
        class_count = np.zeros(num_classes[task_to_run])
        for i in range(num_classes[task_to_run]):
            sum_check = int(sum(label_vec[:,task_to_run]==i))
            print('Number of images of ' + CLASS_NAMES_exp[task_to_run][i] + ': ' + str(sum_check)) 
            
        # assign    
        if curr_phase == 'train':    
            train_imgs = imgs
            train_label = label_vec
            print('Number of images to use for train: ' + str(len(train_imgs)))
        elif curr_phase == 'val':    
            val_imgs = imgs
            val_label = label_vec  
            print('Number of images to use for val: ' + str(len(val_imgs)))    
        elif curr_phase == 'test':   
            test_imgs = imgs
            test_label = label_vec 
            print('Number of images to use for test: ' + str(len(test_imgs))) 
        else:
            print('Phase not defined!')
        
    # get only task relevant labels    
    train_label = train_label[:,task_to_run]   
    val_label = val_label[:,task_to_run]
    
    return train_imgs, train_label, val_imgs, val_label, test_imgs, test_label     
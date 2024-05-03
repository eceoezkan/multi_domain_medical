import json
import os

def config_experiments_hyper(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # seed experiments
    for lr in [5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]:
        for batch_size in [128,256,512]:
                     
            # initialize config 
            config = {}
                            
            # assign config parameters
            config['model_type'] = 'ViTb16'
            config['experiment_id'] = str(id)
            config['model_pretrained'] = True
            config['batch_size'] = batch_size
            config['learning_rate'] = lr
            config['seed'] = 42
            config['criterion_type'] = 'CELoss'
            config['num_epochs'] = 25
            config['img_resize'] = 100
            config['train_ratio'] = 0.75
            config['weight_decay'] = 0.001
            config['step_size'] = 5
            config['experiment_tpye'] = 'organ'
            config['sampling_percentage'] = 100
                            
            if create_json:
                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                    json.dump(config, json_file)
            experiment_list.append(config.copy())
            id += 1
                            
    print(str(id) + " config files created for seed experiments")   
    
    return experiment_list

def config_experiments(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # seed experiments
    for model_type in ['ResNet18','ResNet50','DenseNet121']: # ,'ViTb16']:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
                     
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = model_type
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 128
                if model_type != 'ViTb16':
                        config['learning_rate'] = 0.001
                        config['img_resize'] = 32
                else:
                        config['learning_rate'] = 5e-5
                        config['img_resize'] = 100
                config['seed'] = seed
                config['criterion_type'] = 'CELoss'
                config['num_epochs'] = 25
                config['train_ratio'] = 0.75
                config['weight_decay'] = 0.001
                config['step_size'] = 5
                config['experiment_tpye'] = 'organ'
                config['sampling_percentage'] = sampling_percentage
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1
                            
    print(str(id) + " config files created for seed experiments")   
    
    return experiment_list

def config_experiments_dist(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # seed experiments
    for model_type in ['ResNet18','DenseNet121']:
        for sampling_percentage in [100]: # ,75,50,35,25,10,5]:
            for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
                     
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = model_type
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 128
                config['img_resize'] = 32
                config['learning_rate'] = 5e-5
                config['img_resize'] = 100
                config['seed'] = seed
                config['criterion_type'] = 'CELoss'
                config['num_epochs'] = 25
                config['train_ratio'] = 0.75
                config['weight_decay'] = 0.001
                config['step_size'] = 5
                config['experiment_tpye'] = 'organ_dist'
                config['sampling_percentage'] = sampling_percentage
                config['max_sample'] = 1000
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1
                
    # seed experiments
    for model_type in ['ResNet18']:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            for seed in [42,73,666,777,1009]: # ,1279,1597,1811,1949,2053]:
                     
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = model_type
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 128
                config['img_resize'] = 32
                config['learning_rate'] = 5e-5
                config['img_resize'] = 100
                config['seed'] = seed
                config['criterion_type'] = 'CELoss'
                config['num_epochs'] = 25
                config['train_ratio'] = 0.75
                config['weight_decay'] = 0.001
                config['step_size'] = 5
                config['experiment_tpye'] = 'organ_dist'
                config['sampling_percentage'] = sampling_percentage
                config['max_sample_train'] = 600
                config['max_sample_val'] = 95
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1
                            
    print(str(id) + " config files created for dist experiments")   
    
    return experiment_list

def config_experiments_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for model_type in ['ResNet18','ResNet50','DenseNet121']: # ,'ViTb16']:
        for modality in range(0,3):
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
                    # initialize config 
                    config = {}
                            
                    # assign config parameters
                    config['model_type'] = model_type
                    config['experiment_id'] = str(id)
                    config['model_pretrained'] = True
                    config['batch_size'] = 128
                    if model_type != 'ViTb16':
                        config['learning_rate'] = 0.001
                        config['img_resize'] = 32
                    else:
                        config['learning_rate'] = 5e-5
                        config['img_resize'] = 100
                    config['seed'] = seed
                    config['criterion_type'] = 'CELoss'
                    config['num_epochs'] = 25
                    config['train_ratio'] = 0.75
                    config['weight_decay'] = 0.001
                    config['step_size'] = 5
                    config['experiment_tpye'] = 'secondary'
                    config['sampling_percentage'] = sampling_percentage
                    config['id_task_to_run_secondary'] = modality
                            
                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1

    print(str(id) + " config files created for secondary experiments")
    
    return experiment_list

def config_experiments_secondary_dist(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for model_type in ['ResNet18','DenseNet121']:
        for modality in range(0,3):
            for sampling_percentage in [100]: # ,75,50,35,25]:
                for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
                    # initialize config 
                    config = {}
                            
                    # assign config parameters
                    config['model_type'] = model_type
                    config['experiment_id'] = str(id)
                    config['model_pretrained'] = True
                    config['batch_size'] = 128
                    config['learning_rate'] = 0.001
                    config['img_resize'] = 32
                    config['seed'] = seed
                    config['criterion_type'] = 'CELoss'
                    config['num_epochs'] = 25
                    config['train_ratio'] = 0.75
                    config['weight_decay'] = 0.001
                    config['step_size'] = 5
                    config['experiment_tpye'] = 'secondary_dist'
                    config['sampling_percentage'] = sampling_percentage
                    config['id_task_to_run_secondary'] = modality
                    config['max_sample'] = 1000
                            
                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1
                    
    # secondary experiments 
    for model_type in ['ResNet18']:
        for modality in range(0,3):
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for seed in [42,73,666,777,1009]: # ,1279,1597,1811,1949,2053]:
                    # initialize config 
                    config = {}
                            
                    # assign config parameters
                    config['model_type'] = model_type
                    config['experiment_id'] = str(id)
                    config['model_pretrained'] = True
                    config['batch_size'] = 128
                    config['learning_rate'] = 0.001
                    config['img_resize'] = 32
                    config['seed'] = seed
                    config['criterion_type'] = 'CELoss'
                    config['num_epochs'] = 25
                    config['train_ratio'] = 0.75
                    config['weight_decay'] = 0.001
                    config['step_size'] = 5
                    config['experiment_tpye'] = 'secondary_dist'
                    config['sampling_percentage'] = sampling_percentage
                    config['id_task_to_run_secondary'] = modality
                    config['max_sample_train'] = 600
                    config['max_sample_val'] = 95
                            
                    if create_json:
                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                            json.dump(config, json_file)
                    experiment_list.append(config.copy())
                    id += 1                

    print(str(id) + " config files created for dist secondary experiments")
    
    return experiment_list

def config_experiments_reduced(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # seed experiments
    for seed in [42]: # ,73,666]: # ,777,1009,1279,1597,1811,1949,2053]:
        for model_type in ['ResNet18','DenseNet121']: # ,'ViTb16']:
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for reduced_percentage in [25,50,75,85,95,100]:
                    for organ in range(0,11):
                        for modality in range(0,3):
                   
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 128
                            config['learning_rate'] = 0.001
                            config['img_resize'] = 32
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['num_epochs'] = 25
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['task1'] = organ
                            config['task2'] = modality
                            config['experiment_tpye'] = 'reduced'
                            config['reduced_percentage'] = reduced_percentage
                            config['sampling_percentage'] = sampling_percentage
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1
                            
    # seed experiments
    for seed in [73,666,777,1009]: # ,1279,1597,1811,1949,2053]:
        for model_type in ['ResNet18']: # ,'DenseNet121','ViTb16']:
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for reduced_percentage in [25,50,75,85,95,100]:
                    for organ in range(0,11):
                        for modality in range(0,3):
                   
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 128
                            config['learning_rate'] = 0.001
                            config['img_resize'] = 32
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['num_epochs'] = 25
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['task1'] = organ
                            config['task2'] = modality
                            config['experiment_tpye'] = 'reduced'
                            config['reduced_percentage'] = reduced_percentage
                            config['sampling_percentage'] = sampling_percentage
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1                        
                            
    print(str(id) + " config files created for reduced experiments")   
    
    return experiment_list

def config_experiments_reduced_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # seed experiments
    for seed in [42]: # ,73,666]: # ,777,1009,1279,1597,1811,1949,2053]:
        for model_type in ['ResNet18','DenseNet121']: # ,'ViTb16']:
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for reduced_percentage in [25,50,75,85,95,100]:
                    for organ in range(0,11):
                        for modality in range(0,3):
                   
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 128
                            config['learning_rate'] = 0.001
                            config['img_resize'] = 32
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['num_epochs'] = 25
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['task1'] = organ
                            config['task2'] = modality
                            config['id_task_to_run_secondary'] = modality
                            config['experiment_tpye'] = 'reduced_secondary'
                            config['reduced_percentage'] = reduced_percentage
                            config['sampling_percentage'] = sampling_percentage
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1
                            
    # seed experiments
    for seed in [73,666,777,1009]: # ,1279,1597,1811,1949,2053]:
        for model_type in ['ResNet18']: # ,'DenseNet121','ViTb16']:
            for sampling_percentage in [100,75,50,35,25,10,5]:
                for reduced_percentage in [25,50,75,85,95,100]:
                    for organ in range(0,11):
                        for modality in range(0,3):
                   
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 128
                            config['learning_rate'] = 0.001
                            config['img_resize'] = 32
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['num_epochs'] = 25
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['task1'] = organ
                            config['task2'] = modality
                            config['id_task_to_run_secondary'] = modality
                            config['experiment_tpye'] = 'reduced_secondary'
                            config['reduced_percentage'] = reduced_percentage
                            config['sampling_percentage'] = sampling_percentage
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1                        
                            
    print(str(id) + " config files created for reduced secondary experiments")   
    
    return experiment_list
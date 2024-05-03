import json
import os


def config_experiments(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # hyperparameter experiments
    for model_type in ['ResNet18','ResNet34','ResNet50']:
        for model_pretrained in [True,False]:
            for batch_size in [128]:
                for lr in [1, 1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6]:
                    for seed in [42]:
                        for sample_class in range(1,4):
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = model_pretrained
                            config['batch_size'] = batch_size
                            config['learning_rate'] = lr
                            config['seed'] = seed
                            config['sample_class'] = sample_class
                            config['criterion_type'] = 'CELoss'
                            config['num_concepts_to_work'] = 100
                            config['num_epochs'] = 50
                            config['img_resize'] = 224
                            config['img_pad'] = 24
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 1e-5
                            config['step_size'] = 5
                            config['experiment_tpye'] = 'hyperparameter'
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1
                            
    print(str(id) + " config files created for hyperparameter tuning")   
    
    return experiment_list

def config_experiments_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced experiments 
    # in top 100 concepts there are 26 organs and 6 modalities
    for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            for modality in range(0,6):
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = 'ResNet18'
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 128
                config['learning_rate'] = 5e-4
                config['seed'] = seed
                config['sample_class'] = 3
                config['criterion_type'] = 'CELoss'
                config['num_concepts_to_work'] = 100
                config['num_epochs'] = 25
                config['img_resize'] = 224
                config['img_pad'] = 24
                config['train_ratio'] = 0.75
                config['weight_decay'] = 1e-5
                config['step_size'] = 5
                config['id_task_to_run_secondary'] = modality
                config['experiment_tpye'] = 'secondary'
                config['sampling_percentage'] = sampling_percentage
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    print(str(id) + " config files created for secondary experiments")
    
    return experiment_list

def config_experiments_reduced(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced experiments 
    # in top 100 concepts there are 26 organs and 6 modalities
    for seed in [42,73,666,777,1009]:# ,1279,1597,1811,1949,2053]:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            for reduced_percentage in [25,50,75,85,95,100]:
                for organ in range(1,26,3):
                    for modality in range(0,6):
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 128
                        config['learning_rate'] = 5e-4
                        config['seed'] = seed
                        config['sample_class'] = 3
                        config['criterion_type'] = 'CELoss'
                        config['num_concepts_to_work'] = 100
                        config['num_epochs'] = 25
                        config['img_resize'] = 224
                        config['img_pad'] = 24
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 1e-5
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
    
    # reduced experiments 
    # in top 100 concepts there are 26 organs and 6 modalities
    for seed in [42,73,666,777,1009]: #,1279,1597,1811,1949,2053]:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            for reduced_percentage in [25,50,75,85,95,100]:
                for organ in range(1,26,3):
                    for modality in range(0,6):
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 128
                        config['learning_rate'] = 5e-4
                        config['seed'] = seed
                        config['sample_class'] = 3
                        config['criterion_type'] = 'CELoss'
                        config['num_concepts_to_work'] = 100
                        config['num_epochs'] = 25
                        config['img_resize'] = 224
                        config['img_pad'] = 24
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 1e-5
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

def config_experiments_seeds(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # hyperparameter experiments
    for seed in [42,73,666,777,1009,1279,1597,1811,1949,2053]:
        for sampling_percentage in [100,75,50,35,25,10,5]:
            # initialize config 
            config = {}
                            
            # assign config parameters
            config['model_type'] = 'ResNet18'
            config['experiment_id'] = str(id)
            config['model_pretrained'] = True
            config['batch_size'] = 128
            config['learning_rate'] = 5e-4
            config['seed'] = seed
            config['sample_class'] = 3
            config['criterion_type'] = 'CELoss'
            config['num_concepts_to_work'] = 100
            config['num_epochs'] = 25
            config['img_resize'] = 224
            config['img_pad'] = 24
            config['train_ratio'] = 0.75
            config['weight_decay'] = 1e-5
            config['step_size'] = 5
            config['experiment_tpye'] = 'seed'
            config['sampling_percentage'] = sampling_percentage
                            
            if create_json:
                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                    json.dump(config, json_file)
            experiment_list.append(config.copy())
            id += 1
                            
    print(str(id) + " config files created for seed experiments")   
    
    return experiment_list

def config_experiments_hyper_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # hyperparameter experiments
    for lr in [1, 1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6]:
        for modality in range(0,6):
            # initialize config 
            config = {}
                            
            # assign config parameters
            config['model_type'] = 'ResNet18'
            config['experiment_id'] = str(id)
            config['model_pretrained'] = True
            config['batch_size'] = 128
            config['learning_rate'] = lr
            config['seed'] = 42
            config['sample_class'] = 3
            config['criterion_type'] = 'CELoss'
            config['num_concepts_to_work'] = 100
            config['num_epochs'] = 25
            config['img_resize'] = 224
            config['img_pad'] = 24
            config['train_ratio'] = 0.75
            config['weight_decay'] = 1e-5
            config['step_size'] = 5
            config['id_task_to_run_secondary'] = modality
            config['experiment_tpye'] = 'secondary_hyper'
            config['sampling_percentage'] = 100
                            
            if create_json:
                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                    json.dump(config, json_file)
            experiment_list.append(config.copy())
            id += 1
                            
    print(str(id) + " config files created for hyperparameter tuning for secondary experiments")   
    
    return experiment_list
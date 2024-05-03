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
        for model_pretrained in [True]:
            for max_id in [1000,2000]:
                for lr in [1, 1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6]:
                    for seed in [42]:
                        for weight_decay in [1e-3, 1e-4, 1e-5, 1e-6]:
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = model_type
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = model_pretrained
                            config['batch_size'] = 512
                            config['learning_rate'] = lr
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['max_id'] = max_id
                            config['num_epochs'] = 25
                            config['img_resize'] = 32
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = weight_decay
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
    
    # secondary experiments 
    for modality in range(0,5):
        for max_id in [1000]:
            for lr in [0.005]:
                for seed in [42]:
                    for weight_decay in [0.001]:
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 512
                        config['learning_rate'] = lr
                        config['seed'] = seed
                        config['criterion_type'] = 'CELoss'
                        config['max_id'] = max_id
                        config['num_epochs'] = 25
                        config['img_resize'] = 32
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = weight_decay
                        config['step_size'] = 5
                        config['id_task_to_run_secondary'] = modality
                        config['experiment_tpye'] = 'secondary'
                            
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
    # there are 10 organs and 5 modalities
    for reduced_percentage in [5,10,20,50,75,100,80,85,90,95]:
        for organ in range(0,10):
            for modality in range(0,5):
                for max_id in [1000]:
                    for lr in [0.005]:
                        for seed in [42]:
                            for weight_decay in [0.001]:
                                # initialize config 
                                config = {}
                            
                                # assign config parameters
                                config['model_type'] = 'ResNet18'
                                config['experiment_id'] = str(id)
                                config['model_pretrained'] = True
                                config['batch_size'] = 512
                                config['learning_rate'] = lr
                                config['seed'] = seed
                                config['criterion_type'] = 'CELoss'
                                config['max_id'] = max_id
                                config['num_epochs'] = 25
                                config['img_resize'] = 32
                                config['train_ratio'] = 0.75
                                config['weight_decay'] = weight_decay
                                config['step_size'] = 5
                                config['task1'] = organ
                                config['task2'] = modality
                                config['experiment_tpye'] = 'reduced'
                                config['reduced_percentage'] = reduced_percentage
                            
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
    # there are 10 organs and 5 modalities
    for reduced_percentage in [5,10,20,50,75,100,80,85,90,95]:
        for organ in range(0,10):
            for modality in range(0,5):
                for max_id in [1000]:
                    for lr in [0.005]:
                        for seed in [42]:
                            for weight_decay in [0.001]:
                                # initialize config 
                                config = {}
                            
                                # assign config parameters
                                config['model_type'] = 'ResNet18'
                                config['experiment_id'] = str(id)
                                config['model_pretrained'] = True
                                config['batch_size'] = 512
                                config['learning_rate'] = lr
                                config['seed'] = seed
                                config['criterion_type'] = 'CELoss'
                                config['max_id'] = max_id
                                config['num_epochs'] = 25
                                config['img_resize'] = 32
                                config['train_ratio'] = 0.75
                                config['weight_decay'] = weight_decay
                                config['step_size'] = 5
                                config['task1'] = organ
                                config['task2'] = modality
                                config['id_task_to_run_secondary'] = modality
                                config['experiment_tpye'] = 'reduced_secondary'
                                config['reduced_percentage'] = reduced_percentage
                            
                                if create_json:
                                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                        json.dump(config, json_file)
                                experiment_list.append(config.copy())
                                id += 1

    print(str(id) + " config files created for reduced secondary experiments")
        
    
    return experiment_list

def config_experiments_control(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # control experiments 
    for reduced_percentage in [5,10,20,50,75,80,85,90,95,100]:
        for organ in range(0,10):
            for modality in range(0,5):
                for max_id in [1000]:
                    for lr in [0.005]:
                        for seed in [42]:
                            for weight_decay in [0.001]:
                                # initialize config 
                                config = {}
                            
                                # assign config parameters
                                config['model_type'] = 'ResNet18'
                                config['experiment_id'] = str(id)
                                config['model_pretrained'] = True
                                config['batch_size'] = 512
                                config['learning_rate'] = lr
                                config['seed'] = seed
                                config['criterion_type'] = 'CELoss'
                                config['max_id'] = max_id
                                config['num_epochs'] = 25
                                config['img_resize'] = 32
                                config['train_ratio'] = 0.75
                                config['weight_decay'] = weight_decay
                                config['step_size'] = 5
                                config['task1'] = organ
                                config['task2'] = modality
                                config['id_task_to_run_secondary'] = modality
                                config['experiment_tpye'] = 'control'
                                config['reduced_percentage'] = reduced_percentage
                            
                                if create_json:
                                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                        json.dump(config, json_file)
                                experiment_list.append(config.copy())
                                id += 1

    print(str(id) + " config files created for control experiments")
        
    return experiment_list

def config_experiments_nbrData(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # number of datapoints experiments
    for mean_organ in [0]:
        for std_organ in [3,5,9,17]:
            for mean_modality in [0,2]:
                for std_modality in [1,3,5]:
                    for seed in [42]:
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 512
                        config['learning_rate'] = 0.005
                        config['seed'] = seed
                        config['criterion_type'] = 'CELoss'
                        config['max_id'] = 1000
                        config['num_epochs'] = 25
                        config['img_resize'] = 32
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 0.001
                        config['step_size'] = 5
                        config['experiment_tpye'] = 'nbrData'
                        config['mean_organ'] = mean_organ
                        config['std_organ'] = std_organ
                        config['mean_modality'] = mean_modality
                        config['std_modality'] = std_modality
                            
                        if create_json:
                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                json.dump(config, json_file)
                        experiment_list.append(config.copy())
                        id += 1
                            
    print(str(id) + " config files created for nbrData experiments")   
    
    return experiment_list

def config_experiments_nbrData_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for modality in range(0,5):
        for mean_organ in [0]:
            for std_organ in [3,5,9,17]:
                for mean_modality in [0,2]:
                    for std_modality in [1,3,5]:
                        for seed in [42]:
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = 'ResNet18'
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 512
                            config['learning_rate'] = 0.005
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['max_id'] = 1000
                            config['num_epochs'] = 25
                            config['img_resize'] = 32
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['id_task_to_run_secondary'] = modality
                            config['experiment_tpye'] = 'nbrData_secondary'
                            config['mean_organ'] = mean_organ
                            config['std_organ'] = std_organ
                            config['mean_modality'] = mean_modality
                            config['std_modality'] = std_modality
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1

    print(str(id) + " config files created for nbrData secondary experiments")
        
    return experiment_list

def config_experiments_nbrData_secondary_upsampled(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for modality in range(0,5):
        for mean_organ in [0]:
            for std_organ in [3,5,9,17]:
                for mean_modality in [0,2]:
                    for std_modality in [1,3,5]:
                        for seed in [42]:
                            # initialize config 
                            config = {}
                            
                            # assign config parameters
                            config['model_type'] = 'ResNet18'
                            config['experiment_id'] = str(id)
                            config['model_pretrained'] = True
                            config['batch_size'] = 512
                            config['learning_rate'] = 0.005
                            config['seed'] = seed
                            config['criterion_type'] = 'CELoss'
                            config['max_id'] = 1000
                            config['num_epochs'] = 25
                            config['img_resize'] = 32
                            config['train_ratio'] = 0.75
                            config['weight_decay'] = 0.001
                            config['step_size'] = 5
                            config['id_task_to_run_secondary'] = modality
                            config['experiment_tpye'] = 'nbrData_secondary_upsampled'
                            config['mean_organ'] = mean_organ
                            config['std_organ'] = std_organ
                            config['mean_modality'] = mean_modality
                            config['std_modality'] = std_modality
                            
                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1

    print(str(id) + " config files created for nbrData secondary upsampled experiments")
        
    return experiment_list

def config_experiments_nbrData_reduced(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced experiments 
    for reduced_percentage in [25,50,75,100,85,95]:
        for mean_organ in [0]:
            for std_organ in [3,5,9,17]:
                for mean_modality in [0,2]:
                    for std_modality in [1,3,5]:
                        for organ in range(0,10):
                            for modality in range(0,5):
                                for seed in [42]:
                                    # initialize config 
                                    config = {}
                            
                                    # assign config parameters
                                    config['model_type'] = 'ResNet18'
                                    config['experiment_id'] = str(id)
                                    config['model_pretrained'] = True
                                    config['batch_size'] = 512
                                    config['learning_rate'] = 0.005
                                    config['seed'] = seed
                                    config['criterion_type'] = 'CELoss'
                                    config['max_id'] = 1000
                                    config['num_epochs'] = 25
                                    config['img_resize'] = 32
                                    config['train_ratio'] = 0.75
                                    config['weight_decay'] = 0.001
                                    config['step_size'] = 5
                                    config['task1'] = organ
                                    config['task2'] = modality
                                    config['experiment_tpye'] = 'nbrData_reduced'
                                    config['reduced_percentage'] = reduced_percentage
                                    config['mean_organ'] = mean_organ
                                    config['std_organ'] = std_organ
                                    config['mean_modality'] = mean_modality
                                    config['std_modality'] = std_modality
                            
                                    if create_json:
                                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                            json.dump(config, json_file)
                                    experiment_list.append(config.copy())
                                    id += 1

    print(str(id) + " config files created for nbrData reduced experiments")
        
    return experiment_list

def config_experiments_nbrData_reduced_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced secondary experiments 
    for reduced_percentage in [25,50,75,100,85,95]:
        for mean_organ in [0]:
            for std_organ in [3,5,9,17]:
                for mean_modality in [0,2]:
                    for std_modality in [1,3,5]:
                        for organ in range(0,10):
                            for modality in range(0,5):
                                for seed in [42]:
                                    # initialize config 
                                    config = {}
                            
                                    # assign config parameters
                                    config['model_type'] = 'ResNet18'
                                    config['experiment_id'] = str(id)
                                    config['model_pretrained'] = True
                                    config['batch_size'] = 512
                                    config['learning_rate'] = 0.005
                                    config['seed'] = seed
                                    config['criterion_type'] = 'CELoss'
                                    config['max_id'] = 1000
                                    config['num_epochs'] = 25
                                    config['img_resize'] = 32
                                    config['train_ratio'] = 0.75
                                    config['weight_decay'] = 0.001
                                    config['step_size'] = 5
                                    config['task1'] = organ
                                    config['task2'] = modality
                                    config['id_task_to_run_secondary'] = modality
                                    config['experiment_tpye'] = 'nbrData_reduced_secondary'
                                    config['reduced_percentage'] = reduced_percentage
                                    config['mean_organ'] = mean_organ
                                    config['std_organ'] = std_organ
                                    config['mean_modality'] = mean_modality
                                    config['std_modality'] = std_modality
                            
                                    if create_json:
                                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                            json.dump(config, json_file)
                                    experiment_list.append(config.copy())
                                    id += 1

    print(str(id) + " config files created for nbrData reduced secondary experiments")
        
    return experiment_list

def config_experiments_nbrData_reduced_secondary_test(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced secondary experiments 
    for reduced_percentage in [25,50,75,100,85,95]:
        for mean_organ in [0]:
            for std_organ in [5]:
                for mean_modality in [2]:
                    for std_modality in [3]:
                        for organ in [2]:
                            for modality in [3]:
                                for seed in [42]:
                                    for modality_test in range(0,5):
                                        # initialize config 
                                        config = {}
                            
                                        # assign config parameters
                                        config['model_type'] = 'ResNet18'
                                        config['experiment_id'] = str(id)
                                        config['model_pretrained'] = True
                                        config['batch_size'] = 512
                                        config['learning_rate'] = 0.005
                                        config['seed'] = seed
                                        config['criterion_type'] = 'CELoss'
                                        config['max_id'] = 1000
                                        config['num_epochs'] = 25
                                        config['img_resize'] = 32
                                        config['train_ratio'] = 0.75
                                        config['weight_decay'] = 0.001
                                        config['step_size'] = 5
                                        config['task1'] = organ
                                        config['task2'] = modality
                                        config['id_task_to_run_secondary'] = modality
                                        config['id_task_to_test_secondary'] = modality_test
                                        config['experiment_tpye'] = 'nbrData_reduced_secondary_test'
                                        config['reduced_percentage'] = reduced_percentage
                                        config['mean_organ'] = mean_organ
                                        config['std_organ'] = std_organ
                                        config['mean_modality'] = mean_modality
                                        config['std_modality'] = std_modality
                            
                                        if create_json:
                                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                                json.dump(config, json_file)
                                        experiment_list.append(config.copy())
                                        id += 1

    print(str(id) + " config files created for nbrData reduced secondary experiments")
        
    return experiment_list

def config_experiments_nbrData_reduced_secondary_upsampled(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced secondary experiments 
    for reduced_percentage in [25,50,75,100,85,95]:
        for mean_organ in [0]:
            for std_organ in [3,5,9,17]:
                for mean_modality in [0,2]:
                    for std_modality in [1,3,5]:
                        for organ in range(0,10):
                            for modality in range(0,5):
                                for seed in [42]:
                                    # initialize config 
                                    config = {}
                            
                                    # assign config parameters
                                    config['model_type'] = 'ResNet18'
                                    config['experiment_id'] = str(id)
                                    config['model_pretrained'] = True
                                    config['batch_size'] = 512
                                    config['learning_rate'] = 0.005
                                    config['seed'] = seed
                                    config['criterion_type'] = 'CELoss'
                                    config['max_id'] = 1000
                                    config['num_epochs'] = 25
                                    config['img_resize'] = 32
                                    config['train_ratio'] = 0.75
                                    config['weight_decay'] = 0.001
                                    config['step_size'] = 5
                                    config['task1'] = organ
                                    config['task2'] = modality
                                    config['id_task_to_run_secondary'] = modality
                                    config['experiment_tpye'] = 'nbrData_reduced_secondary_upsampled'
                                    config['reduced_percentage'] = reduced_percentage
                                    config['mean_organ'] = mean_organ
                                    config['std_organ'] = std_organ
                                    config['mean_modality'] = mean_modality
                                    config['std_modality'] = std_modality
                            
                                    if create_json:
                                        with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                            json.dump(config, json_file)
                                    experiment_list.append(config.copy())
                                    id += 1

    print(str(id) + " config files created for nbrData reduced secondary upsampled experiments")
        
    return experiment_list

def config_experiments_sampled(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/')
    
    # number of datapoints experiments
    for seed in [42]:
        for sampling_percentage in [100,75,50,35,25,10,5]:
                    
            # initialize config 
            config = {}
                            
            # assign config parameters
            config['model_type'] = 'ResNet18'
            config['experiment_id'] = str(id)
            config['model_pretrained'] = True
            config['batch_size'] = 512
            config['learning_rate'] = 0.005
            config['seed'] = seed
            config['criterion_type'] = 'CELoss'
            config['max_id'] = 1000
            config['num_epochs'] = 25
            config['img_resize'] = 32
            config['train_ratio'] = 0.75
            config['weight_decay'] = 0.001
            config['step_size'] = 5
            config['experiment_tpye'] = 'sampled'
            config['sampling_percentage'] = sampling_percentage
                            
            if create_json:
                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                    json.dump(config, json_file)
            experiment_list.append(config.copy())
            id += 1
                            
    print(str(id) + " config files created for sampled experiments")   
    
    return experiment_list

def config_experiments_sampled_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for seed in [42]:
        for modality in range(0,5):
            for sampling_percentage in [100,75,50,35,25,10,5]:
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = 'ResNet18'
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 512
                config['learning_rate'] = 0.005
                config['seed'] = seed
                config['criterion_type'] = 'CELoss'
                config['max_id'] = 1000
                config['num_epochs'] = 25
                config['img_resize'] = 32
                config['train_ratio'] = 0.75
                config['weight_decay'] = 0.001
                config['step_size'] = 5
                config['id_task_to_run_secondary'] = modality
                config['experiment_tpye'] = 'sampled_secondary'
                config['sampling_percentage'] = sampling_percentage
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    print(str(id) + " config files created for sampled secondary experiments")
        
    return experiment_list

def config_experiments_sampled_secondary_upsampled(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # secondary experiments 
    for seed in [42]:
        for modality in range(0,5):
            for sampling_percentage in [100,75,50,35,25,10,5]:
                # initialize config 
                config = {}
                            
                # assign config parameters
                config['model_type'] = 'ResNet18'
                config['experiment_id'] = str(id)
                config['model_pretrained'] = True
                config['batch_size'] = 512
                config['learning_rate'] = 0.005
                config['seed'] = seed
                config['criterion_type'] = 'CELoss'
                config['max_id'] = 1000
                config['num_epochs'] = 25
                config['img_resize'] = 32
                config['train_ratio'] = 0.75
                config['weight_decay'] = 0.001
                config['step_size'] = 5
                config['id_task_to_run_secondary'] = modality
                config['experiment_tpye'] = 'sampled_secondary_upsampled'
                config['sampling_percentage'] = sampling_percentage
                            
                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1

    print(str(id) + " config files created for sampled secondary upsampled experiments")
        
    return experiment_list

def config_experiments_sampled_reduced(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced experiments 
    for seed in [42]:
        for organ in range(0,10):
            for modality in range(0,5):
                for sampling_percentage in [100,75,50,35,25,10,5]:
                    for reduced_percentage in [25,50,75,100,85,95]:
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 512
                        config['learning_rate'] = 0.005
                        config['seed'] = seed
                        config['criterion_type'] = 'CELoss'
                        config['max_id'] = 1000
                        config['num_epochs'] = 25
                        config['img_resize'] = 32
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 0.001
                        config['step_size'] = 5
                        config['task1'] = organ
                        config['task2'] = modality
                        config['experiment_tpye'] = 'sampled_reduced'
                        config['reduced_percentage'] = reduced_percentage
                        config['sampling_percentage'] = sampling_percentage
                            
                        if create_json:
                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                json.dump(config, json_file)
                        experiment_list.append(config.copy())
                        id += 1

    print(str(id) + " config files created for sampled reduced experiments")
        
    return experiment_list

def config_experiments_sampled_reduced_secondary(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced secondary experiments 
    for seed in [42]:
        for organ in range(0,10):
            for modality in range(0,5):
                for sampling_percentage in [100,75,50,35,25,10,5]:
                    for reduced_percentage in [25,50,75,100,85,95]:
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 512
                        config['learning_rate'] = 0.005
                        config['seed'] = seed
                        config['criterion_type'] = 'CELoss'
                        config['max_id'] = 1000
                        config['num_epochs'] = 25
                        config['img_resize'] = 32
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 0.001
                        config['step_size'] = 5
                        config['task1'] = organ
                        config['task2'] = modality
                        config['id_task_to_run_secondary'] = modality
                        config['experiment_tpye'] = 'sampled_reduced_secondary'
                        config['reduced_percentage'] = reduced_percentage
                        config['sampling_percentage'] = sampling_percentage
                            
                        if create_json:
                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                json.dump(config, json_file)
                        experiment_list.append(config.copy())
                        id += 1

    print(str(id) + " config files created for sampled reduced secondary experiments")
        
    return experiment_list

def config_experiments_sampled_reduced_secondary_upsampled(results_dir, create_json=True):

    id = 0
    experiment_list = []
    
    # check the path
    if not os.path.exists(results_dir + 'configs/'):
        os.makedirs(results_dir + 'configs/') 
    
    # reduced secondary experiments 
    for seed in [42]:
        for organ in range(0,10):
            for modality in range(0,5):
                for sampling_percentage in [100,75,50,35,25,10,5]:
                    for reduced_percentage in [25,50,75,100,85,95]:
                        # initialize config 
                        config = {}
                            
                        # assign config parameters
                        config['model_type'] = 'ResNet18'
                        config['experiment_id'] = str(id)
                        config['model_pretrained'] = True
                        config['batch_size'] = 512
                        config['learning_rate'] = 0.005
                        config['seed'] = seed
                        config['criterion_type'] = 'CELoss'
                        config['max_id'] = 1000
                        config['num_epochs'] = 25
                        config['img_resize'] = 32
                        config['train_ratio'] = 0.75
                        config['weight_decay'] = 0.001
                        config['step_size'] = 5
                        config['task1'] = organ
                        config['task2'] = modality
                        config['id_task_to_run_secondary'] = modality
                        config['experiment_tpye'] = 'sampled_reduced_secondary_upsampled'
                        config['reduced_percentage'] = reduced_percentage
                        config['sampling_percentage'] = sampling_percentage
                            
                        if create_json:
                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                json.dump(config, json_file)
                        experiment_list.append(config.copy())
                        id += 1

    print(str(id) + " config files created for sampled reduced secondary upsampled experiments")
        
    return experiment_list
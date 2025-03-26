from misc.DataReader import *
import os
import torch
from utilz.utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import yaml
import datetime

from train_test import *
import argparse
from models.HDAAGT import *
import torch.optim as optim

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the main script with a config file.")
    parser.add_argument("-c", "--config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    # Otherwise we use the default directory to the config path
    if args.config is None:
        args.config = "configs/config.yaml"
        print(f"Using the '{args.config}' file")

    with  open(args.config ) as file:
        config = yaml.safe_load(file)
    cwd = os.getcwd()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M") # Current time, used for saving the log
    print(f"Total  # of GPUs {torch.cuda.device_count()}")
    print(f"Using {device} device")
    path = 'data/Changchun/'

    # config non-indipendent parameters
    config['NFeatures']= len(config["Columns_to_keep"]) # This goes to the data preparation process
    config["input_size"] = config["NFeatures"]# This goes to the model, we add couple of more features to the input
    config["output_size"] = len(config['xy_indx']) # The number of columns to predict
    config['device'] = device
    config['ct'] = ct
    log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
    if config['verbal']:
        for arg in config.items(): # Print the arguments to the log file
            savelog(f"{arg[0]} = {arg[1]}", ct)
    config['sos'] = torch.tensor(config['sos']).repeat(config['Nnodes'], 1).to(device)
    config['eos'] = torch.tensor(config['eos']).repeat(config['Nnodes'],1).to(device)


    # Create datasets, we split the datasets to make sure there is no data leakage between training and test samples
    Scene_sind = Scenes(config)
    if not config['generate_data']:
        savelog("Loading data ...", ct)
        Scene_sind.load()
        indices = torch.load(os.path.join(cwd, 'Pickled', 'indices.pth'))
    else:
        read_CSV(Scene_sind, config)
        indices = torch.randperm(len(Scene_sind.Scene)).tolist()
        torch.save(indices, os.path.join(cwd, 'Pickled', 'indices.pth'))
        print('Total data lengths is: ',len(indices))
    
    scene_tr, scene_tst = prep_data(Scene_sind, indices, config)
    train_loader = DataLoader(scene_tr, batch_size = config['batch_size'], shuffle=False)
    test_loader = DataLoader(scene_tst, batch_size = config['batch_size'], shuffle=False)


    model = HDAAGT(config).to(device) # Here we define our model
    savelog("Creating model from scratch",ct)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['schd_stepzise'], gamma=config['gamma'])
        
    if config['Load_Model']:
        savelog("Loading model from the checkpoint", ct)
        model = model.load_state_dict(torch.load(config['Load_Model_Path'])).to(device)

    if config['Train']:
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        Best_Model, train_loss, trainADE, trainFDE = train_model(model, optimizer, criterion, scheduler, train_loader,test_loader, config)
        savelog(f"{log_code} with Avg loss of {train_loss[-1]}, ADE of {trainADE[-1]}, FDE of {trainFDE[-1]}", f"summary {ct}")
        if train_loss[-1] < 1.5:
            savelog(f"Saving result of {log_code}", ct)
            torch.save(Best_Model, os.path.join(cwd,'Pickled', 'best_trained_model.pth'))
            torch.save(train_loss, os.path.join(cwd,'Pickled', 'epoch_losses.pt'))
        savelog(f"Training finished for {ct}!", ct)

    if config['Test']: # If not training, then test the model
        Topk_Selected_words, ADE, FDE = test_model(model, test_loader, config)
        savelog(f"Average Displacement Error: {ADE}, Final Displacement Error: {FDE}", ct)
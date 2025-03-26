
    
    if Seed:
        scene_tr = torch.load(os.path.join('Pickled', 'Train', 'Changchuntrain.pth'))
        scene_tst = torch.load(os.path.join('Pickled', 'Test', 'Changchuntest.pth'))
    else:
        Scene_clss = Scenes(sl, future, Nusers, NFeatures, device)
        Scene_clss.load(path)
        sencelen = Scene_clss.Scene.shape[0]
        indices = list(range(sencelen))
        random.shuffle(indices)
        torch.save(indices, os.path.join(cwd, 'Pickled', 'indices.pth'))
        print('Total data lengths is: ',len(indices))
        scene_tr, scene_tst = prep_data(Scene_clss, indices, 5, 0.6, 0.8, batch_size, 'Pickled', 'Changchun')

    train_loader = DataLoader(scene_tr, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(scene_tst, batch_size=batch_size, shuffle=False)
    print("approximately 80% of the data is used for training and 20% for testing", len(train_loader), len(test_loader))

        

    # Use argparse to get the parameters
    parser = argparse.ArgumentParser(description='Trajectory Prediction')
    parser.add_argument('--ct', type=str, default=ct, help='Current time')
    parser.add_argument('--Nfeatures', type=int, default=NFeatures, help='Number of features')
    parser.add_argument('--Nnodes', type=int, default=Nnodes, help='Number of nodes')
    parser.add_argument('--NTrfL', type=int, default=NTrfL, help='Number of traffic lights')
    parser.add_argument('--sl', type=int, default=sl, help='Sequence length')
    parser.add_argument('--future', type=int, default=future, help='Future length')
    parser.add_argument('--sw', type=int, default=sw, help='Sliding window')
    parser.add_argument('--sn', type=int, default=sn, help='Sliding number')
    parser.add_argument('--Columns_to_keep', type=list, default=Columns_to_keep, help='Columns to keep')
    parser.add_argument('--Columns_to_Predict', type=list, default=Columns_to_Predict, help='Columns to predict')
    parser.add_argument('--TrfL_Columns', type=list, default=TrfL_Columns, help='Traffic light columns')
    parser.add_argument('--Nusers', type=int, default=Nusers, help='Number of maneuvers')
    parser.add_argument('--sos', type=int, default=sos, help='Start of sequence')
    parser.add_argument('--eos', type=int, default=eos, help='End of sequence')
    parser.add_argument('--xyidx', type=list, default=xyid, help='X and Y index')
    parser.add_argument('--Centre', type=list, default=Centre, help='Centre')

    parser.add_argument('--input_size', type=int, default=input_size, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='Hidden size')
    parser.add_argument('--num_layersGAT', type=int, default=num_layersGAT, help='Number of layers')
    parser.add_argument('--num_layersGRU', type=int, default=num_layersGRU, help='Number of layers')
    parser.add_argument('--output_size', type=int, default=output_size, help='Output size')
    parser.add_argument('--n_heads', type=int, default=n_heads, help='Number of heads')
    parser.add_argument('--concat', type=bool, default=concat, help='Concat')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout')
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2, help='Leaky relu slope')
    parser.add_argument('--expansion', type=int, default=expansion, help='Expantion')


    parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--patience_limit', type=int, default=patience_limit, help='Patience limit')
    parser.add_argument('--schd_stepzise', type=int, default=schd_stepzise, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=gamma, help='Scheduler Gamma')


    parser.add_argument('--only_test', type=bool, default=only_test, help='Only test')
    parser.add_argument('--Train', type=bool, default=Train, help='Train')
    parser.add_argument('--test_in_epoch', type=bool, default=test_in_epoch, help='Test in epoch')
    parser.add_argument('--model_from_scratch', type=bool, default=model_from_scratch, help='Model from scratch')
    parser.add_argument('--load_the_model', type=bool, default=load_the_model, help='Load the model')
    parser.add_argument('--Seed', type=bool, default=Seed, help='Seed')
    parser.add_argument('--device', type=str, default=device, help='device')

    
    args = parser.parse_args()
    LR = [args.learning_rate] #, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
    # LR = [1e-2]
    HS = [args.hidden_size] #, 32, 64, 128, 256, 512, 1024]
    NL = [num_layersGAT] #, 2, 3, 4]
    BS = [args.batch_size] #, 32, 64, 128, 256, 512, 1024]
    datamax, datamin, ZoneConf = 1,0,[]
    Hyperloss = []
    if Train:
        print("Starting training phase")
        model = GGAT(args).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        print(f"Learning rate: {learning_rate}, Hidden size: {hidden_size}, Batch size: {batch_size}")
        Best_Model, predicteddata,Targetdata, test_loss, epoch_losses, Max_Acc =\
            train_model(model, optimizer, criterion, train_loader, test_loader, clip, args, datamax, datamin, ZoneConf)
        Hyperloss += [epoch_losses]
        code = f"LR{int(learning_rate*100000)}_HS{hidden_size}_NL{num_layersGAT}_BS{batch_size} {ct}"
        savelog(f"{code} with Max Acc of {Max_Acc}", f"summary {ct}")
        if Max_Acc > 0.001:
            savelog(f"Saving result of {code} with Max Acc of {Max_Acc}", ct)
            #save output data as a pickle file
            torch.save(Best_Model, os.path.join(cwd,'Processed', code + 'Bestmodel.pth'))
            torch.save(predicteddata, os.path.join(cwd,'Pickled', code + 'Predicteddata.pt'))
            torch.save(test_loss, os.path.join(cwd,'Pickled', code + 'test_losses.pt'))
            torch.save(epoch_losses, os.path.join(cwd,'Pickled', code + 'epoch_losses.pt'))
            torch.save(Hyperloss, os.path.join(cwd,'Pickled', code + 'Hyperloss.pt'))
            torch.save(Targetdata, os.path.join(cwd,'Pickled', code + 'Targetdata.pt'))
        savelog(f"Training finished for {ct}!", ct)
    if Test:
        predicteddata,Targetdata, l, acc, log = test_model(model,criterion, test_loader, args.Columns_to_Predict, future, Nusers, input_size, sos, eos, args)

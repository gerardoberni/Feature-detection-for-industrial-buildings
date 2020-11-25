import numpy as np
import utils, os, torch, torchvision, argparse
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from TerniumDataset import TerniumDataset
from auxiliary import loadModel, createModel, saveCheckpoint, loadCheckpoint, get_transform, mapArgs

def main(epochs, data_path, params, device, fromCheckpoint):

    # Datasets y Generators
    training_path = os.path.join(data_path,'Training')
    torch.manual_seed(0)
    np.random.seed(0)
    training_set = TerniumDataset(training_path, get_transform(train=True))
    
    # split the dataset in train and test set
    #indices = torch.randperm(len(training_set)).tolist()
    #training_set = torch.utils.data.Subset(training_set, indices[0:100])
    
    training_loader = torch.utils.data.DataLoader(training_set, **params)
    
    # Construír modelo y optimizador
    model = createModel()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0009)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # OPCIONAL: Resumir entrenamiento a partir de checkpoint
    if fromCheckpoint:
        model, optimizer, last_epoch = loadCheckpoint(model,optimizer,"checkpoint_best.tar")
    
    # Entrenar durante x epochs
    for epoch in range(epochs):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))
        
        # Training con utils
        train_one_epoch(model, optimizer, training_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

    weight_path = os.path.join(data_path,'Weights','last_weight.pth')
    state = model.state_dict()
    torch.save(state, weight_path)
    print("Saved last weight to: ",weight_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs", type=int, help='Número de epochs, DEFAULT=1')
    parser.add_argument("-bs","--batch_size", type=int, help='Tamaño de batch, DEFAULT=3')
    parser.add_argument("-w","--workers", type=int, help='Número de procesos para cargar datos, DEFAULT=6')
    parser.add_argument("-d","--device", type=int, help='CPU = -1, GPU = 1...n_GPU, DEFAULT=cuda:0')
    parser.add_argument("-chk","--checkpoint", type=bool, help='True o False, DEFAULT=False')
    parser.add_argument("-data","--data", type=str, help='Ruta a la carpeta base con las imágenes de train y test, DEFAULT=cwd')
    args = parser.parse_args()

    # Parámetros de entrenamiento default
    device = torch.device('cuda')
    epochs = 1
    params = {
    'batch_size':3,
    'shuffle': True,
    'num_workers': 6,
    'collate_fn':utils.collate_fn
    }
    fromCheckpoint = False
    data_path = os.getcwd()

    # Valores de CLI opcionales
    epochs, device = mapArgs(params, args, epochs, device, True)
    
    if args.checkpoint is not None:
        fromCheckpoint = args.checkpoint

    if args.data is not None:
        data_path = args.data
    
    torch.cuda.empty_cache()
    main(epochs, data_path, params, device, fromCheckpoint)
    torch.cuda.empty_cache()
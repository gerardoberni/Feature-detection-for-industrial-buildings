import torch, utils, torchvision, os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T

# Carga modelo desde algún peso guardado.
def loadModel(name, model):
    weight_path = os.path.join(os.getcwd(), 'Weights', name)
    model.load_state_dict(torch.load(weight_path))
    return model

# Crear modelo pre entrenado de FasterRCNN.
def createModel():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 44
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Guardar checkpoints durante el entrenamiento para retomarlo más adelante.
def saveCheckpoint(model,optimizer,epoch,name,loss):
    checkpoint_path = os.path.join(os.getcwd(), 'Checkpoints', name)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, checkpoint_path)
    print("Saved checkpoint to: ",checkpoint_path)

# Cargar un checkpoint previo para continuar el entrenamiento previo.
def loadCheckpoint(model,optimizer,name):
    checkpoint_path = os.path.join(os.getcwd(), 'Checkpoints', name)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint from: ", checkpoint_path)
    print("Checkpoint loss: ", checkpoint['loss'])
    print("Checkpoint epoch: ", epoch)
    return model, optimizer, epoch

# Convertir de PIL a tensor y hacer pequeño Data Augmentation.
def get_transform(train):
    transforms = []
    # Convertimos de PIL a tensor
    transforms.append(T.ToTensor())
    if train:
        # Hacemos flip de imagenes random como data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Procesamos los argumentos recibidos en el CLI
def mapArgs(params,args,epochs,device,train):
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.workers is not None:
        params['num_workers'] = args.workers
    if args.device is not None:
        if args.device == -1:
            device = torch.device('cpu')
        else:
            d = 'cuda:' + str(args.device)
            device = torch.device(d)
    if train:
        if args.epochs is not None:
            epochs = args.epochs
    return epochs, device
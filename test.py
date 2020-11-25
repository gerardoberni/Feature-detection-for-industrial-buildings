from PIL import Image
import numpy as np
import utils, os, torch, torchvision, argparse, cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from auxiliary import loadModel, createModel, saveCheckpoint, loadCheckpoint, get_transform, mapArgs
from TerniumDataset import TerniumDataset

# Agrupamos las predicciones a listas
def group_by_lists(_predictions, _images):
    images = []
    for img in _images:
        for i in img:
            im = Image.fromarray(i.mul(255).permute(1, 2, 0).cpu().byte().numpy())
            images.append(im)

    boxes = []
    labels = []
    scores = []

    for batch in _predictions:
        for b in batch:
            boxes.append(b['boxes'])
            labels.append(b['labels'])
            scores.append(b['scores'])

    return images, boxes, labels, scores

# Realizar inferencias con el test dataset y transferirlas a CPU.
def test(model, test_loader, device=torch.device('cuda')):
    cpu_device = torch.device('cpu')
    torch.cuda.empty_cache()
    predictions = []
    images = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img_batch, _ in test_loader:
            image = list(img.to(device) for img in img_batch)
            outputs = model(image)
            
            # Liberar memoria de GPU
            img = list(im.to(cpu_device) for im in image)
            images.append(img)
            del image
            del img
            torch.cuda.empty_cache()
            
            pred_dict = {}
            pred_list = []
            for output in outputs:
                pred_dict['boxes']  = output['boxes'].cpu().numpy()
                pred_dict['labels'] = output['labels'].cpu().numpy()
                pred_dict['scores'] = output['scores'].cpu().numpy()
                pred_list.append(pred_dict)
            predictions.append(pred_list)
            del outputs
            del pred_dict
            del pred_list
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return predictions, images

# Cargar el dataset e inicializar los parámetros a utilizar en la inferencia.
def main(params,data_path,device,thresh):
    testing_path = os.path.join(data_path,'Testing')
    PATH = 'last_weight.pth'
    txt = open("classes.txt","r")
    txtfile = txt.read()
    CLASSES = txtfile.split('\n')
    CLASSES.pop()

    model = createModel()
    model = loadModel(PATH, model)
    model.to(device)
    
    testing_set = TerniumDataset(testing_path, get_transform(train=False))
    #indices = torch.randperm(len(testing_set)).tolist()
    #testing_set = torch.utils.data.Subset(testing_set, indices[:10])
    indices = list(range(5,10))
    testing_set = torch.utils.data.Subset(testing_set, indices)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)
    predictions, images = test(model, test_loader, device)

    del model
    del test_loader
    del testing_set
    del indices

    images, boxes, labels, scores = group_by_lists(predictions, images)

    categorias = { i : CLASSES[i] for i in range(0, len(CLASSES) ) }

    # Se dibujan los bounding boxes en las imágenes utilizadas del test dataset y se guardan a disco.
    for n_img, img in enumerate(images):
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_file = "saved_image_" + str(n_img) + ".jpg"
        img_file = os.path.join('Outputs', img_file)
        nombres = labels[n_img]
        for n_box, box in enumerate(boxes[n_img]):
            score = scores[n_img][n_box]
            if score > thresh:
                x_min = box[0]
                y_min = box[1]
                x_max = box[2]
                y_max = box[3]
                categoria = categorias[nombres[n_box]]
                cv2.rectangle(image, (x_min,y_min), (x_max,y_max), color=(0,255,0), thickness=2)
                cv2.putText(image, str(categoria), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), thickness=3)
                cv2.putText(image, str(score), (x_min, int(y_min + 50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (124,255,80), thickness=3)

        # Guardar imagen con boxes en disco
        cv2.imwrite(img_file, image)
        print("Saved image to disk: ", img_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--threshold", type=float, help='Threshold de confianza en inferencias, DEFAULT=0.3')
    parser.add_argument("-bs","--batch_size", type=int, help='Tamaño de batch, DEFAULT=3')
    parser.add_argument("-w","--workers", type=int, help='Número de procesos para cargar datos, DEFAULT=6')
    parser.add_argument("-d","--device", type=int, help='CPU = -1, GPU = 1...n_GPU, DEFAULT=cuda:0')
    parser.add_argument("-data","--data", type=str, help='Ruta a la carpeta base con las imágenes de train y test, DEFAULT=cwd')
    args = parser.parse_args()

    params = {
        'batch_size':3,
        'shuffle': False,
        'num_workers': 6,
        'collate_fn':utils.collate_fn
        }
    
    data_path = os.getcwd()
    device = torch.device('cuda')
    threshold = 0.3
    
    # Valores de CLI opcionales
    _, device = mapArgs(params, args, 0,device, False)

    if args.threshold is not None:
        threshold = args.threshold

    if args.data is not None:
        data_path = args.data

    torch.cuda.empty_cache()
    main(params,data_path,device,threshold)
    torch.cuda.empty_cache()
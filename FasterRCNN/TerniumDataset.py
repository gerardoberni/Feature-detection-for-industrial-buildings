import os, torch, cv2, torch.utils.data
import numpy as np
import pandas as pd
from PIL import Image

class TerniumDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, training=True):
        # Guardamos nuestras variables base
        self.root = root
        self.transforms = transforms
        self.training = training
        self.images = list(sorted(os.listdir(os.path.join(root, "Images"))))

        if training:
            self.anotations = list(sorted(os.listdir(os.path.join(root, "Anotations"))))

    def __getitem__(self, idx):
        # Obtenemos paths de imagen y anotaciones
        img_path = os.path.join(self.root, "Images", self.images[idx])
        imgPIL = Image.open(img_path).convert("RGB")

        if self.training:
            txt_path = os.path.join(self.root, "Anotations", self.anotations[idx])
            imgPIL = Image.open(img_path).convert("RGB")
        
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            txt_file = pd.read_csv(txt_path, header=None, delim_whitespace=True)
            txt_file = txt_file.to_numpy()

            # Convertimos anotaciones de formato YOLO a Pascal
            coords = txt_file[:, 1:]
            labels = txt_file[:, 0]

            size = (img.shape[1], img.shape[0])
            n_boxes = coords.shape[0]

            boxes = []
            for i in range(n_boxes):
                box = self.convertYoloToPascal(size, coords[i])
                boxes.append(box)

            boxes = torch.tensor(boxes,dtype = torch.int)
            labels = torch.tensor(labels, dtype = torch.int64)
        
            # Creamos nuestro diccionario que contiene la información
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels

            if self.transforms is not None:
                imgPIL, target = self.transforms(imgPIL, target)

            return imgPIL, target

        target = {}
        target["id"] = img_path
        if self.transforms is not None:
                imgPIL, target = self.transforms(imgPIL, target)
        return imgPIL, target

    def __len__(self):
        return len(self.images)
    
    # Conversor de YOLO a Pascal
    def convertYoloToPascal(self, size, coord):
        x2 = int( ( (2 * size[0] * float(coord[0])) + (size[0] * float(coord[2]))) / 2)
        x1 = int( ( (2 * size[0] * float(coord[0])) - (size[0] * float(coord[2]))) / 2)

        y2 = int( ( (2 * size[1] * float(coord[1])) + (size[1] * float(coord[3]))) / 2)
        y1 = int( ( (2 * size[1] * float(coord[1])) - (size[1] * float(coord[3]))) / 2)

        return (x1,y1,x2,y2)
# Guía para instalar y correr FasterRCNN.
Este modelo de [FasterRCNN](https://arxiv.org/abs/1506.01497) utiliza como backbone [resnet50](https://arxiv.org/abs/1512.03385), el cuál está entrenado en el dataset de [COCO](https://cocodataset.org/).
Fue implementada mediante [Pytorch](https://pytorch.org/) y cuenta con 2 scripts (train y test) con los cuales se puede entrenar o se pueden hacer inferencias en imágenes.
Al momento actual, se ha entrenado con el dataset de *Ternium* que fue etiquetado por todo el grupo, cuenta con un costo actual del *0.3*.
También se implementó un sistema de checkpoints el cuál al final de cada epoch guarda el estado del modelo en ese momento, de esta manera
se puede retomar el entrenamiento posteriormente.

## Instalación
*LOS SCRIPTS DE TRAIN Y TEST DEBEN CORRERSE DESDE LA CARPETA BASE DEL PROYECTO.*

Correr el siguiente comando en la terminal para instalar los requisitos.
```
pip install requirements.txt --ignore-installed
```

## Setup de entorno
Una vez que hayas instalado los requisitos procede a crear las siguientes carpetas en caso de que no las tengas.

```
/Weights/
/Checkpoints/
/Outputs/
```
Descarga de la [carpeta](https://drive.google.com/drive/u/1/folders/1WLNqWjLJeV6onZQCkB5k4fGdXHakS_A6) el archivo `best_weight.pth` y guárdalo en la carpeta `/Weights/`.

## Preparar dataset
Descargar de esta [carpeta](https://drive.google.com/drive/u/1/folders/1WXjElJai_S5QxWEQlW4Ui71KOQaJSURu) las sub carpetas *Anotations* e *Images* a la computadora de manera local y agrupa en una carpeta "Train" un porcentaje de la carpeta Images y Anotations; haz lo mismo pero con una carpeta llamada "Test" de tal manera que termines con la siguiente estructura:

```
/Test/
----- /Images/
----- /Anotations/

/Train/
----- /Images/
----- /Anotations/
```

Ya que tengas tu entorno listo procede a correr el siguiente script en la terminal en caso de que no tengas tu dataset *limpio*.

```
python setDataset.py
```

## Entrenar FasterRCNN
Para iniciar el entrenamiento se debe correr el siguiente script considerando la siguiente CLI:

`python train.py`
Script base, a continuación se muestra la interface de comando para pasar argumentos al entrenamiento.

`-e EPOCHS, --epochs EPOCHS`
Número de epochs, DEFAULT=1

`-bs BATCH_SIZE, --batch_size BATCH_SIZE`
Tamaño de batch, DEFAULT=3

`-w WORKERS, --workers WORKERS`
Número de procesos para cargar datos, DEFAULT=6

`-d DEVICE, --device DEVICE`
CPU = -1, GPU = 1...n_GPU, DEFAULT=cuda:0

`-chk CHECKPOINT, --checkpoint CHECKPOINT`
Resumir entrenamiento a partir de un checkpoint, , DEFAULT=False

`-data DATA, --data DATA`
Ruta a la carpeta base con las imágenes de train y test, DEFAULT=cwd

## Probar FasterRCNN
Para hacer inferencias en imágenes de testing se debe correr el siguiente script considerando la siguiente CLI:

`python test.py`
Script base, a continuación se muestra la inferface de comando para pasar argumentos a testing.

`-t THRESHOLD, --threshold THRESHOLD`
Threshold de confianza en inferencias, DEFAULT=0.3

`-bs BATCH_SIZE, --batch_size BATCH_SIZE`
Tamaño de batch, DEFAULT=3

`-w WORKERS, --workers WORKERS`
Número de procesos para cargar datos, DEFAULT=6

`-d DEVICE, --device DEVICE`
CPU = -1, GPU = 1...n_GPU, DEFAULT=cuda:0

`-data DATA, --data DATA`
Ruta a la carpeta base con las imágenes de train y test, DEFAULT=cwd

## Comentarios adicionales
Las inferencias se guardan en la carpeta `/Outputs/`, los pesos en `/Weights/` y los checkpoins de cada epoch
en `/Checkpoints/`.

El tamaño del batch size aproximadamente utiliza 2 GB de VRAM por cada unidad de batch ex:
* batch_size = 2 -> 4GB VRAM
* batch_size = 3 - > 6GB VRAM

El número de workers dependerá de la cantidad de procesadores que tenga el CPU, un número recomendable es entre 4 y 6.
Se recomienda correr en otra terminal el comando `watch -n 1 nvidia-smi` para monitorear el consumo de VRAM durante el uso del modelo.

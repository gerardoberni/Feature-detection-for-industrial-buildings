Todo esto se hizo bajo el sistema operativo *UBUNTU 20* y con *Docker*.

En este punto ya deberías tener los requirements originales del REPO de Jakarta pero tuve que instalar
al final los que vienen en *requirements_extras.txt*, puede que se necesiten o puede que no.

También se incluye el Dockerfile que hice para crear mi imagen propia con OpenCV,Pytorch y CUDA.

*NO SE INCLUYEN LOS WEIGHTS NI LAS IMAGENES, ESO SE TIENE QUE BAJAR DE DONDE VIENE EN EL REPO ORIGINAL*

Lista de archivos modificados:

* `scripts_for_datasets/COWC_EESRGAN_FRCNN_dataset.py`
* `scripts_for_datasets/scripts_GAN_HR-LR.py`
* `train.py`
* `config_GAN.json`
* `test.py`

Pasos
* 1.- Descargar el pre-made dataset.
* 2.- Modificar los archivos listados previamente.
* 3.- Correr scripts_GAN_HR-LR.py
* 4.- Copiar todos los .jpg de LR/x4 a Validation/val_images
* 5.- Correr test.py -c config_GAN.json //Me dice que mi driver de NVIDIA (v 10010) está desactualizado
* 6.- Correr train.py -c config_GAN.json //Tengo un bug en que num_samples = 0

# Guía para instalar y correr la EESRGAN.
Actualmente no se ha podido correr la GAN debido a falta de poder computacional, esperamos que con su VPC se puedan tener resultados favorables.
La prueba será con el dataset de COWC el cual puedes bajar de [aquí] (https://gdo152.llnl.gov/cowc/download/cowc-m/datasets/DetectionPatches_256x256.tgz)
También es necesario que descargues los weights del [google drive] (https://drive.google.com/drive/folders/15xN_TKKTUpQ5EVdZWJ2aZUa4Y-u-Mt0f).

## Instalación
*LOS SCRIPTS DEBEN CORRERSE DESDE LA CARPETA BASE EN DONDE ESTÁ EL CÓDIGO FUENTE.*

Asegurarse de tener python 3.6 > y haber descargado el dataset de COWC y los weights.
Correr el siguiente comando en la terminal para instalar los requisitos.
`pip install requirements_EESRGAN.txt --ignore-installed`

## Setup de entorno
Una vez que hayas instalado los requisitos procede a correr el siguiente script en la terminal.
`python set_workdir.py`
Sigue las instrucciones que te muestra el script para generar las carpetas necesarias y archivos
adicionales de configuración.
Tu dataset tiene que ser alguna de las carpetas que descargaste del COWC, por ejemplo la carpeta *Potsdam_ISPRS*.

## Preparar dataset
Ya que tengas tu entorno listo procede a correr el siguiente script en la terminal.
`python ./scripts_for_datasets/scripts_GAN_HR-LR.py`

## Entrenar GAN
Aún no se ha podido entrenar porque pide mucha memoria de la GPU, esperamos que con su VPC
ya se pueda correr sin problema.
Para iniciar el entrenamiento procede a correr el siguiente comando en la terminal.
`python train.py -c config_GAN.json`

## Testear GAN
Para hacer inferencias en imágenes de prueba con el dataset COWC procede a correr el siguiente comando en la terminal.
`python test.py -c config_GAN.json`

## Comentarios adicionales
La GAN utiliza pytorch por lo que es posible instalar en la VPC Pytorch de manera "nativa" similar a como funciona Docker, quizás esta opción pueda ser de mucha utilidad para simplificar el proceso de instalar los componentes necesarios.
Esta arquitectura no está implementada al 100% por lo que estamos en una fase meramente experimental para analizar más a fondo la utilidad que nos puede ofrecer y tomar una decisión más acertada acerca de si vale la pena implementarla o no.
Este README es una guía básica y cualquier duda favor de aclararla con el equipo 2.

## Equipo 2
Eloy
Berni
Gustavo
Jacobo


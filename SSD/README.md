# Guía para utilizar Single Shot Detection (SSD)

## Instalación
Corre el siguiente comando en la terminal para instalar los requisitos.

`pip install requirements.txt --ignore-installed`

## Setup de entorno
Es necesario descargar los pesos de esta [carpeta](https://drive.google.com/drive/u/1/folders/1zJLJ2r9rQgH1WD315hlERMbBssqqcOpg) de manera local.
En los comentarios del código se detalla la estructura de los archivos, las carpetas y cómo utilizar los pesos previamente descargados.

## Para entrenamiento
1.	Eliminar la seccion donde se monta la unidad del drive (gmail) en caso de utilizar una maquina local.
2.	Tener las etiquetas en formato YOLO.
3.	Eliminar la seccion donde copiamos y descomprimimos `data.zip`´
4.	Cambiar la direccion donde se extraen las classes al una direccion donde se encuentre el archivo classes a utilizar.
5.	Cambiar el resto de las direcciones a una direccion local.

## Para test
1.	Eliminar la seccion donde se monta la unidad del drive (gmail) en caso de utilizar una maquina local.
2.	Cambiar las direcciones a direcciones locales.
3.	La funcion *SSD_detection* tiene como argumentos:
	```
	direccion de la imagen a realizar la deteccion.
	[anchura de boundinbox].
	[altura de boinding de boundinbox].
	modelo cargado con el que se realizara la deteccion.
	```

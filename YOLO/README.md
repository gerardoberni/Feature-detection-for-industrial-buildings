# Guía para instalar y correr YOLO v4

Este modelo de [YOLO](https://arxiv.org/abs/2004.10934) fue implementado mediante [TensorFlow v1](https://www.tensorflow.org/), el modelo está pre entrenado a partir del dataset [COCO](https://cocodataset.org/#home). El nuevo entrenamiento se llevó a cabo con el dataset de *Ternium* etiquetado por todo el grupo.

## Instalación
Correr el siguiene comando en la terminal para instalar los requisitos.

```bash
pip install requirements.txt
```
## Setup de entorno
Antes de correr los archivos externos deben de estar correctamente actualizados.
1. `Generate_train.py` y `generate_test.py` deben tener las rutas actualizadas de donde están las imágenes de entrenamiento y pruebas respectivamente
2. De entrenar otras categorías distitntas a las ya entrenadas, es necesario cambiar el `obj.names` a las categorias en el orden de las etiquetas
3. Si se hará más o nuevo entrenamiento se requiere de actualizar la ruta en `obj.data` de en dónde se estarán guardando los respaldos, los respaldos se guardan cada *1000 iteraciones* en un nuevo archivo y *cada 100* en el archivo de `yolov4-obj_last.weights`.
4. Actualemente el archivo de configuración está hecho para entrenar sobre *45 categorías*, en caso de que eso cambie es necesario ajustar parámetros que se ajusten al número de categorías:
```
 * max_batches = (# de classes) * 2000 (pero siempre mayor a 6000)
 * steps = (80% de max_batches), (90% de max_batches)
 * filters = (# de classes + 5) * 3 
 * Opcional: En caso de tener problemas de memoria o que el entrenamiento tarde mucho. Cambiar la línea con el atributo random = 1 a random = 0
 ```
 5. Los scripts están pensados para ser trabajados en *3 fases*: _configuración inicial, entrenamiento e inferencias y en un formato de notebook_, cada script debe ser actualizado a las direcciones que se manejaran dependiendo del equipo
 
## Entrenar YOLO

## Probar YOLO


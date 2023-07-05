
# Examen Final Tratamiento de Datos - Clasificación de Imágenes
 
El objetivo de esta practica es asentar los conocimientos que se han obtenido durante el curso.

Para ello vamos a crear un clasificador de tipos de carnes que se utiliza en la industria real. 

Es un ejemplo simplificado, el modelo real utiliza imágenes de mayor resolución y el conjunto de entrenamiento es mucho mayor (requiriendo días para su entrenamiento). Pero para practicar, con esta pequeña muestra nos sirve.

El link para la descarga es:
https://drive.google.com/file/d/1Z5DJ-MVS1TQV1kow9mIFWTec-ZdOLRLF/view?usp=sharing

Tras descomprimir el archivo de drive obtendremos dos carpetas, la carpeta de entrenamiento y la carpeta de test. 

Con el fin de que todo el mundo elija el mismo conjunto de entrenamiento y test.

Los datos de test NO pueden utilizarse durante el proceso de entrenamiento, únicamente para evaluar el modelo final.

Se pide:

1. Crear un repositorio de Github (publico) en el que se va a subir un jupyter notebook y un archivo README.md (como mínimo)

2. Obtener un clasificador de imágenes de forma que dada una nueva imagen se pueda obtener la clase
correspondiente.

3) Se pide obtener las matrices de confusión del modelo, la matriz de confusión del error en training y la de test.

Se valorará la justificación de las técnicas de procesamiento de imágenes utilizadas (cambios de color, reducción de colores, aumento del contraste, detección de bordes, uso de convoluciones....), asi como el uso de clasificadores que otros grupos no estén utilizando, para que en el debate podáis compartir los resultados que se obtienen con otros tipos de modelos (por modelos se entiende: Modelos lineales, SVM, KNN, CNN...)

Sistema de evaluación:
Repositorio de GitHub y Readme.md                                     1 punto
Preprocesado de imagenes:                                             3 puntos
Obtención del clasificador:                                           3 puntos
Evaluación del clasificador (matrices de confusión):                  3 puntos

Obtener más de 95% de acierto se considerá un 10 automáticamente.

#################################################################################################################

Este repositorio proporciona una implementación de la clasificación de imágenes utilizando la optimizacion del modelo VGG16 preentrenado utilizando la técnica de aprendizaje de transferencia. 

El modelo es capaz de clasificar las imágenes de tipos de carnes en las siguientes cuatro clases:

'CLASS_01'
'CLASS_02'
'CLASS_03'
'CLASS_04'
'CLASS_05'
'CLASS_06'
'CLASS_07'
'CLASS_08'

# Dependencies

Languages - Python

Frameworks - Matplotlib, Scikit-learn, Numpy, TensorFlow, Keras

Entornos adicionales - Jupyter Notebook


# Entrenamiento, prueba y almacenamiento del modelo

Ejecute 'ImageClassification.ipynb' con Jupyter Notebook. 

He proporcionado un conjunto de datos de muestra en la carpeta "CarneDataset" como referencia. 

Puedes añadir tus propias imágenes o modificar el conjunto de datos por completo. 

Asegúrate de encargarte de las rutas configuradas.


# Resultado

Los resultados del entrenamiento y la prueba del modelo en el conjunto de datos de la muestra son los siguientes:


Precisión en el conjunto de entrenamiento: 0.8000

Pérdida en el conjunto de entrenamiento: 0.9473

Precisión en el conjunto de validación: 0.6154

Pérdida en el conjunto de validación: 1.2619

La matriz de confusión se genera al ejecutar 'ImageClassification.ipynb' con Jupyter Notebook:


# Procedimiento

------------------------------------------------------------------------------

import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import *
from keras.layers.convolutional import *

import tensorflow as tf

--------------------------------------------------------------------------------

Este código importa las bibliotecas y módulos necesarios para construir un modelo de aprendizaje profundo utilizando Keras y TensorFlow.

Numpy es una biblioteca para cálculos numéricos en Python.
Matplotlib es una biblioteca de trazado para Python.
Sklearn es una biblioteca para algoritmos y herramientas de aprendizaje automático en Python.
Keras es una API de redes neuronales de alto nivel en Python.
Tensorflow es un marco de aprendizaje automático de código abierto desarrollado por Google.

El código también importa funciones y clases específicas de estas bibliotecas para construir el modelo, como Sequential para crear un modelo secuencial, Dense para agregar capas densas, Flatten para aplanar la entrada, Adam para optimizar el modelo, categorical_crossentropy para calcular la pérdida, ImageDataGenerator para generar datos de imagen y varias capas para crear redes neuronales convolucionales

El código también configura el entorno para mostrar gráficos en línea usando %matplotlib en línea.

--------------------------------------------------------------------------------

train_Directorio = 'CarneDataset/train'
test_Directorio = 'CarneDataset/test'

--------------------------------------------------------------------------------

Estas son rutas de directorio para los conjuntos de datos de entrenamiento y prueba para un modelo de aprendizaje automático.

La variable train_Directorio apunta al directorio donde se almacenan las imágenes de entrenamiento. El directorio debe contener subdirectorios para cada clase, con las imágenes de cada clase almacenadas en el subdirectorio correspondiente.

La variable test_Directorio apunta al directorio donde se almacenan las imágenes de prueba. La estructura del directorio debe ser similar al directorio de entrenamiento, con subdirectorios para cada clase y las imágenes para cada clase almacenadas en el subdirectorio correspondiente.

Estas rutas de directorio se utilizan como entrada a la clase ImageDataGenerator en Keras, que genera lotes de datos de imagen para entrenar y probar el modelo de aprendizaje automático.

--------------------------------------------------------------------------------

print("Informacion de la carpeta TRAIN:")
train_batches = ImageDataGenerator().flow_from_directory(train_Directorio, target_size=(224,224), classes=['CLASS_01', 'CLASS_02', 'CLASS_03', 'CLASS_04', 'CLASS_05', 'CLASS_06', 'CLASS_07', 'CLASS_08'], batch_size=10)
print("Informacion de la carpeta TEST:")
test_batches = ImageDataGenerator().flow_from_directory(test_Directorio, target_size=(224,224), classes=['CLASS_01', 'CLASS_02', 'CLASS_03', 'CLASS_04', 'CLASS_05', 'CLASS_06', 'CLASS_07', 'CLASS_08'], batch_size=4)

--------------------------------------------------------------------------------

Este código utiliza la clase ImageDataGenerator de Keras para generar lotes de datos de imagen para entrenar y probar el modelo de aprendizaje automático.

El método flow_from_directory de la clase ImageDataGenerator lee imágenes de un directorio y genera lotes de datos de imágenes aumentadas.

--------------------------------------------------------------------------------

def plots(ims, figsize=(17,5), rows=2, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

--------------------------------------------------------------------------------

Este código define una función llamada gráficos que se puede utilizar para mostrar una cuadrícula de imágenes.

La función tiene varios parámetros:

Ims: una lista de imágenes para mostrar
figsize: el tamaño de la figura
rows: el número de filas en la cuadrícula
interp: si se debe aplicar la interpolación a las imágenes
titles: una lista de títulos para cada imagen

La función comprueba si las imágenes se almacenan como matrices numpy y las convierte al formato uint8 si es necesario. 
Luego crea una figura con el tamaño especificado y añade subtramas para cada imagen de la cuadrícula. 
El parámetro titles se utiliza para establecer el título de cada subtrama, y la función imshow de matplotlib se utiliza para mostrar cada imagen.

--------------------------------------------------------------------------------

imgs, labels = next(train_batches)

--------------------------------------------------------------------------------

Este código genera un lote de datos de imagen y las etiquetas correspondientes del generador train_batches creado anteriormente.

--------------------------------------------------------------------------------

plots(imgs, titles=labels)

--------------------------------------------------------------------------------

Este código llama a la función de gráficos definida anteriormente para mostrar una cuadrícula de imágenes con sus etiquetas correspondientes.










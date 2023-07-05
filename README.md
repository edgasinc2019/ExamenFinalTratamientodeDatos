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


# Train, test and save the model

Ejecute 'ImageClassification.ipynb' con Jupyter Notebook. 

He proporcionado un conjunto de datos de muestra en la carpeta "CarneDataset" como referencia. 

Puedes añadir tus propias imágenes o modificar el conjunto de datos por completo. 

Asegúrate de encargarte de las rutas configuradas.


# Resultado

Los resultados del entrenamiento y la prueba del modelo en el conjunto de datos de la muestra son los siguientes:


Precisión en el conjunto de entrenamiento: 1.0000

Pérdida en el conjunto de entrenamiento: 0,2596

Precisión en el conjunto de validación: 1.0000

Pérdida en el conjunto de validación: 0,2279

La matriz de confusión se genera al ejecutar 'ImageClassification.ipynb' con Jupyter Notebook.

<<<<<<< HEAD
=======
Vaca
>>>>>>> a842b776d31bce5dd302b323eca512731d074f80

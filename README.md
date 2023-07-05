
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

![Resultados](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/Matriz.png)


# Procedimiento

------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/01.png)

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

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/02.png)

--------------------------------------------------------------------------------

Estas son rutas de directorio para los conjuntos de datos de entrenamiento y prueba para un modelo de aprendizaje automático.

La variable train_Directorio apunta al directorio donde se almacenan las imágenes de entrenamiento. El directorio debe contener subdirectorios para cada clase, con las imágenes de cada clase almacenadas en el subdirectorio correspondiente.

La variable test_Directorio apunta al directorio donde se almacenan las imágenes de prueba. La estructura del directorio debe ser similar al directorio de entrenamiento, con subdirectorios para cada clase y las imágenes para cada clase almacenadas en el subdirectorio correspondiente.

Estas rutas de directorio se utilizan como entrada a la clase ImageDataGenerator en Keras, que genera lotes de datos de imagen para entrenar y probar el modelo de aprendizaje automático.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/03.png)

--------------------------------------------------------------------------------

Este código utiliza la clase ImageDataGenerator de Keras para generar lotes de datos de imagen para entrenar y probar el modelo de aprendizaje automático.

El método flow_from_directory de la clase ImageDataGenerator lee imágenes de un directorio y genera lotes de datos de imágenes aumentadas.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/04.png)

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

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/05.png)

--------------------------------------------------------------------------------

Este código genera un lote de datos de imagen y las etiquetas correspondientes del generador train_batches creado anteriormente.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/06.png)

--------------------------------------------------------------------------------

Este código llama a la función de gráficos definida anteriormente para mostrar una cuadrícula de imágenes con sus etiquetas correspondientes.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/07.png)

--------------------------------------------------------------------------------

Este código imprime los índices de clase para las clases en el conjunto de datos de entrenamiento.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/08.png)

--------------------------------------------------------------------------------

Este código define una función llamada plot_confusion_matrix que se puede utilizar para mostrar una matriz de confusión para un conjunto de etiquetas predichas y verdaderas.

La función tiene varios parámetros:

Cm: la matriz de confusión para mostrar
Clases: una lista de los nombres de las clases
normalize: si normalizar la matriz de confusión
title: el título de la trama
Cmap: el mapa de color a usar para la trama

La función utiliza imshow de matplotlib para mostrar la matriz de confusión como una imagen. Establece el título de la trama y añade una barra de color a la trama. Luego establece las marcas de verificación para el eje x y el eje y en los nombres de las clases.

Si normalizar se establece en True, la función normaliza la matriz de confusión dividiendo cada fila por la suma de la fila. Luego imprime la matriz de confusión normalizada.

A continuación, la función establece un umbral para el color del texto en el gráfico basado en el valor máximo en la matriz de confusión. Añade el texto de cada celda de la matriz de confusión a la trama. Finalmente, establece las etiquetas del eje x y del eje y para el gráfico.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/09.png)

--------------------------------------------------------------------------------

El código que proporcionó es una instancia del modelo VGG16 en la API Keras de TensorFlow. Aquí hay una explicación de los diferentes argumentos:

Include_top: 
Este argumento especifica si se debe incluir la capa totalmente conectada en la parte superior de la red. Si se establece en True, el modelo incluirá la capa superior, que consiste en una capa de agrupación promedio global seguida de una capa densa con el número especificado de clases (por defecto: 1000). Si se establece en False, se excluye la capa totalmente conectada, y la salida serán las características de 7x7x512.

Weights: 
este argumento determina la inicialización del peso del modelo. Al establecerlo en "imagenet", el modelo se inicializará con pesos preentrenados en el conjunto de datos de ImageNet. Si lo estableces en Ninguno, el modelo se inicializará al azar.

Input_tensor: 
Este argumento le permite especificar un tensor Keras como entrada al modelo. Si Ninguno, el modelo creará una nueva capa de entrada.

Input_shape: 
Este argumento se utiliza para especificar la forma de la imagen de entrada. Debe ser una tupla (altura, anchura, canales). Este argumento se ignora si se especifica input_tensor.

Pooling: 
Este argumento especifica el tipo de agrupación que se aplicará después del último bloque convolucional. Si es Ninguno, no se aplica la agrupación, y la salida serán las características de 7x7x512. Si se establece en "promedio", se aplicará la agrupación promedio global, lo que resultará en una salida 1D con forma (batch_size, 512). Si se establece en "max", se aplicará la agrupación global max en su lugar.

Classes: 
Este argumento determina el número de clases para la tarea de clasificación. Solo es relevante cuando include_top se establece en True.

Classifier_activation: 
Este argumento especifica la función de activación que se utilizará para la capa superior. El valor predeterminado es "softmax", que es adecuado para tareas de clasificación de varias clases.

Al instanciar el modelo VGG16 con estos argumentos, obtendrá un modelo VGG16 con la configuración especificada.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/10.png)

--------------------------------------------------------------------------------

El código que proporcionó crea una instancia del modelo VGG16 y luego llama al método summary() para imprimir un resumen de la arquitectura del modelo.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/11.png)

--------------------------------------------------------------------------------

La función type() en Python se utiliza para determinar el tipo de un objeto. 
En este caso, vgg16_model es una instancia del modelo VGG16 de la biblioteca Keras. 
Sin embargo, el tipo específico del objeto depende de la versión de Keras que estés usando.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/12.png)

--------------------------------------------------------------------------------

El tipo correcto de vgg16_model sería keras.engine.training.Model. Esta es la clase base para todos los modelos de Keras, incluido el modelo VGG16. Representa un modelo de red neuronal que se puede entrenar y utilizar para la inferencia.

La clase keras.engine.training.Model proporciona funcionalidades para compilar el modelo, entrenarlo con datos, evaluar el rendimiento y hacer predicciones. Sirve como la clase central para construir y trabajar con varios tipos de modelos de redes neuronales en Keras.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/13.png)

--------------------------------------------------------------------------------

El código que proporcionó tiene como objetivo crear un nuevo modelo secuencial y agregar todas las capas del modelo VGG16 preexistente (vgg16_model) al nuevo modelo (modelo). Después, se llama al método summary() para mostrar el resumen del nuevo modelo.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/14.png)

--------------------------------------------------------------------------------

El resumen muestra la arquitectura del modelo después de añadir las capas. Incluye el tipo de capa, la forma de salida y el número de parámetros en cada capa.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/15.png)

--------------------------------------------------------------------------------

El fragmento de código que proporcionó establece el atributo entrenable de cada capa del modelo en False. Esto significa que durante el entrenamiento, los pesos de estas capas no se actualizarán y permanecerán fijos.

Al establecer layer.trainable = False, esencialmente está congelando las capas del modelo. Esto puede ser útil en escenarios en los que desea utilizar las capas preentrenadas de un modelo (como VGG16) como extractores de características y solo entrenar las capas recién añadidas en la parte superior.

Congelar las capas puede ser beneficioso cuando tienes datos de entrenamiento limitados o cuando quieres evitar que se modifiquen las representaciones preentrenadas. Le permite aprovechar las características aprendidas de un gran conjunto de datos mientras adapta el modelo a una tarea específica con un conjunto de datos más pequeño.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/16.png)

--------------------------------------------------------------------------------

Después de añadir una capa densa con 8 unidades y una función de activación softmax al modelo.
La capa densa recién añadida con 8 unidades y activación de softmax se ha incluido en el modelo

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/17.png)

--------------------------------------------------------------------------------

El método compile() en Keras se utiliza para configurar el proceso de aprendizaje de un modelo. Requiere especificar el optimizador, la función de pérdida y las métricas de evaluación que se utilizarán durante el entrenamiento y la evaluación.

En el fragmento de código que proporcionó, se llama al método compile() en el modelo con los siguientes argumentos:

Adam(lr=0.07): 
El optimizador Adam se especifica con una tasa de aprendizaje de 0,07. Adam es un popular algoritmo de optimización utilizado para la optimización basada en gradientes. Al pasar lr=0.07, estableces la tasa de aprendizaje del optimizador en 0.07.

loss='categorical_crossentropy': 
La entropía cruzada categórica se especifica como la función de pérdida. La entropía cruzada categórica se utiliza comúnmente para tareas de clasificación de varias clases cuando los objetivos están codificados en caliente.

metrics=['accuracy']: 
La métrica utilizada para la evaluación durante el entrenamiento y las pruebas es la precisión. Es una métrica de uso común para las tareas de clasificación que mide la proporción de muestras correctamente clasificadas.

Al llamar a model.compile(), configura el modelo para el entrenamiento con el optimizador, la función de pérdida y la métrica de evaluación especificadas.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/18.png)

--------------------------------------------------------------------------------

El método fit() en Keras se utiliza para entrenar un modelo en un conjunto de datos determinado. Requiere especificar los datos de entrenamiento, los datos de validación, el número de épocas, el tamaño del lote y otros parámetros opcionales.

En el fragmento de código que proporcionó, se llama al método fit() en el modelo con los siguientes argumentos:

Train_batches:
Estos son los datos de entrenamiento proporcionados como lotes. Se supone que es un iterable que proporciona los datos de entrada y las etiquetas en lotes durante el entrenamiento.

Steps_per_epoch=1: 
El parámetro steps_per_epoch especifica el número de pasos (lote) a procesar antes de declarar una época como terminada. En este caso, lo establece en 1, lo que indica que después de procesar un lote, una época se considera completa.

Validation_data=test_batches: 
Estos son los datos de validación proporcionados como lotes. Se supone que es un iterable que proporciona los datos de entrada y las etiquetas en lotes durante la validación.

Validation_steps=13: 
El parámetro validation_steps especifica el número de pasos (batches) a procesar desde el conjunto de datos de validación antes de detener la validación para esa época. En este caso, lo establece en 13, lo que indica que se procesarán 13 lotes para su validación en cada época.

Epochs=15: 
El número de épocas especifica el número de veces que se debe pasar todo el conjunto de datos de entrenamiento a través del modelo durante el entrenamiento. En este caso, la formación continuará durante 15 épocas.

Batch_size=32: 
El tamaño del lote determina el número de muestras que se procesarán a la vez antes de actualizar los pesos del modelo. En este caso, cada lote contendrá 32 muestras.

Verbose=2: 
El parámetro verbose controla la verbosidad del proceso de entrenamiento. Establecerlo en 2 significa que se mostrará el progreso de cada época, incluidos los valores de pérdida y métricas.

Al llamar a model.fit(), se inicia el proceso de entrenamiento para el número especificado de épocas, utilizando los datos de entrenamiento y validación proporcionados.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/19.png)

--------------------------------------------------------------------------------

Parece que está tratando de realizar predicciones en un lote de imágenes de prueba utilizando el modelo entrenado. Aquí hay un desglose del fragmento de código que proporcionó:

Test_imgs, test_labels = next(test_batches): 
Esta línea obtiene un lote de imágenes de prueba y sus etiquetas correspondientes del iterador test_batches.

Plots(test_imgs, titles=test_labels): 
Esta línea llama a la función plots() para mostrar las imágenes de prueba con sus etiquetas correspondientes como títulos.

Print(test_labels): 
Esta línea imprime las etiquetas verdaderas de las imágenes de prueba.

predictions = model.predict_on_batch(np.array(test_imgs)): 
Esta línea utiliza el modelo entrenado para predecir las etiquetas de las imágenes de prueba llamando a model.predict_on_batch() en el lote de imágenes de prueba convertidas en una matriz NumPy.

Print(predictions): 
Esta línea imprime las etiquetas previstas para las imágenes de prueba.

Test_labels = np.array(test_labels.argmax(axis=1)): 
Esta línea convierte las etiquetas verdaderas en una matriz y aplica la función argmax() a lo largo del eje de la fila para obtener el índice del valor máximo (es decir, la clase predicha) para cada etiqueta.

predictions = np.array(predictions.argmax(axis=1)): 
Esta línea convierte las etiquetas predichas en una matriz y aplica la función argmax() a lo largo del eje de la fila para obtener el índice del valor máximo (es decir, la clase predicha) para cada predicción.

Parece que estás comparando las verdaderas etiquetas (test_labels) con las etiquetas predichas (predicciones) para evaluar el rendimiento del modelo. Al comparar las etiquetas verdaderas y predichas, puede calcular métricas como la precisión o crear una matriz de confusión para evaluar el rendimiento del modelo.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/20.png)

--------------------------------------------------------------------------------

Esto mostrará las etiquetas verdaderas de las imágenes de prueba en la consola o en la ventana de salida. La variable test_labels debe contener las verdaderas etiquetas de las imágenes de prueba.

Asegúrese de haber ejecutado el código necesario para cargar los datos de prueba y asignar las etiquetas verdaderas a la variable test_labels antes de llamar a print(test_labels).

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/21.png)

--------------------------------------------------------------------------------

Esto mostrará las etiquetas previstas para las imágenes de prueba en la consola o en la ventana de salida. La variable de predicciones debe contener las etiquetas predichas generadas por el modelo.

Asegúrese de haber ejecutado el código necesario para predecir en las imágenes de prueba y asigne las etiquetas predichas a la variable de predicciones antes de llamar a print(predicciones).

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/22.png)

--------------------------------------------------------------------------------

La función confusion_matrix() en scikit-learn se utiliza para calcular la matriz de confusión a partir de las etiquetas verdaderas y las etiquetas previstas.

La variable cm resultante contendrá la matriz de confusión, que es una matriz cuadrada con dimensiones iguales al número de clases. Los elementos de la matriz representan los recuentos de predicciones de verdadero positivo, falso positivo, verdadero negativo y falso negativo para cada clase.

Puede utilizar la matriz de confusión para analizar el rendimiento del modelo, calcular métricas como la precisión, el recuerdo y la puntuación F1, o visualizar los resultados utilizando un mapa de calor u otras técnicas de visualización.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/23.png)

--------------------------------------------------------------------------------

Este código comprueba si cada etiqueta de clase está presente tanto en test_labels como en las predicciones, y si no, elimina la etiqueta de clase correspondiente de la lista cm_plot_labels.

Después de ejecutar este código, la lista cm_plot_labels solo contendrá las etiquetas de clase que están presentes tanto en test_labels como en las predicciones. A continuación, puede utilizar esta lista actualizada para trazar o mostrar la matriz de confusión.

--------------------------------------------------------------------------------

![Codigo](https://github.com/edgasinc2019/ExamenFinalTratamientodeDatos/blob/main/Imagenes%20GitHub/24.png)

--------------------------------------------------------------------------------

Este código define la función plot_confusion_matrix(), que toma la matriz de confusión (cm), la lista de etiquetas de clase (cm_plot_labels) y un título opcional como entrada. 

Utiliza imshow() de matplotlib para visualizar la matriz de confusión como una imagen codificada por colores. La función también incluye etiquetas de eje, marcas de verificación y anotaciones de texto para las celdas de la matriz.

Puedes personalizar la apariencia de la trama y las etiquetas en función de tus requisitos. Después de definir la función, puede llamar a plot_confusion_matrix() con los argumentos apropiados para mostrar la matriz de confusión.

Asegúrate de haber importado las bibliotecas necesarias (matplotlib y numpy) antes de ejecutar este código.






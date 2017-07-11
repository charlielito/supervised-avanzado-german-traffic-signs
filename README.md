# Reto German Traffic Signs
## Descripcion
El [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) es un conjunto de imágenes de señales de tránsito Alemanas.

![alt text][s1] ![alt text][s2] ![alt text][s3] ![alt text][s4] ![alt text][s5] ![alt text][s6] ![alt text][s7] ![alt text][s8] ![alt text][s9]

Este dataset tiene mas de 50,000 imágenes separadas en 43 clases. El reto es construir un clasificador de imágenes que sea capaz de reconocer estas señales de tránsito. Adicionalmente, el dataset incluye las posiciones (aka bounding boxes) de los objetos dentro de la imagen.


### Objetivo
1. Crear un algoritmo que tome una imagen de entrada, ya sea como vector o matriz, y retorne el clase (`class_id`) a la que pertenece esa imagen.
1. Entrenar este algoritmo utilizando los datos de la carpeta `data/training-set`.
1. Medir el performance/score del algoritmo utilizando los datos de la carpeta `data/test-set`. El performance debe ser medido como
```python
score = n_aciertos / n_imagenes * 100
```
donde `n_aciertos` es el numero de imagenes clasificadas de forma correcta y `n_imagenes` es el numero total de imagenes en el `test-set`.

### Requerimientos
Clona este repositorio y ejecuta el commando
```bash
git checkout all_cnn2
```
Despues puedes instalar los requirementos fácilmente utilizando el commando

```bash
pip install -r requirements.txt
```
Dependiendo de tu entorno puede que necesites instalar paquetes del sistema adicionales, si tienes problemas revisa la documentación de estas librerías.

### Descarga y Preprocesamiento
Para descargar los datos ejecuta el comando
```bash
dataget get german-traffic-signs
```
Esto descarga los archivos en la carpeta `.dataget/data`, los divide en los conjuntos `training-set` y `test-set`, convierte las imagenes en `jpg` de dimensiones `32x32`. Las originalmente vienen en formato `.ppm` y con dimensiones variables. Si deseas mantener el formato original ejecuta en vez

```bash
dataget get --dont-process german-traffic-signs
```

### Metodo
##### Modelo
Se utilizó una Red Neuronal Convolucional con la siguiente arquitectura:

* Inputs: 3 filtros (RGB)
* Capa Convolucional: 16 filtros, kernel 5x5, padding 'same', funcion de activacion ELU
* Capa Convolucional: 32 filtros, kernel 5x5, padding 'same', stride 2, funcion de activacion ELU
* Capa Convolucional: 64 filtros, kernel 3x3, padding 'same', stride 2, funcion de activacion ELU
* Capa Convolucional: 64 filtros, kernel 3x3, padding 'same', funcion de activacion ELU
* Flatten: se aplana a vector
* Capa Densa: 256 neuronas, activacion ELU, dropout = 0.15
* Capa Densa: 128 neuronas, activacion ELU
* Capa Densa Output: 43 neuronal, activacion softmax

###### Parametros
Este modelo utiliza `314,955` parametros.

##### Entrenamiento
Se utilizó un Stocastic Gradient Descent con los siguentes parámetros

* Optimizador: ADAM
* Learning Rate: 0.001
* Batch Size: 128

##### Notas
No se intentó optimizar el modelo de ninguna manera, en especial:

* No se preprocesaron los datos de ninguna manera excepto estandarizar su tamaño de las imagenes a `32x32`

### Procedimiento
El modelo se encuentra en el archivo `model.py`. Para entrenarlo ejecuta el comando
```
python train.py
```
Este script realiza lo siguiente

* Utiliza `seed = 32` para controlar la aleatoreidad y que los resultados sean reproducibles
* Entrena el modelo por `10000` iteraciones
* Graba el modelo en los archivos `basic-conv-net.tf.*`


### Resultados
Ver el score del `test-set` ejecuta
```
python test.py
```

Resultado: **0.971576094627**


### Visualizacion
El cuaderno de jupyter `solucion.ipynb` incluye visualizaciones de algunos resultados, para verlo ejecuta el comando
```bash
jupyter notebook .
```
y abrelo desde el explorador de archivos de jupyter.


[s1]: http://benchmark.ini.rub.de/Images/gtsrb/0.png "S"
[s2]: http://benchmark.ini.rub.de/Images/gtsrb/1.png "S"
[s3]: http://benchmark.ini.rub.de/Images/gtsrb/2.png "S"
[s4]: http://benchmark.ini.rub.de/Images/gtsrb/3.png "S"
[s5]: http://benchmark.ini.rub.de/Images/gtsrb/4.png "S"
[s6]: http://benchmark.ini.rub.de/Images/gtsrb/5.png "S"
[s7]: http://benchmark.ini.rub.de/Images/gtsrb/6.png "S"
[s8]: http://benchmark.ini.rub.de/Images/gtsrb/11.png "S"
[s9]: http://benchmark.ini.rub.de/Images/gtsrb/8.png "S"

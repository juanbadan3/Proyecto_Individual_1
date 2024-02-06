# Proyecto_Individual_1- Henry
# Proyecto individual 1 carrera Data Science rol Data Engineer  Machine Learning Operations (MLOps)



![MLOps](https://miro.medium.com/v2/resize:fit:1200/1*G4QIhWno7rWFu391uoxLFg.jpeg)

# Introducción

Introducción:
Este proyecto se enfoca en la creación de una API mediante el uso del framework FastAPI, destinada a exhibir un sistema de recomendación y análisis de bases de datos de juegos ficticios. Aunque los resultados no derivan de un análisis real, tienen el propósito de demostrar las habilidades adquiridas durante el bootcamp.

La tarea asignada implicó desarrollar un Producto Mínimo Viable (MVP) que comprendiera cinco puntos finales de función y un sistema de recomendación basado en aprendizaje automático.

![PI1_MLOps_Mapa1](https://raw.githubusercontent.com/pjr95/PI_ML_OPS/main/src/DiagramaConceptualDelFlujoDeProcesos.png)


# Descripción y diccionario del conjunto de datos
Para descargar los archivos originales, ya que tienen mucho peso, se pueden encontrar en el siguiente enlace. [Datasets originales](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)



_**ETL**_:
Para conocer más sobre el desarrollo del proceso ETL, existe el siguiente enlace
[Notebook ETL](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/PI_MLOPs_ETL_EDA.ipynb)

_**Nombres de los datasets**_:
- australian_user_reviews
- australian_users_items
- output_steam_games

_**Proceso de Desanidado:**_:
1. Se procedió a desanidar algunas columnas que contenían diccionarios o listas como valores, facilitando así consultas futuras en la API.

_**Eliminación de Columnas no Utilizadas:**_:

2. Columnas no esenciales, como:
 Publisher, url, tags, price, specs, early_access, user_url, funny, last_edited, entre otras, fueron eliminadas de los conjuntos de datos.

_**Control de valores nulos**_:

3. Se eliminan valores nulos:
   - De output_steam_games: genres, release_date.
   - De australian_user_reviews: item_id.
   - De australian_user_items: user_id

_**Cambio del tipo de datos**_:

4. Las fechas se cambian a datetime para luego extraer el año:
   - De australian_user_reviews: la columna posted .
   - De output_steam_games: la columna release_date.

_**Se quitan datos sin valor**_:

5. Los datos que no tienen ningún valor:
   - De australian_user_items: la columna playtime_forever.

_**Fusión de conjuntos de datos**_:

6. Los datasets fueron combinados, generando un archivo .csv para las funciones 1 y 2, [Archivo para funciones 1 y 2](https://github.com/juanbadan3/Proyecto_Individual_1/blob/main/df_f1_2.csv) y otro archivo .csv para las funciones 3, 4 y 5.[Archivo para funciones 3, 4 y 5](https://github.com/juanbadan3/Proyecto_Individual_1/blob/main/df_f3_4_5.csv)


_**Análisis de sentimiento**_:

7.En el dataset australian_user_reviews, se llevó a cabo un análisis de sentimiento en las reseñas de juegos, creando así una nueva columna 'sentiment_analysis' con valores 0, 1 o 2 según la polaridad de la reseña (negativa, neutral o positiva). 

# _Funciones_
- _**Para obtener más información sobre el desarrollo de las diferentes funciones y una explicación más detallada de cada una, haga clic en el siguiente enlace**_
[Notebook de funciones](https://github.com/NPontisLedda/PI01_MLOPs_Henry/blob/main/FastAPI/fastapi-env/main.py)

Desarrollo API: Se propone disponibilizar los datos de la empresa usando el framework FastAPI . Las consultas que proponemos son las siguientes:

Cada aplicación tiene un decorador (@app.get('/')) con el nombre de la aplicación para poder reconocer su función.

Las consultas son las siguientes:

1. **developer(desarrollador: str, df)**:
Debe devolver la Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. Ejemplo de retorno:
Año	Cantidad de Items	Contenido Free
2023	50	27%
2022	45	25%
xxxx	xx	xx%

2. **userdata(User_id: str)**:
Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}

3. **UserForGenre(genero)**:
Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
4. **best_developer_year(año: int)**:
Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
5. **developer_reviews_analysis(devs:str)**:
Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
Ejemplo de retorno: {'Valve' : [Negative = 182, Positive = 278]}

# _**EDA (Análisis exploratorio de datos)**_
Ajustes en variables numéricas, específicamente corrigiendo valores atípicos en la columna playtime_forever, se realizaron durante el EDA.


# _**Aprendizaje automático(Machine Learning)**_

Se implementó un modelo de recomendación artículo-artículo utilizando K-Neighbours, explorando previamente modelos alternativos.

El método de aprendizaje automático utilizado es K-Neighbours. No es el mejor método para abordar los conjuntos de datos y parte de este proyecto se centra en eso. Debido a que el proyecto debe implementarse en Render, la memoria RAM disponible es limitada y lo importante aquí era comprender la diferencia entre los diferentes modelos de Machine Learning. Anteriormente, probé árboles de decisión y procesamiento de lenguaje natural utilizando similitud de coseno.

El sistema de recomendación item-item se planteó originalmente así:

6. **get_recommendations(item_id)**: 
Ingresando el id de producto(item_id), deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.



# _**Implementación de API**_
La API se desplegó mediante FastAPI en Render. Se puede acceder a la aplicación aquí para consumir los seis endpoints y realizar consultas sobre estadísticas de juegos.


Haga clic para acceder a mi aplicación FastAPI: [API Deployment](https://proyecto-individual-1-qwl5.onrender.com/docs)

Para consumir la API, utilice los 6 endpoints diferentes para obtener información y realizar consultas sobre estadísticas de juegos.



# Requisitos
- Python
- Scikit-Learn
- Pandas
- NumPy
- FastAPI
- nltk
- [Render](https://render.com/)

# _Autor_
- Juan David Albadan Alvarez
- Mail: juanalbadan3@gmail.com
- Linkedin: [Linkedin](https://www.linkedin.com/in/juan-david-albadan-689855216/)

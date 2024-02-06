# Importamos la librerías 

from fastapi import FastAPI
import pandas as pd
import json
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
# ----------------

app = FastAPI()
# ----------------
df_f1_2= pd.read_csv('df_f1_2.csv')
df_f3_4_5= pd.read_csv('df_f3_4_5.csv')
# ----------------
df_f3_4_5.drop(columns='Unnamed: 0',inplace=True)
df_f3_4_5.drop(columns='title',inplace=True)
df_f3_4_5.drop(columns='release_year',inplace=True)

df_f3_4_5['genres'] = df_f3_4_5['genres'].apply(lambda x: x.replace("'", "").strip("[]"))

df_f3_4_5.dropna(inplace=True)
lista=[]
for i in range(0,len(df_f3_4_5)):
    string = df_f3_4_5.iloc[i][6]

    try:
      b = int(string[-5:-1])
    except ValueError:
      b = float('nan') 

    lista.append(b)

df_f3_4_5['posted_year'] = lista
df_f3_4_5.dropna(inplace=True)
df_f3_4_5['posted_year'] = df_f3_4_5['posted_year'].astype('int')

df_f3_4_5.drop(columns='posted', inplace=True)
# --------------------
# Crear una instancia del codificador
label_encoder = LabelEncoder()

# Cargar los datos
df_ml1 = pd.read_csv('df_ML1.csv')
# Crear una nueva columna llamada genres_encoded, que tiene los generos codificados como int.
df_ml1["genres_encoded"] = label_encoder.fit_transform(df_ml1["genres"])

# Crear un diccionario de los títulos asociados a cada item_id
titles_by_item_id = {}
for i in range(len(df_ml1)):
    titles_by_item_id[df_ml1.loc[i, "item_id"]] = df_ml1.loc[i, "app_name"]

# Crear el modelo K-Nearest Neighbors
k = 5
model = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
model.fit(df_ml1[['genres_encoded']], df_ml1['app_name'])

# Guardar el modelo
with open('modelo1.pkl', 'wb') as f:
    pickle.dump(model, f)

# Guardar el diccionario
with open('titles_by_item_id.pkl', 'wb') as f:
    pickle.dump(titles_by_item_id, f)
#---------------------------------------

#http://127.0.0.1:8000/ 

@app.get("/")
def presentacion():
  return 'Hola, soy JUAN DAVID ALBADAN , formo parte de la cohorte DATAFT19 de Henry y este es el apartado de presentación de mi PI01 - MLOPs'
#-------------------------------------------
@app.get('/developer')
def developer(desarrollador: str, df):
     # Filtra el DataFrame para obtener solo las filas donde la columna 'developer' 
    df_desarrollador = df[df['developer'] == desarrollador]
   
    # Agrupa el DataFrame resultante por año de publicación ('posted_year') y cuenta la cantidad de 'item_id' en cada grupo
    df_total = df_desarrollador.groupby(['posted_year'])['item_id'].count().reset_index()
    # Renombra la columna 'item_id' a 'Cantidad de Items'
    df_total.rename(columns={'item_id': 'Cantidad de Items'}, inplace=True)
    # Filtra el DataFrame original para obtener solo las filas con precio igual a 0
    # Agrupa el resultado por año de publicación y cuenta la cantidad de 'item_id' en cada grupo
    df_free = df_desarrollador[df_desarrollador['price'] == 0].groupby(['posted_year'])['item_id'].count().reset_index()
    # Renombra la columna 'item_id' a 'Contenido Free'
    df_free.rename(columns={'item_id': 'Contenido Free'}, inplace=True)
    # Realiza una fusión (merge) de los DataFrames df_total y df_free en la columna 'posted_year', utilizando un left join
    # Llena los valores NaN con 0
    df_nuevo = pd.merge(df_total, df_free, on='posted_year', how='left').fillna(0)

    # Manejar NaN directamente en las operaciones
    df_nuevo['Contenido Free %'] = (df_nuevo['Contenido Free'].fillna(0) / df_nuevo['Cantidad de Items'].replace(0, 1) * 100).round().astype(int)
    # Convierte el DataFrame resultante a una lista de diccionarios, donde cada diccionario representa una fila del DataFrame
    lista = df_nuevo.to_dict(orient='records')
    
    return lista

#--------------------------------------------
@app.get('/userdata')
def userdata(User_id: str):
    # Filtrar el DataFrame por el usuario
    user_data = df_f3_4_5[df_f3_4_5['user_id'] == User_id]

    # Verificar si se encontraron registros para el usuario
    if user_data.empty:
        return {"Mensaje": "No se encontraron registros para el usuario especificado."}

    # Calcular el dinero gastado por el usuario
    dinero_gastado = user_data['price'].sum()

    # Calcular el porcentaje de recomendación
    porcentaje_recomendacion = (user_data['recommend'].sum() / len(user_data)) * 100

    # Calcular la cantidad de items
    cantidad_items = len(user_data)

    # Crear el diccionario de retorno
    resultado = {
        "Usuario": User_id,
        "Dinero gastado": f"{dinero_gastado} USD",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "Cantidad de items": cantidad_items
    }

    return resultado

#-----------------------------------------------
@app.get('/UserForGenre')
def UserForGenre(genero):

    # Filtrar el dataframe por género.
    df_filtrado = df_f1_2[df_f1_2["genres"].str.contains(genero)]

    # Calcular la acumulación de horas jugadas por usuario.
    df_acumulado = df_filtrado.groupby("user_id")["playtime_forever"].sum()

    # Obtener el usuario con más horas jugadas.
    usuario_mas_horas = df_acumulado.idxmax()

    # Calcular la acumulación de horas jugadas por año.
    df_acumulado_por_ano_1 = df_filtrado.groupby(["release_year"])["playtime_forever"].sum().to_frame()
    df_1 = df_acumulado_por_ano_1.add_suffix("_Sum").reset_index()

    # Convertir el dataframe a una lista de diccionarios.
    df_1 = df_1.rename(columns={"release_year": "Año", "playtime_forever_Sum": "Horas"})
    lista_acumulado_por_ano = df_1.to_dict(orient="records")
    lista_acumulado_por_ano
    # Devolver el resultado.
    return {
        "Usuario con más horas jugadas para Género X": usuario_mas_horas,
        "Horas jugadas": lista_acumulado_por_ano
    }

#--------------------------------------------
@app.get('/best_developer_year')
def best_developer_year(año: int):
    # Verificar si el año es igual a -1 y mostrar un mensaje personalizado
    if año == -1:
        return "El año ingresado es -1, lo cual no es válido."

    # Verificar que el año sea un número entero
    if not isinstance(año, int):
        return "El año debe ser un número entero."

    # Verificar que el año ingresado esté en la columna 'posted_year'
    if año not in df_f3_4_5['posted_year'].unique():
        return "El año no se encuentra en la columna 'posted_year'."

    # Filtrar el dataset para obtener solo las filas correspondientes al año dado
    juegos_del_año = df_f3_4_5[df_f3_4_5['posted_year'] == año]

    # Calcular la cantidad de recomendaciones para cada developer
    recomendaciones_por_juego = juegos_del_año.groupby('developer')['recommend'].sum().reset_index()

    # Ordenar los juegos por la cantidad de recomendaciones en orden ascendente
    devs_ordenados = recomendaciones_por_juego.sort_values(by='recommend')

    # Tomar los tres primeros lugares
    primer_puesto = devs_ordenados.iloc[0]['developer']
    segundo_puesto = devs_ordenados.iloc[1]['developer']
    tercer_puesto = devs_ordenados.iloc[2]['developer']

    # Crear el diccionario con los tres primeros lugares
    primeros_tres = {
        "Puesto 1": primer_puesto,
        "Puesto 2": segundo_puesto,
        "Puesto 3": tercer_puesto
    }

    return primeros_tres

#-----------------------------------------
@app.get('/developer_reviews_analysis')
def developer_reviews_analysis(devs:str):

    # Filtrar el DataFrame por la desarrolladora ingresada
    df_devs = df_f3_4_5[df_f3_4_5['developer'] == devs]

    # Verificar si se encontraron registros para el año
    if df_devs.empty:
        return {"Mensaje": "No se encontraron registros para la desarrolladora especificada."}

    # Contar la cantidad de registros para cada categoría de análisis de sentimiento
    developer_reviews_analysis_column = df_devs['sentiment_analysis']
    sentiment = developer_reviews_analysis_column.value_counts().to_dict()

    # Crear una lista con los resultados, tuve que colocar en formato de str todo para que me lo tomara como válido 
    resultado= list(["Negative= "+ str(sentiment.get(0, 0)), "Positive= "+ str(sentiment.get(2, 0))])

    #Crear el diccionario final
    result = {
        devs:resultado
    }

    return result

#-----------------------------------------
#Función 6, recomendación de juegos segun item_id indicado
@app.get('/get_recomendations')
def get_recommendations(item_id: int):

    # Cargar el modelo1 
    with open("modelo1.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar el diccionario de títulos
    with open("titles_by_item_id.pkl", "rb") as f:
        titles_by_item_id = pickle.load(f)

    # Buscar el género codificado del juego proporcionado por el usuario
    input_game = df_ml1[df_ml1["item_id"] == item_id]["genres_encoded"].values[0]

    # Encontrar los juegos más similares
    _, indices = model.kneighbors([[input_game]])

    # Obtener los títulos de los juegos similares
    similar_games = [titles_by_item_id[df_ml1.loc[i, "item_id"]] for i in indices[0]]

    # Devolver un diccionario de los títulos
    return {"similar_games": similar_games}

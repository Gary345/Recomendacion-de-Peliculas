# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 08:15:34 2022

@author: diana
"""
from rake_nltk import Rake  #libreria Rake para extraer y comparar palabras clave
import nltk #libreria de lectura de lenguaje natural
import pandas as pd #libreria para lectura de archivo csv
import numpy as np #libreria de analisis y calculo numerico
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #libreria para analisis estadistico
import warnings
warnings.filterwarnings("ignore")

class Usuario:
    P1=0
    P2=0
    P3=0
    P4=0
    P5=0
    Peliculas=[]
    
    def __init__(self,accion,drama,comedia,biografica,terror):
        self.P1=accion
        self.P2=drama
        self.P3=comedia
        self.P4=biografica
        self.P5=terror
#def test(genero):
    print('Responde las siguientes preguntas para darte una recomendacion de peliculas')
    print('ESCRIBE SOLO EL NUMERO DEL INCISO')
    print('¿Una buena pelicula de accion incluye disparos?')
    print('1)si')
    print('2)no')
    a=int(input())
    print('¿las peliculas basadas en hechos reales son interesantes?')
    print('1)si')
    print('2)no')
    b=int(input())
    print('¿las mejores peliculas de comedia son las de humor negro?')
    print('1)si')
    print('2)no')
    c=int(input())
    print('¿las peliculas de drama me ponen sentimental?')
    print('1)si')
    print('2)no')
    d=int(input())
    print('¿"El Resplandor" es la mejor pelicula de terror?')
    print('1)si')
    print('2)no, de suspenso')
    t=int(input())
    if (a==1):
        print('accion')#INSERTAR LISTA DE TRES PELICULAS DE ACCION
        df = pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')
        df = df[['Title','Director','Actors','Plot','Genre']]
        df['Plot'] = df['Plot'].str.replace('[^\w\s]','')
        #se usa la funcion Rake para extraer las palabras mas relevantes de todas las oraciones in la columna Plot se aplica la funcion para cada fila  debajo de la columna Plot y asignarlas a la lista de Key words  en una nueva columna
        df['Key_words'] = ''   # inicia una nueva columna
        r = Rake()   # usamos Rake para descartar palabras vacias o espacios en blanco

        for index, row in df.iterrows():
            r.extract_keywords_from_text(row['Plot'])   # extraer palabras clave del Plot, en minusculas por defecto 
            key_words_dict_scores = r.get_word_degrees()    # para obtener un diccionario con las palabras clave y sus puntuaciones
            row['Key_words'] = list(key_words_dict_scores.keys())   # para asignar la lista de claves a una columna
        df['Plot'][249]
        key_words_dict_scores
        # para extraer todos los titulos en una lista, solo los tres primeros actores en una lista y todos los directores en una lista
        df['Title'] = df['Title'].map(lambda x: x.split(','))
        df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
        df['Director'] = df['Director'].map(lambda x: x.split(','))

            # creamos una identidad unica nombrandolos por primer nombre y apellido y los convertimos a una sola palabra todo en minuscula 
        for index, row in df.iterrows():
            row['Title'] = [x.lower().replace(' ','') for x in row['Title']]
            row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
            row['Director'] = [x.lower().replace(' ','') for x in row['Director']]
    # para combinar 4 listas (4 columnas) de palabras clave en 1 oración en la columna Bag_of_words
        df['Bag_of_words'] = ''
        columns = ['Title', 'Director', 'Actors', 'Key_words']
        for index, row in df.iterrows():
            words = ''
            for col in columns:
                words += ' '.join(row[col]) + ' '
            row['Bag_of_words'] = words
            # eliminar los espacios en blanco delante y detrás, reemplazar varios espacios en blanco (si los hay)
        df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')
        # para generar la matriz de conteo
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['Bag_of_words'])
        count_matrix
    # para generar la matriz de similitud del coseno (tamaño 250 x 250)
    # filas representan todas las películas; las columnas representan todas las películas
    # similitud de coseno: similitud = cos (ángulo) = rango de 0 (diferente) a 1 (similar)
    # todos los números en la diagonal son 1 porque cada película es idéntica a sí misma (el valor del coseno es 1 significa exactamente idéntico)La matriz
    # también es simétrica porque la similitud entre A y B es la misma que la similitud entre B y A.
    # para otros valores, por ejemplo, 0,1578947, la película x y la película y tienen un valor de similitud de 0,1578947
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        # para crear una serie para títulos de películas por el genero que se pueden usar como índices (cada índice se asigna a un título de película)
        indices = pd.Series(df['Genre'])
        indices[:20]
  
    # esta función toma el genero de una película como entrada y devuelve las 10 mejores películas recomendadas (similares)  
        genre = ('Action')
        cosine_sim = cosine_sim
        recommended_movies = []
        idx = indices[indices == genre].index[0]   # para obtener el índice del título de la película que coincide con la película de entrada
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # puntajes similares en orden descendenter
        top_10_indices = list(score_series.iloc[1:11].index)   # para obtener los índices de las 10 películas más similares
            # [1:11] para excluir 0 (el índice 0 es la película de entrada en sí)
        for i in top_10_indices:    # para agregar los títulos de las 10 mejores películas similares a la lista de películas recomendadas
            recommended_movies.append(list(df['Genre'])[i])
    print(f"Te recomiendo estas peliculas segun el genero:\n",recommended_movies)
    if b==1:
        print('biografica') #INSERTAR LISTA DE TRES PELICULAS BIOGRAFICAS
    if c==1:
        print('comedia') #INSERTAR LISTA DE TRES PELICULAS DE COMEDIA
    if d==1:
        print('drama') #INSERTAR LISTA DE TRES PELICULAS DE DRAMA
    if t==1:
        print('terror') #INSERTAR LISTA DE TRES PELICULAS DE TERROR
    else:
        print('suspenso') #INSERTAR LISTA DE TRES PELICULAS DE suspenso


#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Boris Panes</em><br>
# 
# </div>
# 
# ### Análisis de Datos: Tarea 01
# #### Integrantes: 
# ` Mariangel Arrieta, Giuseppe Lavarello, Ingrid Solís, Rosario Valderrama `

# #### 1. Importación de Librerias

import pandas as pd               #Manejo de datos en Tablas
import numpy as np
import matplotlib.pyplot as plt   #Creación de visualizaciones 
import seaborn as sns             #Creación de visualizaciones  
import missingno as msgn          #Visualizacion de NaNs 
from sklearn.preprocessing import MinMaxScaler #Para Normalizar la data
from sklearn.preprocessing import StandardScaler #Para Estandarizar la data

sns.set_theme() #inicializacion de tematica de seaborn


# #### 2. Lectura de archivo

# Ruta relativa al archivo
ruta_archivo = r'.\Data\2023-03-08 Precios Casas RM.csv'
df = pd.read_csv(ruta_archivo)
df.columns = df.columns.str.replace(' ', '_') # Normalizar los nombres
df.columns = df.columns.str.lower()
# Mostrar las primeras filas del DataFrame
df.head()


# #### 3. Análisis exploratorio

# Recopilación de información básica sobre el conjunto de datos
df.info()


# Resumen estadistico de las variables numericas
df.describe()


# Dimensiones del df
df.shape


# Total de elementos únicos por cada columna.
df.nunique()


# ### 4. Limpieza de datos
# ##### 4.1. Busqueda de Valores Nulos

# Se verifica la cantidad de datos faltantes
df.isna().sum()


# Se visualiza los Nulos
msgn.matrix(df)


# ##### 4.2 Protocolo de acción con respecto a los Nulos
# **Parking**

df[df["parking"]==0] # Se busca las propiedades sin Parking


# 
# **Decisión:** 
# 
# Dado que no se encontraron valores numéricos iguales a 0 en la característica "parking", se concluye que los valores NaN en esta misma característica pueden ser reemplazados por 0. 
# 
# **Justificación:** 
# * La ausencia de ceros sugiere que NaN en este contexto indica la falta de datos o un valor no aplicable, que puede ser interpretado como una ausencia de la característica a la hora de realizar el web scraping.
# 

df["parking"].fillna(0, inplace=True) # Se remplazan los NaNs de parking por 0


# **Realtor**

df.realtor.dtype


# **Decisión:** Se ha tomado la determinación de excluir la columna "realtor" del modelo de regresión lineal. 
# 
# **Justificación:**
# * **Naturaleza categórica:** La variable "realtor" es de naturaleza categórica (nominal), lo cual dificulta su incorporación directa en un modelo de regresión lineal.
# * **Valores faltantes:** La presencia significativa de valores NaN en esta columna podría afectar la precisión de las predicciones.
# * **Irrelevancia para la predicción:** Se considera que la identidad del realtor no tiene una relación causal directa con la variable objetivo que se busca predecir en este análisis.
# 
# **Implicaciones:**
# * Al eliminar esta columna, se simplifica el modelo y se reduce el riesgo de overfitting.
# * Seria valido considerar si la información contenida en "realtor" podría ser relevante para otros análisis, como por ejemplo, un estudio exploratorio de los diferentes realtors, sus areas de trabajo o un modelo de clasificación.

df.drop("realtor", axis=1, inplace=True) # Se descarta la columna Realtor


df.isna().sum()/df.shape[0]*100 # Se calcula el % de los datos que quedan nulos


# **Decisión:** Se decide eliminar los valores nulos restantes en esta etapa inicial del análisis.
# 
# **Justificación:**
# * **Baja proporción:** La proporción de valores nulos es relativamente pequeña en comparación con el tamaño total del dataset. (0.8 al 3%)
# * **Primera iteración:** En esta primera iteración, el objetivo es obtener un modelo inicial para luego realizar ajustes posteriores.
# * **Impacto limitado:** Se considera que la eliminación de estos nulos tendrá un impacto limitado en la precisión y generalización del modelo.
# 
# **Consideraciones futuras:**
# * En futuras iteraciones, se puede explorar métodos de imputación de valores nulos para mejorar la calidad del dataset y la robustez del modelo.

df.dropna(inplace=True) #se descarta el resto de los Nans


msgn.matrix(df)


# ##### 4.3 Distinción de Tipos

df.info()


# |Variable|	Tipo|	Descripción|
# |---|---|---|
# |price_clp|	Numérico (entero)|	Precio en pesos chilenos|
# |price_uf|	Numérico (entero)|	Precio en Unidades de Fomento|
# price_usd|	Numérico (entero)|	Precio en dólares estadounidenses|
# |dorms|	Numérico (entero)|	Número de dormitorios|
# |baths|	Numérico (flotante)|	Número de baños|
# |built_area|	Numérico (flotante)|	Área construida (m²)|
# |total_area|	Numérico (flotante)|	Área total (m²)|
# |parking|	Numérico (flotante)|	Número de estacionamientos|
# |id|	Categórico Numérico|	Identificador único interno|
# |comuna|	Categórico|	Nombre de la comuna|
# |ubicacion|	Categórico|	Ubicación específica|

# ##### 4.4 Busqueda y manejo de duplicados

# Verificar datos duplicados
duplicados = df.duplicated().sum()

# Porcentaje de data duplicada
porcentaje = df.duplicated().sum() / df.shape[0] * 100

print(f'{duplicados} el numero de filas duplicadas representa {porcentaje.round(2)}% del total de la data.')


# Borramos dato duplicado y creamos un nuevo df1
df1 = df.drop_duplicates(keep='first').copy()
df1.drop("id",axis=1, inplace=True)
# Mostramos las primeras filas del df1
df1.head()


# ##### 4.5 Normalización

# Se selecciona las columnas numéricas
num_columns = df1.select_dtypes(include=np.number)

# Se normaliza los valores usando min-max scaling
minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(num_columns)

# Se crea df con datos normalizados
df1_normalized = pd.DataFrame(normalized_data, columns=num_columns.columns)
df1_normalized.head(10)


# ##### 4.6 Estandarización

# Estandarizamos la data
std_scaler = StandardScaler()
std_data = std_scaler.fit_transform(num_columns) # num_columns es las columnas numericas de df1

# Crear df con datos normalizados
df1_std = pd.DataFrame(std_data, columns=num_columns.columns)
df1_std.head(10)


# ##### 4.6 Ingenieria de columnas

# **Decisión:** Será de poco uso considerar las 3 monedas para analizar, por lo que solo se utilizará la UF para Ingeniería de columnas.

df2=df1.copy()
df2["price_built_m2_uf"] = df2["price_uf"]/df2["built_area"] 
df2["price_total_m2_uf"] = df2["price_uf"]/df2["total_area"]
df2["price_parking_uf"] = df2["price_uf"]/df2["parking"]
df2["price_dorms_uf"] = df2["price_uf"]/df2["dorms"]
df2["price_baths_uf"] = df2["price_uf"]/df2["baths"]


# * **price_built_m2_uf**  
#     - Descripción: Precio por metro cuadrado construido en Unidades de Fomento (UF).  
#     - Cálculo: Se divide el precio total de la propiedad en UF por su área construida.  
#     - Significado: Esta variable proporciona una medida del valor de la propiedad por unidad de área construida, lo que puede ser útil para comparar propiedades de diferentes tamaños.
# 
# * **price_total_m2_uf**  
#     - Descripción: Precio por metro cuadrado total en Unidades de Fomento (UF).  
#     - Cálculo: Se divide el precio total de la propiedad en UF por su área total.  
#     - Significado: Esta variable proporciona una medida del valor de la propiedad por unidad de área total, incluyendo áreas comunes como estacionamientos y jardines.  
# 
# * **price_parking_uf**  
#     - Descripción: Precio por estacionamiento en Unidades de Fomento (UF).  
#     - Cálculo: Se divide el precio total de la propiedad en UF por el número de estacionamientos.  
#     - Significado: Esta variable proporciona una medida del valor de cada estacionamiento en la propiedad.  
# 
# * **price_dorms_uf**  
#     - Descripción: Precio por dormitorio en Unidades de Fomento (UF).  
#     - Cálculo: Se divide el precio total de la propiedad en UF por el número de dormitorios.  
#     - Significado: Esta variable proporciona una medida del valor de cada dormitorio en la propiedad.  
# 
# * **price_baths_uf**  
#     - Descripción: Precio por baño en Unidades de Fomento (UF).  
#     - Cálculo: Se divide el precio total de la propiedad en UF por el número de baños.  
#     - Significado: Esta variable proporciona una medida del valor de cada baño en la propiedad.  

df2.head()


# Definir el tamaño de la figura
fig, axs = plt.subplots(nrows=4, ncols=2,figsize=(15, 20), squeeze=False)  # Crear una cuadrícula de gráficos (4 filas, 2 columnas)
# Iterar sobre cada columna numérica para crear Histograma
for ax, column in zip(axs.flat, num_columns.columns):
    
    sns.histplot(num_columns[column],ax=ax, stat="density") #si les va muy lento borrar stat="density" y poner bins=40 o algo asi
    sns.kdeplot(num_columns[column],ax=ax)
    ax.set_title(f'Histograma de {column}')

# Ajustar el espacio entre los gráficos
plt.tight_layout
plt.show()


# Crear un boxplot para visualizar posibles Outliers

plt.figure(figsize=(10,8))
sns.boxplot(data= df1_normalized)
plt.xticks(rotation=45)


plt.show()


# Calcular la matriz de correlación
correlation_matrix = num_columns.corr(numeric_only=True)

# Crear un mapa de calor para mostrar la correlación
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de correlación')
plt.show()


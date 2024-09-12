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
df.head(10)


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
# ##### 4.1. Verificamos nombre de comunas.

# Obtener el listado único de comunas
comunas_unicas = df['comuna'].unique()

# Ordenar alfabéticamente para facilitar la revisión
comunas_unicas.sort()

# Mostrar el listado de comunas únicas
print("Listado de comunas únicas:")
print(comunas_unicas)


# ##### 4.2. Busqueda de Valores Nulos

# Filtrar y mostrar las filas con NaN
nan_rows = df[df.isna().any(axis=1)]
print("Listado de filas con NaN:")
nan_rows.head(10)


# Contar cuántos NaN hay en cada columna
nan_count = df.isna().sum()
print("Valores NaN por columna:")
print(nan_count)


# Se visualiza los Nulos
msgn.matrix(df)


# ##### 4.3 Protocolo de acción con respecto a los Nulos
# ### **a) Parking**

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


# ### **b) Realtor**

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


# ### **c) Valores NaN que quedan:** baths (65), built_area (246), total_area (208)

# **Decisión:** Se ha tomado la determinación de reemplazar los valores NaN restantes, pero normalizando con la mediana cada uno de ellos según valor total de la columna dorms y comuna.
# 
# **Justificación:**
# 
# 
# **Implicaciones:**
# 

# Visualizamos la cantidad de casas que tienen 1 o más valores NaN en las columnas de baths, built_area, total_area

# Contar cuántos NaN hay en cada comuna (solo casas únicas que tienen al menos un NaN)
nan_by_comuna = df[df.isna().any(axis=1)].groupby('comuna').size().reset_index(name='casas_con_nan')

# Contar el total de casas por comuna
total_casas_por_comuna = df.groupby('comuna').size().reset_index(name='total_casas')

# Unir los dos resultados en un solo DataFrame
comparacion = pd.merge(total_casas_por_comuna, nan_by_comuna, on='comuna', how='left')

# Reemplazar NaN en 'casas_con_nan' con 0, en caso de que algunas comunas no tengan casas con NaN
comparacion['casas_con_nan'].fillna(0, inplace=True)

# Calcular el porcentaje de casas con valores NaN respecto al total de casas en cada comuna
comparacion['porcentaje_nan'] = (comparacion['casas_con_nan'] / comparacion['total_casas']) * 100

# Mostrar el resultado
print(comparacion)


# Hay algunos valores NaN que tienen un % considerable dentro del total de la comuna que podrían cambiar los datos si no los tomamos en cuenta. Por ejemplo en comunas: Cerro Navia, Conchalí, Curacaví, La Granja, Pirque y San Ramón (sobre 10%). 

# Tenemos que tener cuidado para reemplazar por normalización según la media (ya que hay outliers) y también según la mediana, ya que no podemos comparar casas de 7 dorms con casas de 1 dorm. Entonces, lo que haremos es buscar la mediana para cada casa según comuna y según cantidad de dorms.

# Calcular la mediana por comuna y número de dormitorios para 'baths', 'built_area', y 'total_area'
medianas_por_comuna_dorms = df.groupby(['comuna', 'dorms'])[['baths', 'built_area', 'total_area']].median().reset_index()

# Mostrar la tabla con las medianas por comuna y número de dormitorios
print(medianas_por_comuna_dorms)


# Verificar cuántos NaN quedan en las columnas baths, built_area, y total_area
nan_remaining = df[['baths', 'built_area', 'total_area']].isna().sum()

# Mostrar el resultado para identificar si hay NaN restantes
nan_remaining


# Calcular las medianas de baths, built_area, total_area por comuna y cantidad de dorms
medianas_por_comuna_dorms = df.groupby(['comuna', 'dorms'])[['baths', 'built_area', 'total_area']].median().reset_index()

# Reemplazar los valores NaN en el DataFrame original según la mediana de cada comuna y cantidad de dorms
for i, row in medianas_por_comuna_dorms.iterrows():
    # Crear la máscara para seleccionar las filas con NaN en 'baths', 'built_area', 'total_area' en la comuna y dorms específicos
    mask = (df['comuna'] == row['comuna']) & (df['dorms'] == row['dorms'])
    
    # Reemplazar NaN en 'baths'
    df.loc[mask & df['baths'].isna(), 'baths'] = row['baths']
    
    # Reemplazar NaN en 'built_area'
    df.loc[mask & df['built_area'].isna(), 'built_area'] = row['built_area']
    
    # Reemplazar NaN en 'total_area'
    df.loc[mask & df['total_area'].isna(), 'total_area'] = row['total_area']

# Verificar que los valores NaN hayan sido reemplazados
print(df[['baths', 'built_area', 'total_area']].isna().sum())


# Eliminar las filas donde hay valores NaN en 'baths', 'built_area', o 'total_area'
df_sin_nan = df.dropna(subset=['baths', 'built_area', 'total_area'])

# Verificar que ya no hay valores NaN en las columnas mencionadas
print(df_sin_nan[['baths', 'built_area', 'total_area']].isna().sum())


# Guardar el DataFrame sin valores NaN en un nuevo DataFrame
df_nuevo_normalizado = df_sin_nan.copy()

# Verificar que el nuevo DataFrame ha sido creado
print(f"El nuevo DataFrame tiene {df_nuevo_normalizado.shape[0]} filas y {df_nuevo_normalizado.shape[1]} columnas.")

# Si deseas guardarlo en un archivo Excel para futuras referencias
ruta_guardado = r'.\Data\df_nuevo_normalizado.xlsx'
df_nuevo_normalizado.to_excel(ruta_guardado, index=False)

print(f"Archivo guardado en: {ruta_guardado}")


# **Decisión:** Se decide eliminar los valores nulos restantes en esta etapa inicial del análisis.
# 
# **Justificación:**
# * **Baja proporción:** La proporción de valores nulos es relativamente pequeña en comparación con el tamaño total del dataset. (0.8 al 3%)
# * **Primera iteración:** En esta primera iteración, el objetivo es obtener un modelo inicial para luego realizar ajustes posteriores.
# * **Impacto limitado:** Se considera que la eliminación de estos nulos tendrá un impacto limitado en la precisión y generalización del modelo.
# 
# **Consideraciones futuras:**
# * En futuras iteraciones, se puede explorar métodos de imputación de valores nulos para mejorar la calidad del dataset y la robustez del modelo.

#df.dropna(inplace=True) #se descarta el resto de los Nans


df_nuevo_normalizado.isna().sum()/df.shape[0]*100 # Se calcula el % de los datos que quedan nulos


# Cargar el archivo Excel proporcionado
df_nuevo = pd.read_excel(ruta_guardado)

# Mostrar las primeras filas del DataFrame para verificar que el archivo se ha cargado correctamente
df_nuevo.head()


msgn.matrix(df_nuevo)


# ##### 4.4 Distinción de Tipos

df_nuevo.info()


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

# ##### 4.5 Busqueda y manejo de duplicados

# Verificar datos duplicados
duplicados = df_nuevo.duplicated().sum()

# Porcentaje de data duplicada
porcentaje = df_nuevo.duplicated().sum() / df_nuevo.shape[0] * 100

print(f'{duplicados} el numero de filas duplicadas representa {porcentaje.round(2)}% del total de la data.')


# Borramos dato duplicado y creamos un nuevo df1
df1 = df_nuevo.drop_duplicates(keep='first').copy()
df1.drop("id",axis=1, inplace=True)
# Mostramos las primeras filas del df1
df1.head()


# ##### 4.6 Normalización

# Se selecciona las columnas numéricas
num_columns = df1.select_dtypes(include=np.number)

# Se normaliza los valores usando min-max scaling
minmax_scaler = MinMaxScaler()
normalized_data = minmax_scaler.fit_transform(num_columns)

# Se crea df con datos normalizados
df1_normalized = pd.DataFrame(normalized_data, columns=num_columns.columns)
df1_normalized.head(10)


# ##### 4.7 Estandarización

# Estandarizamos la data
std_scaler = StandardScaler()
std_data = std_scaler.fit_transform(num_columns) # num_columns es las columnas numericas de df1

# Crear df con datos normalizados
df1_std = pd.DataFrame(std_data, columns=num_columns.columns)
df1_std.head(10)


# ##### 4.8 Ingenieria de columnas

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


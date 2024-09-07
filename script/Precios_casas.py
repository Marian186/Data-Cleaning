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
# ` Mariangel Arrieta Giuseppe Lavarello Ingrid Solís Rosario Valderrama `

# 1. Importación de Librerias

import pandas as pd               #Manejo de datos en Tablas
import numpy as np
import matplotlib.pyplot as plt   #Creación de visualizaciones 
import seaborn as sns             #Creación de visualizaciones   


# 2. Cargar archivo

# Ruta relativa al archivo
ruta_archivo = r'.\Data\2023-03-08 Precios Casas RM.csv'
df = pd.read_csv(ruta_archivo)

# Mostrar las primeras filas del DataFrame
df.head()


# 3. Análisis exploratorio

#información general
df.info()


#resumen estadistico de las variables numericas
df.describe()


#Dimensiones del df
df.shape


#Total de elementos únicos por cada columna.
df.nunique()


# 4. Limpieza de datos

#Verificar datos faltantes
df.isna().sum()


# Verificar datos duplicados
duplicados = df.duplicated().sum()

# Porcentaje de data duplicada
porcentaje = df.duplicated().sum() / df.shape[0] * 100

print(f'{duplicados} el numero de filas duplicadas representa {porcentaje.round(2)}% del total de la data.')


# Borramos dato duplicado y creamos un nuevo df1
df1 = df.drop_duplicates(keep='first')

# Mostramos las primeras filas del df1
len(df1)


# Histograma de todas las columnas.
# Definir el tamaño de la figura para los histogramas
df1.hist(bins=15, figsize=(15, 10), layout=(5, 3), edgecolor='black')

# Ajustar el espacio entre los gráficos
plt.tight_layout()
plt.show()


# Asegurarse de que las columnas sean numéricas
df1_numeric = df1.select_dtypes(include='number')

# Eliminar la columna 'ID'
df1_numeric = df1_numeric.drop(columns=['id'])

# Definir el tamaño de la figura para los gráficos KDE
plt.figure(figsize=(15, 20))

# Iterar sobre cada columna numérica para crear gráficos KDE
for i, column in enumerate(df1_numeric.columns, 1):
    plt.subplot(7, 2, i)  # Crear una cuadrícula de gráficos (7 filas, 2 columnas)
    sns.kdeplot(df1_numeric[column], fill=True)
    plt.title(f'Gráfico KDE de {column}')

# Ajustar el espacio entre los gráficos
plt.tight_layout()
plt.show()


#Verificamos Outliers

# diagrama de caja para visualizar la distribución de todas las variables numéricas

# normalizar la escala
from sklearn.preprocessing import MinMaxScaler

# seleccionar columnas numéricas
num_columns = df[['Price_CLP', 'Price_UF', 'Price_USD', 'Dorms', 'Baths' , 'Built Area' , 'Total Area', 'Parking']]

#normalize values using min-max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(num_columns)

# Crear df con datos normalizados
df_normalized = pd.DataFrame(normalized_data, columns=num_columns.columns)

sns.boxplot(data= df_normalized)
plt.xticks(rotation=45)

plt.show()


# Calcular la matriz de correlación
correlation_matrix = df1_numeric.corr(numeric_only=True)

# Crear un mapa de calor para mostrar la correlación
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de correlación')
plt.show()


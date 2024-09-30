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
# ### Análisis de Datos: Tarea 02
# #### Integrantes: 
# ` Mariangel Arrieta, Giuseppe Lavarello, Ingrid Solís, Rosario Valderrama `

# #### 1. Importes

import pandas as pd               #Manejo de datos en Tablas
import numpy as np
from  matplotlib import pyplot as plt   #Creación de visualizaciones 
import seaborn as sns             #Creación de visualizaciones  
import statsmodels.api as sm

sns.set_theme() #inicializacion de tematica de seaborn


# #### 2. Lectura de archivo

# Ruta relativa al archivo
ruta_archivo = r'.\Data\2023-03-08 Precios Casas RM Limpio.csv'
df = pd.read_csv(ruta_archivo, index_col=0)
# Mostrar las primeras filas del DataFrame
df.head(10)


# **Desición** Se eliminal las columnas price_clp y price_usd pues son colineales con price_uf

df.drop(['price_clp','price_usd'],axis=1 ,inplace=True)
df.head()


# #### 3. Teoria detras del modelo

# **Ecuación de Regresión Lineal |** $y = \beta_0 + \beta_1 x + \epsilon$
# 
# Los parámetros son propiedades de las poblaciones, por lo que nunca podemos conocer sus valores verdaderos a menos que se observe toda la población.
# 
# - Las estimaciones de los parámetros se calculan a partir de datos muestrales.
# - Las estimaciones se denotan con un ^ sombrero.
# 
# **Estimación de la Regresión Lineal |** $\hat{y} = \hat{\beta_0} + \hat{\beta_1} x + \epsilon$
# 
# **Coeficientes de Regresión |** Los betas estimados en un modelo de regresión. Se representan como $\hat{\beta_i}$.
# 
# **Estimación de Mínimos Cuadrados Ordinarios $MCO$ |** Método común para calcular los coeficientes de regresión lineal $\hat{(\beta)}_n$
# 
# **Función de Pérdida |** Una función que mide la distancia entre los valores observados y los valores estimados por el modelo.

# ##### 3.1 Estimacion de Minimos Cuadrados Ordinarios

# El método de mínimos cuadrados ordinarios $MCO$ se utiliza en el análisis de regresión lineal para estimar los parámetros desconocidos del modelo de regresión lineal. El objetivo de la estimación por $MCO$ es encontrar los valores de los coeficientes de regresión que minimicen la suma de los errores al cuadrado entre los valores predichos y los valores reales de la variable dependiente.
# 
# **Línea de Mejor Ajuste |** La línea que ajusta mejor los datos al minimizar alguna función de pérdida o error.
# 
# **Valores Predichos |** Los valores estimados $y$ para cada $x$ calculados por un modelo.
# 
# **Residuo |** La diferencia entre los valores observados o reales y los valores predichos de la línea de regresión.
# - Residuo = Observado - Predicho $\rightarrow$ $\epsilon_i = y_i - \hat{y_i}$
# 
# **Suma de Residuos al Cuadrado (RSS) |** La suma de las diferencias al cuadrado entre cada valor observado y su valor predicho asociado.
# - $RSS = \sum\limits_{i=1}^{n}(Observado - Predicho)^2$
# - $RSS = \sum\limits_{i=1}^{n}(y_i - \hat{y_i})^2$
# 
# **Mínimos Cuadrados Ordinarios (MCO) |** Un método que minimiza la suma de los residuos al cuadrado para estimar los parámetros en un modelo de regresión lineal.
# - Usado para calcular: $\hat{y}=\hat{\beta_0} + \hat{\beta_1(x)}$
# - Donde se obtiene que : $\hat{\beta_1} = \frac{\sum\limits_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sum\limits_{i=1}^{n}(X_i-\bar{X})^2} = \frac{cov_{x,y}}{var_x}$
# - $\hat{\beta_0} = \bar{Y} - \hat{\beta_1}\bar{X}$

# ##### 3.2 Evaluación del ajuste
# 
# **Error Stándard de la Regresión $ESR$ |** Parámetro utilizado para evaluar la regresión  
# - $ESR = s_e$, donde $s_e^2=\frac{\sum{e_i}^2}{n-2} $
# 
# Las unidades de $e_i$ son las mismas que las de $Y_i$, por lo tanto, el valor de $ESR$ proporciona información sobre el error promedio entre las predicciones y los valores observados.  
# $ESR$ es el promedio de la suma de los residuos al cuadrado $RSS$ ajustado por los grados de libertad.
# 
# **Coeficiente de Determinación $R^2$ |** Medida estadística que indica el porcentaje de variación en la variable dependiente que es explicada por el modelo de regresión.
# - $R^2$ toma valores entre 0 y 1. Un valor de $R^2$ cercano a 1 indica que el modelo explica bien la variabilidad de los datos, mientras que un valor cercano a 0 indica que el modelo no lo hace.
#   
# - Se calcula como:
# 
# $R^2 = 1 - \frac{SR}{ST}$ 
# 
# Donde:
# - $SR$ es la suma de los residuos al cuadrado (suma de los errores entre los valores observados y predichos):
#     - $SR = \sum\limits_{i=1}^{n}e_i^2$
# - $ST$ es la suma total de los cuadrados (variabilidad total en los datos observados):
#     - $ST = \sum\limits_{i=1}^{n}(Y_i - \bar{Y})^2$
# 
# En este caso particular, se cumple que:
# - $R^2 = r^2 = \frac{Cov^2(X,Y)}{Var(X)Var(Y)}$
# 

# #### 4 Selección de variables

sns.pairplot(df.select_dtypes(include=np.number),diag_kind=None)

plt.show()


# **Hipótesis:** "El área construida y el número de dormitorios tienen una relación positiva y significativa con el precio de las casas en UF en la Región Metropolitana."
# 
# Variables a considerar para la regresión lineal:
# 
# - Variable dependiente: Precio en UF (`Price_UF`).
# - Variables independientes:
#     - Número de dormitorios (`Dorms`).
#     - Área construida (`Built_Area`).
# 
# Esta hipótesis plantea que, a medida que aumentan el área construida y el número de dormitorios, el precio de las casas en UF también aumentará.
# 

# ##### 4.1 Análisis de regresión basado en el número de dormitorios
# - 4.1.1 Selección de datos

# Seleccionamos las columnas relevantes

# Guardamos el DataFrame resultante en una variable separada para la regresión

rl_data_dorms = df[["price_uf", "dorms"]]

# Primeras 5 filas
rl_data_dorms.head()


ax = sns.scatterplot(data=rl_data_dorms, x='dorms', y='price_uf')
ax.set_title("Gráfica de dispersión")
ax.set_xlabel("Número de dormitorios")
ax.set_ylabel("Precio en UF")
plt.show()


fig, axs = plt.subplots(1,2,figsize=(12,4))
sns.boxplot(data=rl_data_dorms['dorms'],ax=axs[0])
sns.boxplot(data=rl_data_dorms[(rl_data_dorms['dorms']<10)& (rl_data_dorms['price_uf']<100000)]['dorms'],ax=axs[1])
plt.setp(axs, ylabel="Número de dormitorios")
plt.show()


# **Decisión:** Dado que las casas con demasiados dormitorios distorsionan los datos, se elegirán como límites las casas con 10 dormitorios y un precio menor a 100.000 UF.
# 

rl_data_dorms = rl_data_dorms[(rl_data_dorms['dorms']<15)& (rl_data_dorms['price_uf']<100000)]
ax = sns.scatterplot(data=rl_data_dorms, x='dorms', y='price_uf')
ax.set_title("Gráfica de dispersión")
ax.set_xlabel("Número de dormitorios")
ax.set_ylabel("Precio en UF")
plt.show()


# - 4.1.2 Cálculo de la matriz de Covarianza

#Calcular la matriz de covarianza
cov_mat_dorms = rl_data_dorms.cov()
cov_mat_dorms


# - 4.1.3 Cálculo de los Parámetros de la Regresión  
#     - Recordamos que:  
# $\hat{\beta_1} = \frac{\sum\limits_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sum\limits_{i=1}^{n}(X_i-\bar{X})^2} = \frac{cov_{x,y}}{var_x}$

b_1 = cov_mat_dorms.iloc[0,1]/cov_mat_dorms.iloc[1,1]
print("El valor de b_1 es:", b_1)


# $\hat{\beta_0} = \bar{Y} - \hat{\beta_1}\bar{X}$

b_0 = rl_data_dorms.price_uf.mean() - b_1*rl_data_dorms.dorms.mean()
print("El valor de b_0 es:", b_0)


# Crear el dataframe con los valores predichos y sus errores
rl_final_dorms=rl_data_dorms.copy()
rl_final_dorms['predict']=b_1*rl_data_dorms.dorms + b_0
rl_final_dorms['error']=rl_final_dorms.price_uf - rl_final_dorms.predict
rl_final_dorms.head()


# - 4.1.4 Cálculo de las Evaluaciones del Ajuste  
#     - Calculamos el $ESR$:  
# $ESR = s_e,\ donde\ s_e^2=\frac{\sum{e_i}^2}{n-2} $
# 
# 

s_e2 = (rl_final_dorms.error**2).sum()/(len(rl_final_dorms.error)-2)
s_e = np.sqrt(s_e2)
print("El valor de ESR es:", s_e)


# Calculamos el $R^2$  
#     $R^2 = \frac{Cov^2(X,Y)}{Var(X)Var(Y)}$

r2 = cov_mat_dorms.iloc[0,1]**2/(cov_mat_dorms.iloc[0,0]*cov_mat_dorms.iloc[1,1])
print("El valor de r2 es:", r2)


# Comprobamos la igualdad de 
# $R^2 = 1 - \frac{SR}{ST}$ 

SR = (rl_final_dorms.error**2).sum()
ST = ((rl_final_dorms.price_uf - rl_final_dorms.price_uf.mean())**2).sum()
r22 = 1-SR/ST
print("El valor de r2 es:", r22)


# - 4.1.5 Gráficas de Interés

fig, ax = plt.subplots()
sns.scatterplot(data=rl_final_dorms, y='price_uf', x='dorms',ax=ax)
sns.lineplot(data=rl_final_dorms, x='dorms', y='predict', color='orange',ax=ax)
ax.set_title("Predicción vs realidad")
ax.set_xlabel("Número de dormitorios")
ax.set_ylabel("Precio en UF")
plt.show()


residuals=rl_final_dorms.error
fig = sns.histplot(residuals)
fig.set_xlabel("Valor del Error")
fig.set_ylabel("Cantidad")
fig.set_title("Histograma del Error")
plt.show()


fig, ax = plt.subplots(figsize=(6,4))
sm.qqplot(residuals, line='s',ax= ax)
plt.title("Q-Q plot de errores")
ax.set_xlabel("Cuantiles Teoricos")
ax.set_ylabel("Cuantiles de la muestra")
plt.show()


# 
# ##### 4.2 Análisis de Regresión Basado en el Área Construida
# - 4.2.1 Selección de Datos
# 

# Seleccionamos las columnas relevantes

# Guardamos el DataFrame resultante en una variable separada para la regresión

rl_data_built = df[["price_uf", "built_area"]]

# Primeras 5 filas
rl_data_built.head()


ax = sns.scatterplot(data=rl_data_built, x='built_area', y='price_uf')
ax.set_title("Gráfica de dispersión")
ax.set_xlabel("Metros Cuadrados Construidos")
ax.set_ylabel("Precio en UF")
plt.show()


fig, axs = plt.subplots(1,2,figsize=(12,4))
sns.boxplot(data=rl_data_built['built_area'],ax=axs[0])
sns.boxplot(data=rl_data_built[rl_data_built['built_area']<600]['built_area'],ax=axs[1])
plt.setp(axs, ylabel="Metros Cuadrados Construidos")
plt.show()


# **Decisión:** Dado que existe una gran cantidad de outliers, se tomará un conjunto de datos reducido a las casas con menos de 600 $m^2$ construidos y valores bajo las 100.000 UF.

sns.scatterplot(data=rl_data_built[(rl_data_built['built_area']<600) & (rl_data_built['price_uf']<100000)], x='built_area', y='price_uf')
plt.show()


rl_data_built=rl_data_built[(rl_data_built['built_area']<600) & (rl_data_built['price_uf']<100000)]
rl_data_built.head()


# - 4.2.2 Cálculo de la matriz de Covarianza

#Calcular la matriz de covarianza
cov_mat_built = rl_data_built.cov()
cov_mat_built


# - 4.2.3 Cálculo de los Parámetros de la Regresión  
#     - Recordamos que:  
# $\hat{\beta_1} = \frac{\sum\limits_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}{\sum\limits_{i=1}^{n}(X_i-\bar{X})^2} = \frac{cov_{x,y}}{var_x}$

b_1_built = cov_mat_built.iloc[0,1]/cov_mat_built.iloc[1,1]
print("El valor de b_1 es:", b_1_built)


# $\hat{\beta_0} = \bar{Y} - \hat{\beta_1}\bar{X}$

b_0_built = rl_data_built.price_uf.mean() - b_1_built*rl_data_built.built_area.mean()
print("El valor de b_0 es:", b_0_built)


# Crear el dataframe con los valores predichos y sus errores
rl_final_built=rl_data_built.copy()
rl_final_built['predict']=b_1_built*rl_final_built.built_area + b_0_built
rl_final_built['error']=rl_final_built.price_uf - rl_final_built.predict
rl_final_built.head()


# - 4.2.4 Cálculo de las Evaluaciones del Ajuste  
#     - Calculamos el $ESR$:  
# $ESR = s_e,\ donde\ s_e^2=\frac{\sum{e_i}^2}{n-2} $
# 
# 

s_e2 = (rl_final_built.error**2).sum()/(len(rl_final_built.error)-2)
s_e = np.sqrt(s_e2)
print("El valor de ESR es:", s_e)


# Calculamos el $R^2$  
#     $R^2 = \frac{Cov^2(X,Y)}{Var(X)Var(Y)}$

r2 = cov_mat_built.iloc[0,1]**2/(cov_mat_built.iloc[0,0]*cov_mat_built.iloc[1,1])
print("El valor de r2 es:", r2)


# Comprobamos la igualdad de 
# $R^2 = 1 - \frac{SR}{ST}$ 

SR = (rl_final_built.error**2).sum()
ST = ((rl_final_built.price_uf - rl_final_built.price_uf.mean())**2).sum()
r22 = 1-SR/ST
print("El valor de r2 es:", r22)


# - 4.1.5 Gráficas de Interés

fig, ax = plt.subplots()
sns.scatterplot(data=rl_final_built, y='price_uf', x='built_area',ax=ax)
sns.lineplot(data=rl_final_built, x='built_area', y='predict', color='orange',ax=ax)
ax.set_title("Predicción vs realidad")
ax.set_xlabel("Area construida en M2")
ax.set_ylabel("Precio en UF")
plt.show()


residuals_built=rl_final_built.error
fig = sns.histplot(residuals_built)
fig.set_xlabel("Valor del Error")
fig.set_ylabel("Cantidad")
fig.set_title("Histograma del Error")
plt.show()


fig, ax = plt.subplots(figsize=(6,4))
sm.qqplot(residuals_built, line='s',ax= ax)
plt.title("Q-Q plot de errores")
ax.set_xlabel("Cuantiles Teoricos")
ax.set_ylabel("Cuantiles de la muestra")
plt.show()


# ## Extras

# Definir una función para ajustar el modelo de regresión lineal
def regression_by_group(group,col):
    X = group[col]  # Variable independiente (número de dormitorios)
    y = group['price_uf']  # Variable dependiente (precio)
    X = sm.add_constant(X)  # Añadir constante para la intersección
    model = sm.OLS(y, X).fit()  # Ajustar el modelo OLS
    group['predicted'] = model.predict()
    return model  # Devolver el modelo y el grupo


# Eliminamos las comunas que no tengan más de 20 instancias 
df2 = df.groupby('comuna').filter(lambda x: x.shape[0] >= 20).copy()
df2 = df2.query("(dorms < 10) & (price_uf < 100000)")
df2 = df2.groupby('comuna').filter(lambda x: x.shape[0] >= 20)


df3 = df.groupby('comuna').filter(lambda x: x.shape[0] >= 20).copy()
df3 = df3.query("(built_area < 600) & (built_area < 100000)")
df3 = df3.groupby('comuna').filter(lambda x: x.shape[0] >= 20)


# Agrupar por comuna y aplicar la función de regresión
resultados_dorm = df2.groupby('comuna').apply(regression_by_group,'dorms')

d=[]
# Mostrar los resultados para cada comuna (es largisimo)
for comuna, resultado in resultados_dorm.items():
    d.append(
        {
            "Comuna": comuna,
            "R^2": resultado.rsquared,
            "ESR": (resultado.resid**2/(len(resultado.resid)-2)).sum()**0.5,
        }


    )
df_result_dorm=pd.DataFrame(d)    


resultados_built = df3.groupby('comuna').apply(regression_by_group,'built_area')

d=[]
# Mostrar los resultados para cada comuna (es largisimo)
for comuna, resultado in resultados_built.items():
    d.append(
        {
            "Comuna": comuna,
            "R^2": resultado.rsquared,
            "ESR": (resultado.resid**2/(len(resultado.resid)-2)).sum()**0.5,
        }


    )
df_result_built=pd.DataFrame(d)  


df_result_dorm.set_index('Comuna', inplace=True)
df_result_built.set_index('Comuna', inplace=True)
df_result = df_result_dorm.join(df_result_built, lsuffix='_dorm', rsuffix='_built')


df_result


def grupo_regresion(group, resultados):
    # Predecir los valores usando el modelo ajustado
    group['predicted'] = resultados[group.name].predict()
    # Devolver el grupo con la columna de valores predichos
    return group

df_predicho_dorm = df2.groupby('comuna').apply(grupo_regresion,resultados_dorm)
df_predicho_built = df3.groupby('comuna').apply(grupo_regresion,resultados_built)
comunas = df2['comuna'].unique()
num_comunas = len(comunas)



# Iterar por cada comuna para crear subgráficos
fig, axs = plt.subplots(num_comunas, 2, figsize=(8, num_comunas * 4))  # Un gráfico por fila

#  Iterar por cada comuna para crear un gráfico separado
for i, comuna in enumerate(comunas):
    comuna_data = df_predicho_dorm[df_predicho_dorm['comuna'] == comuna]
    
    # Scatter plot de los datos reales en su subplot respectivo
    axs[i,0].scatter(comuna_data['dorms'], comuna_data['price_uf'], label=f'Datos {comuna}', alpha=0.6)
    
    # Línea de regresión en el mismo subplot
    axs[i,0].plot(comuna_data['dorms'], comuna_data['predicted'], color='orange', label=f'Regresión {comuna}')
    
    # Añadir título y etiquetas a cada subplot
    axs[i,0].set_title(f'Regresión Lineal - {comuna}')
    axs[i,0].set_xlabel('Número de Dormitorios')
    axs[i,0].set_ylabel('Precio')
    axs[i,0].legend()

for i, comuna in enumerate(comunas):
    comuna_data = df_predicho_built[df_predicho_built['comuna'] == comuna]
    if comuna== 'Curacaví': continue
    # Scatter plot de los datos reales en su subplot respectivo
    axs[i,1].scatter(comuna_data['built_area'], comuna_data['price_uf'], label=f'Datos {comuna}', alpha=0.6)
    
    # Línea de regresión en el mismo subplot
    axs[i,1].plot(comuna_data['built_area'], comuna_data['predicted'], color='orange', label=f'Regresión {comuna}')
    
    # Añadir título y etiquetas a cada subplot
    axs[i,1].set_title(f'Regresión Lineal - {comuna}')
    axs[i,1].set_xlabel('M2 Construidos')
    axs[i,1].set_ylabel('Precio')
    axs[i,1].legend()    

# Ajustar el layout para evitar sobreposiciones
plt.tight_layout()
plt.show()





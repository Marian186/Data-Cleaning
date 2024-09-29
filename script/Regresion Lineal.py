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

import pandas as pd               #Manejo de datos en Tablas
import numpy as np
from  matplotlib import pyplot as plt   #Creación de visualizaciones 
import seaborn as sns             #Creación de visualizaciones  

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
# **Estimación de Mínimos Cuadrados Ordinarios (MCO) |** Método común para calcular los coeficientes de regresión lineal $\hat{(\beta)}_n$
# 
# **Función de Pérdida |** Una función que mide la distancia entre los valores observados y los valores estimados por el modelo.

# ##### 3.1 Estimacion de Minimos Cuadrados Ordinarios

# El método de mínimos cuadrados ordinarios (MCO) se utiliza en el análisis de regresión lineal para estimar los parámetros desconocidos del modelo de regresión lineal. El objetivo de la estimación por MCO es encontrar los valores de los coeficientes de regresión que minimicen la suma de los errores al cuadrado entre los valores predichos y los valores reales de la variable dependiente.
# 
# **Línea de Mejor Ajuste |** La línea que ajusta mejor los datos al minimizar alguna función de pérdida o error.
# 
# **Valores Predichos |** Los valores estimados (y) para cada (x) calculados por un modelo.
# 
# **Residuo |** La diferencia entre los valores observados o reales y los valores predichos de la línea de regresión.
# - Residuo = Observado - Predicho ---> $\epsilon_i = y_i - \hat{y_i}$
# 
# **Suma de Residuos al Cuadrado (RSS) |** La suma de las diferencias al cuadrado entre cada valor observado y su valor predicho asociado.
# - $RSS = \sum\limits_{i=1}^{n}(Observado - Predicho)^2$
# - $RSS = \sum\limits_{i=1}^{n}(y_i - \hat{y_i})^2$
# 
# **Mínimos Cuadrados Ordinarios (MCO) |** Un método que minimiza la suma de los residuos al cuadrado para estimar los parámetros en un modelo de regresión lineal.
# - Usado para calcular: $\hat{y}=\hat{\beta_0} + \hat{\beta_1(x)}$
# - Con: $\hat{\beta_1} = \frac{\sum_{i=1}^N{(X_i-\bar{X})(Y_i-\bar{Y})}}{\sum_{i=1}^N{(X_i-\bar{X}})^2} = \frac{cov_{x,y}}{var_x}$

sns.pairplot(df.select_dtypes(include=np.number),diag_kind=None)
plt.show()


# Seleccionamos las columnas relevantes
# Guardamos el DataFrame resultante en una variable separada para la regreción

rl_data = df[["price_uf", "dorms"]]

# Primeras 5 filas
rl_data.head()


from statsmodels.formula.api import ols
import statsmodels.api as sm


rl_formula = "price_uf ~ dorms"


OLS = ols(formula= rl_formula, data= rl_data)


model = OLS.fit()


model.summary()


residuals = model.resid


fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()


sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()


fitted_values = model.predict(rl_data["dorms"])


fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()


import plotly.express as px


fig = px.scatter(df, x="dorms", y="price_uf", trendline="ols")
fig.show()


df2 = df.groupby('comuna').filter(lambda x: x.shape[0] >= 20).copy()


import pandas as pd
import statsmodels.api as sm



# Definir una función para ajustar el modelo de regresión lineal
def regression_by_group(group):
    X = group['dorms']  # Variable independiente (número de dormitorios)
    y = group['price_uf']  # Variable dependiente (precio)
    X = sm.add_constant(X)  # Añadir constante para la intersección
    model = sm.OLS(y, X).fit()  # Ajustar el modelo OLS
    return model.summary()  # Devolver el resumen del modelo

# Agrupar por comuna y aplicar la función de regresión
resultados = df2.groupby('comuna').apply(regression_by_group)

# Mostrar los resultados para cada comuna (es largisimo)
for comuna, resultado in resultados.items():
    print(f"\nComuna: {comuna}\n")
    print(resultado)


def regression_and_plot(group):
    X = group['dorms']
    y = group['price_uf']
    X_with_const = sm.add_constant(X)  # Añadir constante para la intersección
    model = sm.OLS(y, X_with_const).fit()  # Ajustar el modelo
    
    # Predecir los valores usando el modelo ajustado
    group['predicted'] = model.predict(X_with_const)
    
    # Devolver el grupo con la columna de valores predichos
    return group
df_with_predictions = df2.groupby('comuna').apply(regression_and_plot)
comunas = df2['comuna'].unique()
num_comunas = len(comunas)

# Iterar por cada comuna para crear subgráficos
fig, axs = plt.subplots(num_comunas, 1, figsize=(8, num_comunas * 4))  # Un gráfico por fila

#  Iterar por cada comuna para crear un gráfico separado
for i, comuna in enumerate(comunas):
    comuna_data = df_with_predictions[df_with_predictions['comuna'] == comuna]
    
    # Scatter plot de los datos reales en su subplot respectivo
    axs[i].scatter(comuna_data['dorms'], comuna_data['price_uf'], label=f'Datos {comuna}', alpha=0.6)
    
    # Línea de regresión en el mismo subplot
    axs[i].plot(comuna_data['dorms'], comuna_data['predicted'], color='orange', label=f'Regresión {comuna}')
    
    # Añadir título y etiquetas a cada subplot
    axs[i].set_title(f'Regresión Lineal - {comuna}')
    axs[i].set_xlabel('Número de Dormitorios')
    axs[i].set_ylabel('Precio')
    axs[i].legend()

# Ajustar el layout para evitar sobreposiciones
plt.tight_layout()
plt.show()





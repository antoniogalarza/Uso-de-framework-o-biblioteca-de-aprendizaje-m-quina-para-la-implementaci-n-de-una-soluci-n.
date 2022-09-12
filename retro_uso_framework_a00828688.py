

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Predicción de la calidad del vino tinto utilizando Random Forest
df = pd.read_csv (r"C:\Users\anton\OneDrive\Documents/winequality-red.csv")   
df.head()
# Descripción de los datos
df.describe()
# Conteo de los registros de calidad del vino
print(df['quality'].value_counts())
# Gráfica registros de la calidad del vino
plt.figure(1, figsize=(10,10))
df['quality'].value_counts().plot.pie(autopct="%1.2f%%")
X = df.drop('quality', axis = 1)
y = df.quality             # Variable dependiente
# Separando Conjunto de datos 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 42)
# Creando Modelo 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# Predicción Calidad del Vino
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
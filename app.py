import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Titulo de la aplicación
st.title("Predicción de Diagnóstico de Salida en Pacientes Psiquiátricos")

# Cargar datos
st.subheader("Datos originales")
df = pd.read_excel("Pacientes_psiquiatria.xlsx")
df = df[['Edad', 'Sexo', 'Diagnóstico_ingreso', 'Diagnóstico_salida']].dropna()
st.write(df.head())

# Codificar variables categóricas
encoder = OneHotEncoder(drop='first', sparse=False)
X_categoricas = encoder.fit_transform(df[['Sexo', 'Diagnóstico_ingreso']])
X_numericas = df[['Edad']].values
X = pd.DataFrame(
    data=np.hstack([X_numericas, X_categoricas]),
    columns=['Edad'] + encoder.get_feature_names_out(['Sexo', 'Diagnóstico_ingreso']).tolist()
)
y = df['Diagnóstico_salida']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Visualización de matriz de confusión
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
st.pyplot(fig)

# Mostrar reporte de clasificación
st.subheader("Reporte de Clasificación")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

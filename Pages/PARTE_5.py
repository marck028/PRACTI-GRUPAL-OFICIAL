import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Título de la aplicación
st.title("📊 Predicción del Rango de Precio de Teléfonos Móviles 📱")

# Especificar la ruta del archivo CSV
dataset_path = "C:\\Users\\soled\\Downloads\\dataset_modificado.csv"

try:
    # Leer el archivo CSV desde la ruta especificada
    df = pd.read_csv(dataset_path, encoding="latin-1", delimiter=",")
    
    # Mostrar las primeras filas del dataset
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head())

    # Selección de columnas para análisis
    selected_columns = st.multiselect(
        "Selecciona las columnas para el modelo:",
        df.columns,
        default=["marca_telefono", "almacenamiento", "ram", "bateria", "precio_usd"]
    )
    
    if len(selected_columns) > 1:
        # Filtrar el dataset
        df_filtered = df[selected_columns]
        
        # Convertir variables categóricas a numéricas
        st.subheader("Conversión de Variables Categóricas")
        cat_columns = df_filtered.select_dtypes(include=["object"]).columns
        for col in cat_columns:
            df_filtered[col] = pd.factorize(df_filtered[col])[0]
            st.write(f"✔ Columna '{col}' convertida a numérica.")
        
        st.write("Dataset procesado:")
        st.dataframe(df_filtered.head())

        # Seleccionar la variable objetivo
        target_column = st.selectbox("Selecciona la variable objetivo (precio o rango de precio):", df_filtered.columns)
        features = [col for col in df_filtered.columns if col != target_column]
        
        # Dividir en conjunto de entrenamiento y prueba
        X = df_filtered[features]
        y = df_filtered[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo
        st.subheader("Entrenamiento del Modelo")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluación del modelo
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"🎯 **Precisión del Modelo:** {accuracy:.2f}")
        
        # Matriz de Confusión
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        st.pyplot(fig)
        
        # Curva ROC (si es aplicable)
        if len(np.unique(y)) == 2:  # Solo para clasificación binaria
            st.subheader("Curva ROC")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name="Curva ROC", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Referencia", line=dict(dash='dash')))
            fig.update_layout(
                title=f"Curva ROC (AUC = {roc_auc:.2f})",
                xaxis_title="Falsos Positivos (FPR)",
                yaxis_title="Verdaderos Positivos (TPR)",
                template="plotly_dark"
            )
            st.plotly_chart(fig)

        # Reporte de Clasificación
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

except FileNotFoundError:
    st.write(f"El archivo no fue encontrado en la ruta especificada: {dataset_path}")
except Exception as e:
    st.write(f"Ocurrió un error: {e}")

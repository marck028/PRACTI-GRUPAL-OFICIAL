import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 🌟 Título Principal
st.title('📊 Identificación de Variables para el Modelo')

# 🌍 Especificar la ruta del archivo CSV directamente
dataset_path = "C:\\Users\\soled\\Downloads\\dataset_modificado.csv"

# Verificar si el archivo existe
if os.path.exists(dataset_path):
    # 📜 Leer el archivo CSV
    df = pd.read_csv(dataset_path, encoding='latin-1', delimiter=',')
    st.write("✅ Archivo cargado con éxito.")

    # **1. Selección de Variables**
    st.subheader('1️⃣ Selección de Variables')
    st.markdown("""🔍 Se seleccionarán las variables relevantes para predecir el **rango de precio**.""")

    # Visualización inicial
    st.write("📌 Primeras filas del dataset cargado:")
    st.dataframe(df.head())

    # Selección automática o manual de columnas
    default_columns = ["marca_telefono", "almacenamiento", "ram", "resolucion_pantalla", "bateria", "resolucion_video", "precio_usd"]
    available_columns = [col for col in default_columns if col in df.columns]
    selected_columns = st.multiselect('🔧 Selecciona las columnas para el modelo:', df.columns.tolist(), default=available_columns)

    if selected_columns:
        df_selected = df[selected_columns]
        st.write("✅ Columnas seleccionadas:")
        st.dataframe(df_selected.head())
    else:
        st.warning('⚠️ Por favor selecciona al menos una columna.')

    # **2. Preparación de Datos para el Modelo**
    st.subheader('2️⃣ Preparación de Datos para el Modelo')
    st.markdown("""✨ Los pasos clave incluyen:
    - Transformar variables categóricas a numéricas.
    - Escalar los datos para mejorar el rendimiento del modelo.
    - Dividir los datos en conjuntos de entrenamiento y prueba.""")

    # 🔄 Conversión de variables categóricas a numéricas
    st.write("🔄 **Conversión de variables categóricas a numéricas**")
    cat_columns = df_selected.select_dtypes(include=['object']).columns
    if len(cat_columns) > 0:
        le = LabelEncoder()
        for col in cat_columns:
            df_selected[col] = le.fit_transform(df_selected[col].astype(str))
            st.write(f"✔️ Columna **{col}** convertida exitosamente.")

    st.write("📊 Dataset después de la conversión:")
    st.dataframe(df_selected.head())

    # 🔢 Escalado de datos numéricos
    st.write("📏 **Escalado de datos numéricos**")
    num_columns = df_selected.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df_selected[num_columns] = scaler.fit_transform(df_selected[num_columns])
    st.write("✅ Escalado completado.")
    st.dataframe(df_selected.head())

    # 🔀 División en entrenamiento y prueba
    st.write("📚 **División del Dataset**")
    if "precio_usd" in selected_columns:
        X = df_selected.drop(columns=["precio_usd"])
        y = df_selected["precio_usd"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        st.write(f"🔹 Tamaño del conjunto de entrenamiento: {X_train.shape[0]} filas.")
        st.write(f"🔹 Tamaño del conjunto de prueba: {X_test.shape[0]} filas.")
    else:
        st.warning("⚠️ Por favor asegúrate de incluir la variable objetivo `precio_usd`.")

else:
    st.error("❌ El archivo no existe en la ruta proporcionada. Por favor verifica la ruta.")

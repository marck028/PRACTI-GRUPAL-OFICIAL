import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Título de la aplicación
st.title('📊 Preparación y Limpieza de Datos 📊')

# Subir archivo CSV
uploaded_file = st.file_uploader('📂 Subir Archivo CSV', type=['csv'])

if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file, encoding='latin-1', delimiter=',')
    
    # Confirmación de carga
    st.success('Archivo subido exitosamente 📄')

    # Mostrar primeras filas
    st.subheader('🎯 Primeras Filas del Dataset')
    st.dataframe(df.head(), height=250)

    # Resumen estadístico
    st.subheader('📈 Resumen Estadístico')
    st.write(df.describe())

    # Nombres de las columnas
    st.subheader('📝 Nombres de las Columnas')
    st.write(df.columns)

    # Tipos de datos
    st.subheader('🔍 Tipos de Datos de las Columnas')
    st.write(df.dtypes)

    # Tamaño del dataset
    st.subheader('📏 Tamaño del Dataset')
    st.write(f'Filas: {df.shape[0]}, Columnas: {df.shape[1]}')

    # Valores nulos
    st.subheader('⚠️ Valores Nulos en el Dataset')
    st.write(df.isnull().sum())

    # Total de valores faltantes
    st.subheader('📉 Total de Valores Faltantes')
    total_nulls = df.isnull().sum().sum()
    st.write(f"Total de valores faltantes: {total_nulls}")

    # Filas duplicadas
    st.subheader('🔁 Filas Duplicadas')
    st.write(df.duplicated().sum())

    # 2. Revisión y Manejo de Datos Faltantes
    st.header('🛠 Manejo de Datos Faltantes')

    # Rellenar los valores faltantes con la media para las columnas clave
    columns_to_fill = ['precio_usd', 'almacenamiento', 'ram']  # Las columnas clave que vamos a rellenar
    for col in columns_to_fill:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            st.write(f"Columna '{col}' rellenada con la media: {mean_value}")

    # Mostrar el DataFrame después del manejo de los valores faltantes
    st.subheader('🎯 Dataset después de manejar los valores faltantes')
    st.dataframe(df.head())

    # Asegurarse de formatos correctos
    st.header('🛠 Formateo de Fechas y Precios')
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
                st.write(f"Columna '{col}' convertida a formato fecha.")
            except ValueError:
                pass
        
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                st.write(f"Columna '{col}' ya está en formato numérico.")

    # Conversión de Variables Categóricas
    st.header('🔢 Conversión de Variables Categóricas a Numéricas')
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            st.write(f'Columna {col} convertida exitosamente.')
        
        st.subheader('Dataset con Variables Categóricas Convertidas')
        st.dataframe(df.head())
    else:
        st.write("No se detectaron columnas categóricas.")

    # Detección de Outliers
    st.header('📊 Detección de Outliers')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Cálculo de los límites para los outliers usando el rango intercuartílico
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        st.subheader(f"📉 Columna: {col}")
        st.write(f"Cantidad de *outliers* detectados: {len(outliers)}")
        
        if len(outliers) > 0:
            st.write(f"🔴 Aquí están los *outliers* encontrados en la columna '{col}':")
            st.dataframe(outliers, height=200)
        else:
            st.write(f"✅ No se encontraron *outliers* en la columna '{col}'.")

    # Descargar el dataset con los valores faltantes completados
    st.subheader('📥 Descargar el Dataset Modificado')

    # Convertir el DataFrame a CSV
    csv_data = df.to_csv(index=False)

    # Crear enlace para descargar el archivo
    st.download_button(
        label="Descargar Dataset Modificado",
        data=csv_data,
        file_name="dataset_modificado.csv",
        mime="text/csv"
    )

else:
    st.info('⚠️ Por favor, sube un archivo CSV para comenzar.')

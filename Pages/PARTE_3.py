import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title('📊 Dashboard 📊')

# Especificar la ruta del archivo CSV
dataset_path = "C:\\Users\\soled\\Downloads\\dataset_modificado.csv"

# Leer el archivo CSV directamente desde la ruta especificada
df = pd.read_csv(dataset_path, encoding='latin-1', delimiter=',')

# Lista de columnas seleccionadas
selected_columns = [
    "marca_telefono",
    "almacenamiento",
    "ram",
    "resolucion_pantalla",
    "bateria",
    "resolucion_video",
    "precio_usd"
]

# Verificar si las columnas seleccionadas existen en el dataset
columns_to_use = [col for col in selected_columns if col in df.columns]
if columns_to_use:
    df_filtered = df[columns_to_use]
    st.subheader('🔍 Columnas Seleccionadas del Dataset')
    st.dataframe(df_filtered.head())
    
    # Conversión de columnas categóricas a numéricas (si existen)
    cat_columns = df_filtered.select_dtypes(include=['object']).columns
    if len(cat_columns) > 0:
        st.subheader('🔄 Convertir Variables Categóricas a Numéricas')
        le = LabelEncoder()
        for col in cat_columns:
            df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))
            st.write(f'✅ Columna {col} convertida exitosamente.')
        
        st.subheader('📊 Dataset con Variables Categóricas Convertidas')
        st.dataframe(df_filtered.head())
        
    # Análisis de la relación entre Precio, RAM y Almacenamiento
    st.title('🔎 Análisis de Relaciones')

    # Relación entre precio_usd y ram
    st.subheader('📉 Relación entre Precio (USD) y RAM')
    fig1 = px.scatter(df, x='ram', y='precio_usd', color='marca_telefono', title='Relación entre Precio y RAM',
                      color_continuous_scale='Rainbow', hover_data=['marca_telefono'])
    st.plotly_chart(fig1)

    # Relación entre precio_usd y almacenamiento
    st.subheader('📊 Relación entre Precio (USD) y Almacenamiento')
    fig2 = px.scatter(df, x='almacenamiento', y='precio_usd', color='marca_telefono', title='Relación entre Precio y Almacenamiento',
                      color_continuous_scale='Blues', hover_data=['marca_telefono'])
    st.plotly_chart(fig2)

    # Relación entre precio_usd y batería
    st.subheader('🔋 Relación entre Precio (USD) y Batería')
    fig3 = px.scatter(df, x='bateria', y='precio_usd', color='marca_telefono', title='Relación entre Precio y Batería',
                      color_continuous_scale='Viridis', hover_data=['marca_telefono'])
    st.plotly_chart(fig3)

    # Heatmap de correlación interactivo
    st.title('🔎 Análisis de Correlación entre Variables')
    st.subheader('🌡️ Correlación entre Variables Numéricas')

    # Filtrar solo las columnas numéricas para el heatmap de correlación
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_cols]

    # Calcular la matriz de correlación solo con las columnas numéricas
    correlation_matrix = df_numeric.corr()

    # Crear un gráfico interactivo de la matriz de correlación
    fig_corr = px.imshow(correlation_matrix, title="Matriz de Correlación",
                         color_continuous_scale='RdBu_r', aspect='auto', text_auto=True)
    st.plotly_chart(fig_corr)

    # Filtros interactivos
    st.title('🛠️ Filtrar Datos')
    marcas = df['marca_telefono'].unique()
    selected_brand = st.selectbox('🏷️ Selecciona una Marca', ['Todos'] + list(marcas))
    
    if selected_brand != 'Todos':
        df_filtered = df[df['marca_telefono'] == selected_brand]
    else:
        df_filtered = df

    if 'modelo_telefono' in df.columns:
        modelos = df_filtered['modelo_telefono'].unique()
        selected_model = st.selectbox('📱 Selecciona un Modelo', ['Todos'] + list(modelos))
        if selected_model != 'Todos':
            df_filtered = df_filtered[df_filtered['modelo_telefono'] == selected_model]

    # Mostrar los datos filtrados
    st.subheader(f"📊 Datos para la Marca: {selected_brand}")
    st.subheader(f"📱 Datos para el Modelo: {selected_model}")
    st.dataframe(df_filtered.head())

    # Gráfico de barras de popularidad por marca
    st.subheader('📊 Popularidad de Marcas')
    brand_counts = df_filtered['marca_telefono'].value_counts().reset_index()
    brand_counts.columns = ['Marca', 'Cantidad']
    fig5 = px.bar(brand_counts, x='Marca', y='Cantidad', title='Popularidad de Marcas de Teléfonos',
                  color='Cantidad', color_continuous_scale='Cividis')
    st.plotly_chart(fig5)

    # Entrenamiento del modelo de regresión con RandomForestRegressor
    st.title('🔧 Entrenamiento de Modelo de Regresión')

    # Eliminar las columnas con valores no numéricos
    df_filtered = df_filtered.select_dtypes(include=['float64', 'int64'])

    # Si deseas convertir fechas a variables numéricas (ejemplo: días desde una fecha base)
    if 'fecha_columna' in df.columns:  # Asegúrate de que 'fecha_columna' sea el nombre correcto
        df['fecha_columna'] = pd.to_datetime(df['fecha_columna'])
        df['dias_desde_inicio'] = (df['fecha_columna'] - pd.to_datetime('2020-01-01')).dt.days
        df_filtered = df_filtered.drop(columns=['fecha_columna'])  # Eliminar la columna original de fecha

    # Ahora X debe contener solo columnas numéricas para que funcione el modelo de regresión
    X = df_filtered.drop(columns=['precio_usd'])  # Eliminar la columna objetivo
    y = df['precio_usd']  # Variable objetivo

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo de regresión
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Entrenar el modelo
    rf_model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = rf_model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('📊 Evaluación del Modelo')
    st.write(f'Error Cuadrático Medio (MSE): {mse}')
    st.write(f'R2 Score: {r2}')

    # Gráfico de las predicciones vs valores reales
    st.subheader('🔮 Predicciones vs Valores Reales')
    fig_pred = plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    st.pyplot(fig_pred)

else:
    st.write('⚠️ Las Columnas Seleccionadas No Están Presentes en el Archivo.')


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Título de la aplicación
st.title('📊 Análisis Exploratorio de Datos 📊')

# Especificar la ruta del archivo CSV
dataset_path = "C:\\Users\\soled\\Downloads\\dataset_modificado.csv"

# Leer el archivo CSV
df = pd.read_csv(dataset_path, encoding='latin-1', delimiter=',')

# Lista de columnas seleccionadas
selected_columns = [
    "marca_telefono",
    "almacenamiento",
    "ram",
    "bateria",
    "precio_usd"
]

# Verificar si las columnas seleccionadas existen en el dataset
columns_to_use = [col for col in selected_columns if col in df.columns]
if columns_to_use:
    df_filtered = df[columns_to_use]
    st.subheader('✨ **Columnas Seleccionadas del Dataset** ✨')
    st.dataframe(df_filtered.head())
    
    # Parte 1: Distribución de Variables
    # Visualizar distribuciones de variables como precio_usd, ram, almacenamiento, y tamaño_bateria
    st.subheader('📈 **Distribución de Variables** 📈')

    # Precio (USD)
    st.write("### 💵 **Distribución de Precio (USD)**")
    fig_price = px.histogram(df, x='precio_usd', nbins=30, title="Distribución de Precio (USD)",
                             color_discrete_sequence=['#FF6347'])
    st.plotly_chart(fig_price)

    # RAM
    st.write("### 🧠 **Distribución de RAM**")
    fig_ram = px.histogram(df, x='ram', nbins=10, title="Distribución de RAM (GB)",
                           color_discrete_sequence=['#FFD700'])
    st.plotly_chart(fig_ram)

    # Almacenamiento
    st.write("### 💾 **Distribución de Almacenamiento**")
    fig_storage = px.histogram(df, x='almacenamiento', nbins=10, title="Distribución de Almacenamiento (GB)",
                               color_discrete_sequence=['#32CD32'])
    st.plotly_chart(fig_storage)

    # Batería
    st.write("### 🔋 **Distribución de Batería**")
    fig_battery = px.histogram(df, x='bateria', nbins=30, title="Distribución de Batería (mAh)",
                               color_discrete_sequence=['#8A2BE2'])
    st.plotly_chart(fig_battery)

    # Parte 2: Correlación entre Variables Numéricas
    # Calcular y visualizar la matriz de correlación entre variables numéricas
    st.subheader('🔍 **Correlación entre Variables Numéricas** 🔍')
    correlation_matrix = df[['precio_usd', 'ram', 'almacenamiento', 'bateria']].corr()
    st.write(correlation_matrix)

    # Graficar la matriz de correlación
    fig_corr = px.imshow(correlation_matrix, title="Matriz de Correlación",
                         color_continuous_scale='Blues')
    st.plotly_chart(fig_corr)

    # Parte 3: Análisis de Variables Categóricas
    # Usar gráficos de barras para analizar la popularidad de diferentes marcas y sistemas operativos
    st.subheader('📊 **Análisis de Variables Categóricas** 📊')

    # Popularidad de marcas de teléfonos
    if 'marca_telefono' in df.columns:
        st.write("### 📱 **Popularidad de Marcas de Teléfonos (Top 10)**")
        brand_counts = df['marca_telefono'].value_counts().head(10).reset_index()
        brand_counts.columns = ['Marca', 'Cantidad']
        fig_brands = px.bar(brand_counts, x='Marca', y='Cantidad', 
                            title='Popularidad de Marcas (Top 10)',
                            labels={'Cantidad': 'Número de Teléfonos', 'Marca': 'Marca'},
                            color='Cantidad', color_continuous_scale='viridis')
        st.plotly_chart(fig_brands)

    # Popularidad de sistemas operativos
    if 'sistema operativo' in df.columns:
        st.write("### 📱 **Popularidad de Sistemas Operativos (Top 10)**")
        os_counts = df['sistema operativo'].value_counts().head(10).reset_index()
        os_counts.columns = ['Sistema Operativo', 'Cantidad']
        fig_os = px.bar(os_counts, x='Sistema Operativo', y='Cantidad',
                        title='Popularidad de Sistemas Operativos (Top 10)',
                        labels={'Cantidad': 'Número de Teléfonos', 'Sistema Operativo': 'Sistema Operativo'},
                        color='Cantidad', color_continuous_scale='cividis')
        st.plotly_chart(fig_os)

    # Parte 4: Análisis de Precio vs. Especificaciones
    # Visualizar la relación entre precio_usd y variables como almacenamiento, ram, y bateria
    st.subheader('💸 **Relación entre Precio y Especificaciones** 💸')

    # Relación entre precio_usd y ram
    st.write("### 💰 **Relación entre Precio (USD) y RAM**")
    fig_price_ram = go.Figure(data=go.Scatter(x=df['ram'], y=df['precio_usd'],
                                             mode='markers', marker=dict(color='blue', opacity=0.6, size=8)))
    fig_price_ram.update_layout(title='Relación entre Precio (USD) y RAM',
                                xaxis_title='RAM (GB)', yaxis_title='Precio (USD)',
                                template='plotly_dark')
    st.plotly_chart(fig_price_ram)

    # Relación entre precio_usd y almacenamiento
    st.write("### 💾 **Relación entre Precio (USD) y Almacenamiento**")
    fig_price_storage = go.Figure(data=go.Scatter(x=df['almacenamiento'], y=df['precio_usd'],
                                                 mode='markers', marker=dict(color='green', opacity=0.6, size=8)))
    fig_price_storage.update_layout(title='Relación entre Precio (USD) y Almacenamiento',
                                    xaxis_title='Almacenamiento (GB)', yaxis_title='Precio (USD)',
                                    template='plotly_dark')
    st.plotly_chart(fig_price_storage)

    # Relación entre precio_usd y batería
    st.write("### 🔋 **Relación entre Precio (USD) y Batería**")
    fig_price_battery = go.Figure(data=go.Scatter(x=df['bateria'], y=df['precio_usd'],
                                                 mode='markers', marker=dict(color='purple', opacity=0.6, size=8)))
    fig_price_battery.update_layout(title='Relación entre Precio (USD) y Batería',
                                    xaxis_title='Batería (mAh)', yaxis_title='Precio (USD)',
                                    template='plotly_dark')
    st.plotly_chart(fig_price_battery)

else:
    st.write('⚠️ Las columnas seleccionadas no están presentes en el archivo. ⚠️')

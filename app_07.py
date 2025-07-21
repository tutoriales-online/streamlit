from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("Gráficas y Visualizaciones en Streamlit")

# Datos de ejemplo
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
sales_data = pd.DataFrame(
    {
        "fecha": dates,
        "ventas": np.random.randint(100, 1000, 100),
        "producto": np.random.choice(["A", "B", "C"], 100),
        "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], 100),
    }
)

# Datos para gráficas de barras
product_sales = sales_data.groupby("producto")["ventas"].sum().reset_index()
region_sales = sales_data.groupby("region")["ventas"].sum().reset_index()

# Datos para gráficas de dispersión
scatter_data = pd.DataFrame(
    {"x": np.random.randn(50), "y": np.random.randn(50), "categoria": np.random.choice(["A", "B", "C"], 50)}
)

st.header("1. Gráfica de Líneas")
st.line_chart(sales_data.set_index("fecha")["ventas"])

st.divider()

st.header("2. Gráfica de Barras")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ventas por Producto")
    st.bar_chart(product_sales.set_index("producto"))

with col2:
    st.subheader("Ventas por Región")
    st.bar_chart(region_sales.set_index("region"))

st.divider()

st.header("3. Gráfica de Área")
st.area_chart(sales_data.set_index("fecha")["ventas"])

st.divider()

st.header("4. Gráfica de Dispersión (scatter)")
st.scatter_chart(scatter_data, x="x", y="y", color="categoria")

st.divider()

st.header("5. Gráficas con Plotly")

# Gráfica de líneas con Plotly
fig_line = px.line(
    sales_data, x="fecha", y="ventas", color="producto", title="Ventas por Producto a lo largo del tiempo"
)
st.plotly_chart(fig_line, use_container_width=True)

# Gráfica de barras con Plotly
fig_bar = px.bar(product_sales, x="producto", y="ventas", title="Ventas Totales por Producto")
st.plotly_chart(fig_bar, use_container_width=True)

# Gráfica de dispersión con Plotly
fig_scatter = px.scatter(scatter_data, x="x", y="y", color="categoria", title="Gráfica de Dispersión")
st.plotly_chart(fig_scatter, use_container_width=True)

# Gráfica de pastel
fig_pie = px.pie(product_sales, values="ventas", names="producto", title="Distribución de Ventas por Producto")
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

st.header("6. Gráficas Interactivas")

# Selector de tipo de gráfica
chart_type = st.selectbox("Selecciona el tipo de gráfica:", ["Líneas", "Barras", "Área", "Dispersión"])

# Selector de datos
data_source = st.selectbox("Selecciona los datos:", ["Ventas por Producto", "Ventas por Región", "Ventas en el Tiempo"])

if data_source == "Ventas por Producto":
    data_to_plot = product_sales
    x_col = "producto"
    y_col = "ventas"
elif data_source == "Ventas por Región":
    data_to_plot = region_sales
    x_col = "region"
    y_col = "ventas"
else:
    data_to_plot = sales_data.groupby("fecha")["ventas"].sum().reset_index()
    x_col = "fecha"
    y_col = "ventas"

if chart_type == "Líneas":
    fig = px.line(data_to_plot, x=x_col, y=y_col, title=f"{chart_type}: {data_source}")
elif chart_type == "Barras":
    fig = px.bar(data_to_plot, x=x_col, y=y_col, title=f"{chart_type}: {data_source}")
elif chart_type == "Área":
    fig = px.area(data_to_plot, x=x_col, y=y_col, title=f"{chart_type}: {data_source}")
else:  # Dispersión
    fig = px.scatter(data_to_plot, x=x_col, y=y_col, title=f"{chart_type}: {data_source}")

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("7. Métricas y KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Ventas Totales",
        value=f"${sales_data['ventas'].sum():,}",
        delta=f"${sales_data['ventas'].sum() - sales_data['ventas'].mean() * 100:,.0f}",
    )

with col2:
    st.metric(
        label="Promedio Diario",
        value=f"${sales_data['ventas'].mean():.0f}",
        delta=f"{((sales_data['ventas'].mean() / sales_data['ventas'].iloc[-10:].mean()) - 1) * 100:.1f}%",
    )

with col3:
    st.metric(
        label="Producto Más Vendido",
        value=product_sales.loc[product_sales["ventas"].idxmax(), "producto"],
        delta=f"${product_sales['ventas'].max():,}",
    )

with col4:
    st.metric(
        label="Región Líder",
        value=region_sales.loc[region_sales["ventas"].idxmax(), "region"],
        delta=f"${region_sales['ventas'].max():,}",
    )

st.divider()

st.header("9. Información Adicional")

with st.expander("Ver datos de ventas"):
    st.dataframe(sales_data)

with st.expander("Ver estadísticas"):
    st.write("**Estadísticas de Ventas:**")
    st.write(sales_data["ventas"].describe())

    st.write("**Ventas por Producto:**")
    st.write(product_sales)

    st.write("**Ventas por Región:**")
    st.write(region_sales)

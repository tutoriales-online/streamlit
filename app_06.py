import pandas as pd
import streamlit as st

st.title("Datos")

datos_pequenos = pd.DataFrame(
    {
        "nombre": ["Juan", "Maria", "Pedro", "Ana"],
        "edad": [20, 25, 30, 35],
        "ciudad": ["Bogota", "Medellin", "Cali", "Barranquilla"],
    }
)

st.table(datos_pequenos)

st.divider()

NUM_ROWS = 1000
NUM_COLUMNS = 10

datos_grandes = pd.DataFrame({f"columna_{i}": [f"valor_{i}" for i in range(NUM_ROWS)] for i in range(NUM_COLUMNS)})

st.dataframe(datos_grandes)

st.divider()

columns = st.columns(3)

with columns[0]:
    json_data = {"name": "John", "age": 30, "city": "New York"}
    st.json(json_data)

with columns[1]:
    st.metric(label="Edad", value=30, delta=10)

with columns[2]:
    st.metric(label="Temperatura", value="30.4°", delta="-1.2")

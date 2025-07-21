import time

import streamlit as st

st.title("Cache")


@st.cache_data
def get_data(count):
    st.write(f"Obteniendo recurso {count}")
    time.sleep(10)
    return f"Recurso {count}"


@st.cache_resource
def get_ml_model(count):
    st.write(f"Obteniendo modelo {count}")
    time.sleep(10)
    return {
        "name": f"Modelo {count}",
    }


count_data = st.number_input("Data count", min_value=0, max_value=10, value=0)

st.write(get_data(count_data))


count_model = st.number_input("Model count", min_value=0, max_value=10, value=0)

st.write(get_ml_model(count_model))

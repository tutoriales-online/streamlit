import streamlit as st

st.title("Layouts y contenedores")

# Dos columnas con el mismo tamaño
left, right = st.columns(2)

with left:
    st.write("Este es el contenido de la columna izquierda")

with right:
    st.write("Este es el contenido de la columna derecha")

st.divider()

# Dos columnas con diferentes tamaños
left, right = st.columns([1, 2])

with left:
    st.write("Este es el contenido de la columna izquierda")

with right:
    st.write("Este es el contenido de la columna derecha")

st.divider()

# Multiples columnas con diferentes tamaños
left, center, right = st.columns([1, 4, 2])

with left:
    st.write("Tamaño 1/7")

with center:
    st.write("Tamaño 4/7")

with right:
    st.write("Tamaño 2/7")

st.divider()


# Expander

expand = st.expander("Click para expandir")

with expand:
    st.write("Contenido del expander")
    name = st.text_input("Ingrese su nombre")

st.divider()

# Tabs

tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.header("Tab 1")
    st.write("Contenido de la pestaña 1")

with tab2:
    st.header("Tab 2")
    st.write("Contenido de la pestaña 2")

with tab3:
    st.header("Tab 3")
    st.write("Contenido de la pestaña 3")

st.divider()

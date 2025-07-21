import streamlit as st

st.title("Modificando valores de forma no secuencial")

top_container = st.container()

name = st.text_input("Nombre")


with top_container:
    if name:
        st.header(f"Hola {name}")

st.divider()


left, right = st.columns(2)

with left:
    name = st.text_input("Text")

with right:
    times = st.number_input("Times", min_value=0, max_value=10, value=0)

with left:
    names = [name] * times

    st.text(", ".join(names))

with top_container:
    if names:
        st.code(", ".join(names))

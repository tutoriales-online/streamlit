import streamlit as st

st.title("Mira a tu izquierda 👈")

with st.sidebar:
    st.title("Configuración")

    st.write("Esta es una aplicación de ejemplo")

    name = st.text_input("Nombre")

    st.text("¿Qué opinas de la clase?")
    feedback = st.feedback(options="thumbs")

if feedback:
    st.subheader(f"Hola {name}, que bien qué te esté gustando la clase.")
    st.write(feedback)
else:
    st.subheader(f"Hola {name}, ¿cómo podríamos mejorar?")

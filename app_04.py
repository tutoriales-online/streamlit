from datetime import datetime

import streamlit as st

st.title("Entradas de datos")

# Text input, number input y text area

izquierda, centro, derecha = st.columns(3)

with izquierda:
    text_variable = st.text_input("Text input variable")

with centro:
    number_variable = st.number_input("Number input variable", min_value=0, max_value=100, value=50)

with derecha:
    text_area_variable = st.text_area("Text area variable", height=100)

izquierda, centro, derecha = st.columns(3)

with izquierda:
    st.markdown(f"Value: **{text_variable or '`None`'}** ({type(text_variable).__name__})")

with centro:
    st.markdown(f"Value: **{number_variable}** ({type(number_variable).__name__})")

with derecha:
    st.markdown(f"Value: **{text_area_variable or '`None`'}** ({type(text_area_variable).__name__})")

st.divider()

# Checkbox, radio y selectbox

izquierda, centro, derecha = st.columns(3)

with izquierda:
    checkbox_variable = st.checkbox("Checkbox variable")

with centro:
    radio_variable = st.radio("Radio variable", ["Opción 1", "Opción 2"])

with derecha:
    selectbox_variable = st.selectbox("Selectbox variable", ["Opción 1", "Opción 2"])


izquierda, centro, derecha = st.columns(3)

with izquierda:
    st.markdown(f"Value: **{checkbox_variable}** ({type(checkbox_variable).__name__})")

with centro:
    st.markdown(f"Value: **{radio_variable}** ({type(radio_variable).__name__})")

with derecha:
    st.markdown(f"Value: **{selectbox_variable}** ({type(selectbox_variable).__name__})")

st.divider()

# Color picker, date input y time input

izquierda, centro, derecha = st.columns(3)

with izquierda:
    color_variable = st.color_picker("Color picker variable")

with centro:
    date_variable = st.date_input("Date input variable", value=datetime.now())

with derecha:
    time_variable = st.time_input("Time input variable", value=datetime.now())


with izquierda:
    st.markdown(f"Value: **{color_variable}** ({type(color_variable).__name__})")

with centro:
    st.markdown(f"Value: **{date_variable}** ({type(date_variable).__name__})")

with derecha:
    st.markdown(f"Value: **{time_variable}** ({type(time_variable).__name__})")

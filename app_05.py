import streamlit as st

st.title("Interactuando con botones")

click_action_1 = st.button("Click action 1")
click_action_2 = st.button("Click action 2")
click_action_3 = st.button("Click action 3")

if click_action_1:
    st.write("Click action 1")

if click_action_2:
    st.write("Click action 2")

if click_action_3:
    st.write("Click action 3")

st.divider()


st.link_button("Visita mi página web", "https://feregri.no")

st.download_button(
    label="Descargar archivo",
    data=b"Hello, world!",
    file_name="hello.txt",
    mime="text/plain",
)

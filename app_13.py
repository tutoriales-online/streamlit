import time

import streamlit as st

st.title("State")


if "count" not in st.session_state:
    st.session_state.count = 0

increase_button = st.button("Increase")

if increase_button:
    st.session_state.count += 1

st.write(f"Count: {st.session_state.count}")

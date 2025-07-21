from dataclasses import dataclass

import streamlit as st

st.title("Mostrando texto de diferentes maneras")
st.header("Encabezado")
st.subheader("Subencabezado")
st.text("Texto de ejemplo")
st.write("Texto de ejemplo")

st.divider()
st.markdown(
    """
# Mostrando texto de diferentes maneras

## Encabezado

### Subencabezado

Texto de ejemplo

Texto de ejemplo

"""
)


st.markdown(
    """
```python
import streamlit as st

st.title("Mi primera app con Streamlit")
st.write("Hola, mundo!")
```
            
*Streamlit* es una biblioteca de Python para crear aplicaciones web _interactivas_.
            
 - Prototipos rápidos
 - Todo en Python
"""
)

st.divider()

dictionary = {"apple": "manzana", "banana": "plátano", "cherry": "cereza"}

st.write(dictionary)


@dataclass
class Person:
    name: str
    age: int
    city: str


person = Person(name="John", age=30, city="New York")

st.write(person)

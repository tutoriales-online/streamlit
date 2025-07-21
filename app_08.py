from datetime import datetime, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

location = [19.432608, -99.133209]

m = folium.Map(location=location, zoom_start=16)


st.header("Mapa de la Ciudad de México")

st_data = st_folium(m, width=725, height=500)

st.write(st_data)

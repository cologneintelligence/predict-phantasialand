"""
Project: Phantasialand  
State: 11/2021

Serve a streamlit app that allows to easily query the model for different attractions 
and dates and to visualize the resulting waiting times.
"""

import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

from src.models.model_estimator import ModelEstimator, BEST_MODEL_PATH
from src.data.constants import WARTEZEITEN_APP_ATTRACTIONS
from src.models.weather_bins import Bin

st.set_page_config(
    page_title="Phantasialand Waiting Times",
    page_icon="🎢",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": None,
        "Report a Bug": "https://github.com/cologneintelligence/predict-phantasialand/issues",
        "About": """
Waiting time predictions for Phantasialand attractions. 

View the code on GitHub: [predict-phantasialand](https://github.com/cologneintelligence/predict-phantasialand)

Created by [Felix B. Müller](https://github.com/felixbmuller), [CIDD](https://www.cologne-intelligence.de/) (2021)""",
    },
)


model = ModelEstimator(BEST_MODEL_PATH)


st.title("Phantasialand Waiting Times")

col1, col2 = st.columns(2)

with col1:
    attraction = st.selectbox("Attraction", WARTEZEITEN_APP_ATTRACTIONS.keys())

with col2:
    date = st.date_input(
        "Date of Visit", datetime.date.today() + datetime.timedelta(days=1)
    )

by_time_df, summary_df = model.predict(date, attraction)

by_time_df.index = pd.to_datetime(by_time_df.index)
summary_df.best_time = pd.to_datetime(summary_df.best_time).dt.strftime("%H:%M")

st.markdown(
    """

## Prediction

<style>
.heatMap {
    width: 100%;
    text-align: right;
}
.heatMap th {
background: white;
word-wrap: break-word;
text-align: center;
}
.heatMap th:nth-child(2) { background: yellow; }
.heatMap th:nth-child(3) { background: green; }
.heatMap th:nth-child(4) { background: lightblue; }
.heatMap th:nth-child(5) { background: red; }
.heatMap th:nth-child(6) { background: grey; }
</style>

<div class="heatMap">

|   | Sunny | Overcast | Light Rain | Heavy Rain | Average  |
|---|---|---|---|---|---|
"""
    f"| Average Waiting Time (min) "
    f"| {summary_df.at[Bin.DRY_SUNNY, 'mean_waiting_time']:.2f}  "
    f"| {summary_df.at[Bin.DRY_OVERCAST, 'mean_waiting_time']:.2f} "
    f"| {summary_df.at[Bin.SLIGHT_RAIN, 'mean_waiting_time']:.2f}  "
    f"| {summary_df.at[Bin.HEAVY_RAIN, 'mean_waiting_time']:.2f}   "
    f"| {summary_df.at[Bin.ALL, 'mean_waiting_time']:.2f}  |\n"
    f"| Best Time To Go | {summary_df.at[Bin.DRY_SUNNY, 'best_time']}  "
    f"| {summary_df.at[Bin.DRY_OVERCAST, 'best_time']}  "
    f"| {summary_df.at[Bin.SLIGHT_RAIN, 'best_time']}  "
    f"| {summary_df.at[Bin.HEAVY_RAIN, 'best_time']}  "
    f"| {summary_df.at[Bin.ALL, 'best_time']}   |\n"
    """
</div>


  """,
    unsafe_allow_html=True,
)


fig = px.line(
    by_time_df,
    labels={"value": "waiting time (min)", "half_hour_time": "time"},
    color_discrete_map={
        "DRY_SUNNY": "yellow",
        "DRY_OVERCAST": "green",
        "SLIGHT_RAIN": "lightblue",
        "HEAVY_RAIN": "red",
        "ALL": "grey",
    },
)
fig.layout.update(showlegend=False)

st.plotly_chart(fig)

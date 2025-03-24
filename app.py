import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

st.title("Laptop Price Predictor")


import altair as alt
import pandas as pd

price_distribution = pd.DataFrame({
    'Price': data['Price'],
    'Brand': data['Company']
})

chart = alt.Chart(price_distribution).mark_boxplot().encode(
    x='Brand',
    y='Price',
    tooltip=['Brand', 'Price']
).interactive()

st.altair_chart(chart, use_container_width=True)



company = st.selectbox('Brand', data['Company'].unique())

type = st.selectbox('Type', data['TypeName'].unique())

ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x980', 
                                                '3840x2160', '3200x1800', '2880x1800', 
                                                '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU', data['CPU_brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', data['Gpu_brand'].unique())

os = st.selectbox('OS', data['os'].unique())

if st.button('Predict Price'):
    if weight <= 0 or screen_size <= 0:
        st.error("Weight and Screen Size must be greater than 0.")
    else:
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        try:
            x_res = int(resolution.split('x')[0])
            y_res = int(resolution.split('x')[1])
            ppi = ((x_res**2 + y_res**2)**0.5) / screen_size
        except ValueError:
            st.error("Invalid screen resolution format!")

        query = np.array([company, type, ram, weight, touchscreen, ips, 
                          ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)
        
        predicted_price = np.exp(model.predict(query)[0])
        st.success(f'The Predicted Price of the Laptop is: ₹{int(predicted_price):,}')
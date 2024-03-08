import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import os

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

model = YOLO(r"runs\detect\train\weights\best.pt")

st.title('app for cars detection')

uploaded_files = st.file_uploader("Выберите изображения", accept_multiple_files=True)


if st.button("Справка"):
    st.write('hello')
    if st.button("Скрыть справку"):
        st.write('')


for file in uploaded_files:
    img = Image.open(file)
    pred = model.predict(img)[0]
    pred.save(file.name)
    filename = file.name[:file.name.rfind(".")]+'.txt'
    open(filename, 'a')
    if filename in os.listdir("qwerty"):
        os.remove("qwerty/"+filename)
    pred.save_txt('qwerty/'+filename)
    st.image(file.name, caption=file.name)
    st.download_button(
        label = "Скачать",
        data = open(filename, 'r'),
        file_name = filename
    )
    os.remove(filename)
    os.remove(file.name)

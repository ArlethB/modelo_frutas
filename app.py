import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "modelo_frutas_mobilenet/frutas_mobilenet.h5"
CLASSES_PATH = "modelo_frutas_mobilenet/class_name.json"


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_name = json.load(f)
    return model, class_name


def formatear_nombre(nombre_clase):
    """Convierte el nombre de la carpeta original del dataset (ej. 'Apple Braeburn')
    en un texto un poco más prolijo para mostrar en la interfaz. Con más de 200
    clases no es práctico traducir cada variedad al español, así que se muestra
    el nombre original con formato de título."""
    return nombre_clase.replace("_", " ").strip().title()


def predict_image(model, class_name, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1]
    return [(class_name[i], preds[i]) for i in top_idx]


st.set_page_config(page_title="Clasificador de Frutas")
st.title("Clasificador de Frutas con MobileNetV2 - José Manuel Romero Martínez")
st.write(
    "Sube una imagen de una fruta o verdura (el dataset Fruits-360 incluye ambas) "
    "y el modelo predecirá su categoría."
)

model, class_name = load_model()

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen cargada", use_container_width=True)

    results = predict_image(model, class_name, img)

    top_class, top_prob = results[0]
    st.success(f"Predicción: **{formatear_nombre(top_class)}** ({top_prob*100:.2f}%)")

    st.subheader("Top 5 probabilidades")
    for raw, prob in results[:5]:
        st.write(f"{formatear_nombre(raw)}: {prob*100:.2f}%")
        st.progress(float(prob))

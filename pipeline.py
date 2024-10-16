import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

def predict_image_classification(image_path, model_cnn, data_cat, img_height, img_width):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model_cnn.predict(img_bat)
    score = tf.nn.softmax(predict)
    return data_cat[np.argmax(score)]

def unblured_image(image_path, autoencoder_model, img_height, img_width):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)/255.0
    img_bat = tf.expand_dims(img_arr, 0)
    image_debruitee = autoencoder_model.predict(img_bat)[0]
    return image_debruitee
    

st.markdown("<h1 style='text-align: center;'>Modèle de Classification d'Images</h1>", unsafe_allow_html=True)
model_cnn = load_model('CNN/Save model2/CNN_Dataset_Model24_K3x3_IM180x180_B16_VS40_DO20_ACTsoftmax_OPTadam_E20.keras', compile=False)
data_cat = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
img_height = 180
img_width = 180

# Remplacer le champ de texte par un sélecteur multiple
image_files = os.listdir('Data pipeline')
selected_images = st.multiselect('Sélectionnez une ou plusieurs images', image_files)

use_autoencoder = st.toggle('Activer l\'autoencodeur')

# Charger l'autoencodeur si activé
if use_autoencoder:
    autoencoder_model = load_model('Auto encodeur/autoencoder_model_0_015_040_pipeline.keras', compile=False)

# Ajouter un bouton "LOAD"
if st.button('LOAD'):
    if selected_images:
        for image in selected_images:
            image_path = os.path.join('Data pipeline', image)
            
            class_image = predict_image_classification(image_path, model_cnn, data_cat, img_height, img_width)
            
            st.subheader(image)
            st.image(image_path, use_column_width='always', caption='Image originale')
            st.write(f'La catégorie de l\'image est : **{class_image}**.')
            # st.write('Avec une précision de ' + s tr(np.max(score)*100) + '%')
            
            if use_autoencoder and class_image == 'Photo':
                # Passer l'image à l'autoencodeur pour obtenir l'image débruitée
            
                image_debruitee = unblured_image(image_path, autoencoder_model, img_height, img_width)
                # Afficher l'image débruitée
                st.image(image_debruitee, use_column_width='always', caption='Image débruitée')
                # st.write("Image débruitée arrivera ici")
                
            if class_image == 'Photo':
                st.write('Le label de cette image est : **INCOMING**')
            
            st.markdown("---")  # Séparateur horizontal

    else:
        st.warning('Veuillez sélectionner au moins une image avant de cliquer sur LOAD.')
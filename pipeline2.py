import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image as PILImage

st.set_page_config(
    page_title="TouNum",  # Titre de l'onglet
    page_icon="📊",  # Icône de l'onglet (optionnel)
)

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        y = self.fc1(output)
        y = tf.reshape(y, (-1, x.shape[2]))
        y = self.fc2(y)
        return y, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

def evaluate(image_tensor):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)

    img_tensor_val = image_features_extract_model(image_tensor)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

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

@st.cache_resource
def load_models():
    model_cnn = load_model('CNN/Save model2/CNN_Dataset_Model24_K3x3_IM180x180_B16_VS40_DO20_ACTsoftmax_OPTadam_E20.keras', compile=False)
    
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    
    checkpoint_path = os.path.abspath("Captionning/checkpoints2")
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=tf.keras.optimizers.Adam())
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    with open('Captionning/tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    
    return model_cnn, encoder, decoder, image_features_extract_model, tokenizer

def process_image(image_path, use_autoencoder, _autoencoder_model=None):
    class_image = predict_image_classification(image_path, model_cnn, data_cat, img_height, img_width)
    
    image_debruitee = None
    
    original_img = PILImage.open(image_path)
    
    if use_autoencoder:
        image_debruitee = unblured_image(image_path, _autoencoder_model, img_height, img_width)
        input_img = image_debruitee
    else:
        input_img = original_img
    
    # Convertir l'image en format attendu par le modèle de captioning
    tf_img = tf.keras.preprocessing.image.img_to_array(input_img)
    tf_img = tf.image.resize(tf_img, (299, 299))
    tf_img = tf.keras.applications.inception_v3.preprocess_input(tf_img)
    tf_img = tf.expand_dims(tf_img, 0)
    
    result, _ = evaluate(tf_img)
    result = [word for word in result if word not in ('<start>', '<end>', '<unk>')]
    
    return class_image, image_debruitee, original_img, result

st.markdown("<h1 style='text-align: center;'>Modèle de Classification d'Images</h1>", unsafe_allow_html=True)

embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
max_length = 47
attention_features_shape = 64
img_height = 180
img_width = 180

model_cnn, encoder, decoder, image_features_extract_model, tokenizer = load_models()

data_cat = ["Painting", "Photo", "Schematics", "Sketch", "Text"]

image_files = os.listdir('Data pipeline')
selected_images = st.multiselect('Sélectionnez une ou plusieurs images', image_files)

use_autoencoder = st.toggle('Activer l\'autoencodeur')

if st.button('LOAD'):
    if selected_images:
        autoencoder_model = None
        if use_autoencoder:
            autoencoder_model = load_model('Auto encodeur/autoencoder_model_0_015_040_pipeline.keras', compile=False)
        
        for image in selected_images:
            image_path = os.path.join('Data pipeline', image)
            
            class_image, image_debruitee, original_img, result = process_image(image_path, use_autoencoder, _autoencoder_model=autoencoder_model) 
            
            st.subheader(image)
            st.image(original_img, use_column_width='always', caption='Image originale')
            st.write(f'La catégorie de l\'image est : **{class_image}**.')
            
            caption = ' '.join(result).capitalize() + '.'
            
            if use_autoencoder and image_debruitee is not None:
                st.image(image_debruitee, use_column_width='always', caption='Image débruitée')
                st.write(f"Description de l'image débruitée : {caption}")
            else:
                st.write(f"Description de l'image originale : {caption}")
            
            st.markdown("---")

    else:
        st.warning('Veuillez sélectionner au moins une image avant de cliquer sur LOAD.')
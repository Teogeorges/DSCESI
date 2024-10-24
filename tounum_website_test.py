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
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile

st.set_page_config(
    page_title="TouNum",
    page_icon="🤖",
    initial_sidebar_state="expanded",
    layout="wide"
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

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def plot_attention(image, result, attention_plot):
    temp_image = np.array(PILImage.open(image))
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.tight_layout()
    plt.show()

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
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

@st.cache_resource
def load_models(classification_type):
    if classification_type == "Multiclasse":
        model_cnn = load_model('CNN/Save model2/CNN_Dataset_Model24_K3x3_IM180x180_B16_VS40_DO20_ACTsoftmax_OPTadam_E20.keras', compile=True)
    else:
        model_cnn = load_model('CNN/Save model2/CNN_Dataset 1v4_Model2_K3x3_IM180x180_B16_VS40_DO20_ACTsigmoid_OPTadam_E10.keras', compile=True)
    
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

def predict_image_classification(image_path, model_cnn, data_cat, img_height, img_width, classification_type):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)
    predict = model_cnn.predict(img_bat)
    
    if classification_type == "Multiclasse":
        score = tf.nn.softmax(predict)
        return data_cat[np.argmax(score)]
    else:
        score = tf.nn.softmax(predict)
        return data_cat[np.argmax(score)]

def unblured_image(image_path, autoencoder_model, img_height, img_width):
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)/255.0
    img_bat = tf.expand_dims(img_arr, 0)
    image_debruitee = autoencoder_model.predict(img_bat)[0]
    return image_debruitee

def process_image(image_path, use_classification, use_captioning, use_autoencoder, _autoencoder_model, classification_type, captioning_model):
    class_image = None
    result = None
    image_debruitee = None
    
    if use_classification:
        class_image = predict_image_classification(image_path, model_cnn, data_cat, img_height, img_width, classification_type)
    
    if use_autoencoder:
        image_debruitee = unblured_image(image_path, _autoencoder_model, img_height, img_width)
    
    original_img = PILImage.open(image_path)
    
    if use_captioning:
        if captioning_model == "CNN GRU":
            tf_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
            tf_img = tf.keras.preprocessing.image.img_to_array(tf_img)
            tf_img = tf.keras.applications.inception_v3.preprocess_input(tf_img)
            tf_img = tf.expand_dims(tf_img, 0)
            
            result, _ = evaluate(image_path)
            result = [word for word in result if word not in ('<start>', '<end>', '<unk>')]
        else:  # PyTorch
            raw_image = PILImage.open(image_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)
            result = processor.decode(out[0], skip_special_tokens=True)
    
    return class_image, image_debruitee, original_img, result

# @st.cache_resource
def load_pytorch_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# Configuration de la barre latérale
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 2em;'>Paramètres ⚙️</h1>", unsafe_allow_html=True)
    st.markdown("---")
    use_classification = st.toggle('Activer la classification', value=True)
    if use_classification:
        classification_type = st.radio("Type de classification", ["Multiclasse", "Binaire"], index=0)
    use_autoencoder = st.toggle('Activer l\'autoencodeur', value=False)
    use_captioning = st.toggle('Activer le captioning', value=True)
    if use_captioning:
        captioning_model = st.radio("Modèle d'étiquetage", ["BLIP", "CNN GRU"], index=0)
        if captioning_model == "CNN GRU":
            max_length = st.slider("Longueur maximale de la description", min_value=10, max_value=50, value=35, step=1)

st.markdown("<h1 style='text-align: center; font-size: 5em;'>🤖 TouNum</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; padding-bottom: 20px;'>L'intelligence artificielle au service de votre patrimoine numérique!</h2>", unsafe_allow_html=True)
st.markdown(" **Made by:** `HAIK .K`, `HUGO .M` **et** `TÉO .G`.", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h2 style='text-align: center;'> L'outil qui permet de classer vos images ainsi que de générer une description !</h2>", unsafe_allow_html=True)

embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
attention_features_shape = 64
img_height = 180
img_width = 180

model_cnn, encoder, decoder, image_features_extract_model, tokenizer = load_models(classification_type if use_classification else "Multiclasse")
autoencoder_model = load_model('Auto encodeur/autoencoder_model_0_015_040_pipeline.keras', compile=False)
processor, model = load_pytorch_model()

if use_classification:
    if classification_type == "Multiclasse":
        data_cat = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
    else:  
        data_cat = ["Other", "Photo"]

image_files = os.listdir('Data pipeline')
uploaded_files = st.file_uploader("Sélectionnez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button('CHARGER'):
    if uploaded_files:
        with st.spinner('Chargement en cours...'):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                class_image, image_debruitee, original_img, result = process_image(
                    temp_path, 
                    use_classification, 
                    use_captioning, 
                    use_autoencoder, 
                    _autoencoder_model=autoencoder_model, 
                    classification_type=classification_type if use_classification else None, 
                    captioning_model=captioning_model if use_captioning else None
                ) 
                
                st.subheader(uploaded_file.name)
                st.image(original_img, use_column_width='always', caption=uploaded_file.name)
                
                if use_classification and class_image:
                    st.write(f'La catégorie de l\'image est : **{class_image}**.')
                
                if use_autoencoder and image_debruitee is not None and (not use_classification or class_image == 'Photo'):
                    st.image(image_debruitee, use_column_width='always', caption='Image débruitée')
                
                if use_captioning and result:
                    caption = result if captioning_model == "BLIP" else ' '.join(result).capitalize() + '.'
                    st.write(f"<u>Description de l'image</u> : {caption}", unsafe_allow_html=True)
                
                st.markdown("---")

                os.unlink(temp_path)
        
    else:
        st.warning('Veuillez sélectionner au moins une image avant de cliquer sur le bouton CHARGER.')
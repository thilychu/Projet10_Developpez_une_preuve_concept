#!/usr/bin/env python
# coding: utf-8

# # Implémentation un moteur d’inférence API . 
# L'objectif de l’API est de renvoyer le sentiment à réception d’un texte brut 'API .

# In[42]:

import numpy as np 
import flask
from flask import Flask, jsonify, request, render_template

import tensorflow as tf
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
from transformers import AutoTokenizer, TFAlbertModel

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def create_model(base_model,max_len):
    # Params
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    accuracy = tf.keras.metrics.BinaryAccuracy()

    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')

    embeddings = base_model(input_ids, attention_mask=attention_masks)[0][:,0,:]

    # Modifiez le nombre de classes de sortie à 1 pour la classification binaire
    output = tf.keras.layers.Dense(1, activation="sigmoid")(embeddings)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

    model.compile(opt, loss=loss, metrics=[accuracy])

    return model

# Load model
maxlen = 70
dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
model = create_model(dbert_model,maxlen)
model.load_weights("model/best_DistilBERT.h5")

@app.route('/') # default route
def index():
    return render_template('index.html') #

@app.route('/predict', methods = ['GET','POST']) 
def predict_sentiment():

    text = request.form['name']
    custom_encoding = dbert_tokenizer.encode_plus(text,add_special_tokens=True,max_length=maxlen,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='tf')

    custom_input_ids = custom_encoding['input_ids']
    custom_mask = custom_encoding['attention_mask']
    
    custom_pred = model.predict([custom_input_ids, custom_mask])[0][0]
    custom_final_pred = np.where(custom_pred >= 0.5, 1, 0)

    # Display the prediction result
    if custom_final_pred == 1:
        result="The sentiment of this text is "+str(int(custom_pred*100))+"% positive"
    else:
        result="The sentiment of this text is "+str(int((1-custom_pred)*100))+"% negative"
	
    return jsonify(result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)






import tqdm
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import cv2


def load_tokenizer(path):
    # Load the tokenizer from JSON
    with open(path, 'r') as json_file:
        tokenizer_json = json.load(json_file)
        tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer


def feature_extractor(dt, col_name, img_path, inp_shp, model, imgM):
    features = {}
    for image_name in tqdm.tqdm(dt[col_name].unique().tolist()):
        img_full_path = os.path.join(img_path, image_name)
        feature = imgM.img_load_and_extrac_feature(
            img_full_path, inp_shp, model)
        features[image_name] = feature

    return features


def load_pret_model(path='/temp', pre_model=None):
    if os.path.exists(path) and (pre_model is None):
        model = keras.models.load_model(path)
        return model
    elif (pre_model is not None) and os.path.exists(path):
        pre_model.load_weights(path)
        return pre_model

    else:
        return pre_model


def make_prediction(features, tokenizer, max_len, model, name):
    start_word = 'startseq'
    feature = features[name]
    indx_word = tokenizer.index_word
    for i in range(max_len):
        # first convert it into sequence
        seq = tokenizer.texts_to_sequences([start_word])[0]
        in_seq = pad_sequences([seq], maxlen=max_len, padding='post')
        prediction = model.predict([feature, in_seq])
        indx = np.argmax(prediction)
        # converting the word into
        pred_word = indx_word[indx]
        start_word = start_word+' '+pred_word
        if pred_word == 'endseq':
            return start_word


def readImage(path, img_out_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_out_size[0], img_out_size[1]))
    return img

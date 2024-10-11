from utils import load_tokenizer, load_pret_model
import os
import pickle
import numpy as np
from imageFeatureExtractor import buildimageModel
from tensorflow.keras.preprocessing.sequence import pad_sequences


class makePrediction:
    def __init__(self, config_class):
        self.conf_ = config_class
        path = os.path.join(
            os.path.curdir, self.conf_.base_net_name, 'tokenizer.json')
        self.tokenizer = load_tokenizer(path)
        model_name = os.path.join(
            os.path.curdir, self.conf_.base_net_name, "finalmodel.weights.keras")
        self.model = load_pret_model(
            path=model_name)
        self.word_index_ = self.tokenizer.word_index
        self.vocab_size = len(self.word_index_)+1
        path = os.path.join(
            os.path.curdir, self.conf_.base_net_name, 'max_len.pkl')
        with open(path, 'rb') as file:
            self.max_len = pickle.load(file)
        file.close()
        # init base net
        self.imgmodel = buildimageModel(
            self.conf_.base_net_name, self.conf_.inp_shp)
        self.base_net = self.imgmodel.initBaseNet()

    def predict(self, image_):
        feature = self.imgmodel.img_load_and_extrac_feature(
            image_, self.conf_.inp_shp, self.base_net)
        start_word = 'startseq'
        indx_word = self.tokenizer.index_word
        for _ in range(self.max_len):
            # first convert it into sequence
            seq = self.tokenizer.texts_to_sequences([start_word])[0]
            in_seq = pad_sequences([seq], maxlen=self.max_len, padding='post')
            prediction = self.model.predict([feature, in_seq])
            indx = np.argmax(prediction)
            # converting the word into
            pred_word = indx_word[indx]
            start_word = start_word+' '+pred_word
            if pred_word == 'endseq':
                break
        return ' '.join(start_word.split(' ')[1:-1])

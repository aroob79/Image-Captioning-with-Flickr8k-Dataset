import re
import os
import json
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from imageFeatureExtractor import buildimageModel
from tensorflow.keras.utils import plot_model
from utils import feature_extractor


class prep_data:
    def __init__(self, conf_):
        self.conf_ = conf_
        self.dataset_path = self.conf_.dataset_path
        self.img_path = self.conf_.img_path

    def TextProcess(self, captions):
        # make the lower case letter
        captions[self.conf_.caption_col_name] = captions[self.conf_.caption_col_name].apply(
            lambda x: x.lower())
        # remove all special charecter and number
        captions[self.conf_.caption_col_name] = captions[self.conf_.caption_col_name].apply(
            lambda x: re.sub("[^A-Za-z]", " ", x))
        # replace multiple white space with one space
        captions[self.conf_.caption_col_name] = captions[self.conf_.caption_col_name].apply(
            lambda x: re.sub("/s+", " ", x))
        # remove single charecter
        captions[self.conf_.caption_col_name] = captions[self.conf_.caption_col_name].apply(
            lambda x: ' '.join([word for word in x.split(' ') if len(word) > 1]))
        # add a starting and ending text
        captions[self.conf_.caption_col_name] = captions[self.conf_.caption_col_name].apply(
            lambda x: 'startseq '+x+' endseq')
        return captions[self.conf_.caption_col_name].tolist()

    def preprocess_dt(self):
        # load the data
        self.captions = pd.read_csv(self.dataset_path)
        pro_text = self.TextProcess(self.captions)

        return self.captions, pro_text

    def tokenizer_(self):
        _, pro_text = self.preprocess_dt()
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(pro_text)
        self.word_index_ = self.tokenizer.word_index
        self.vocab_size = len(self.word_index_)+1
        self.max_len = max([len(sent.split(' ')) for sent in pro_text])

        os.makedirs(self.conf_.base_net_name, exist_ok=True)
        path11 = os.path.join(os.path.curdir, self.conf_.base_net_name)

        # Save tokenizer to JSON
        tokenizer_json = self.tokenizer.to_json()
        with open(os.path.join(path11, 'tokenizer.json'), 'w') as json_file:
            json.dump(tokenizer_json, json_file)
        with open(os.path.join(path11, 'max_len.pkl'), 'wb') as file:
            pickle.dump(self.max_len, file)
        file.close()

        return self.tokenizer, self.word_index_, self.vocab_size, self.max_len

    def imageModel(self, plot_=False, summary_=False):

        self.imgM = buildimageModel(base_net_name=self.conf_.base_net_name,
                               input_shape=self.conf_.inp_shp)
        self.model = self.imgM.initBaseNet()

        # get the final model
        _ = self.tokenizer_()
        finalModel = self.imgM.getFinalModel(
            self.model.output.shape[-1], self.max_len, self.vocab_size)

        if plot_ == True:
            plot_model(finalModel)

        if summary_ == True:
            finalModel.summary()

        return finalModel, self.model

    def imageFeature(self): 
        path11 = os.path.join(os.path.curdir, self.conf_.base_net_name,'features.pkl')
        if (self.conf_.save_feature == False) and os.path.exists(path11):    
            with open(path11,'rb') as file:
                self.img_features=pickle.load(file)
            file.close()
            return self.img_features

        self.img_features = feature_extractor(self.captions, self.conf_.image_col_name,
                                              self.conf_.img_path, self.conf_.inp_shp, self.model, self.imgM)
        
        with open(path11,'wb') as file:
                pickle.dump(self.img_features,file)
        file.close()
        return self.img_features

    def splitDF(self):
        split_size = int(self.conf_.train_size * len(self.captions))
        train_images = self.captions[self.conf_.image_col_name][:split_size]
        val_images = self.captions[self.conf_.image_col_name][split_size:]
        self.train_df = self.captions[self.captions[self.conf_.image_col_name].isin(
            train_images)]
        self.val_df = self.captions[self.captions[self.conf_.image_col_name].isin(
            val_images)]
        self.train_df.reset_index(inplace=True, drop=True)
        self.val_df.reset_index(inplace=True, drop=True)

        return self.train_df, self.val_df

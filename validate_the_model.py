
import numpy as np
from utils import load_pret_model
from utils import make_prediction
import nltk
from nltk.translate.bleu_score import sentence_bleu
# Download WordNet
nltk.download('wordnet')


class validation:
    def __init__(self, config_file_class, prep_data_class, model_path='none', pre_load_model=None):
        self.conf_ = config_file_class
        self.data_prep_class = prep_data_class
        self.model_path = model_path
        self.pre_load_model = pre_load_model

    def make_prediction_all(self, dataset, features, tokenizer, max_len, model):
        # find the unique name
        unique_img = np.unique(dataset[self.conf_.image_col_name].values)
        true_cap = []
        pred_cap = []
        for name in unique_img:
            # find the real caption
            temp_real = dataset[dataset[self.conf_.image_col_name].isin(
                [name])][self.conf_.caption_col_name].values.reshape((-1, 1)).tolist()
            temp_pred = make_prediction(
                features, tokenizer, max_len, model, name)
            true_cap.append(temp_real)
            pred_cap.append([temp_pred])

        return true_cap, pred_cap

    # Tokenization function
    def tokenize(self, sentence):
        return sentence.split()

    def validate(self):
        model = load_pret_model(
            path=self.model_path, pre_model=self.pre_load_model)
        tc, pc = self.make_prediction_all(
            self.data_prep_class.val_df, self.data_prep_class.img_features,
            self.data_prep_class.tokenizer, self.data_prep_class.max_len, model)

        # Initialize metrics
        score=[]
        for i, prediction in enumerate(pc):
            if (prediction[0]) == None:
                continue
            # Tokenize prediction
            tokenized_prediction = self.tokenize(prediction[0])

            # Calculate BLEU
            bleu_scores = [sentence_bleu([self.tokenize(ref[0])], tokenized_prediction, weights=(
                0.25, 0.25, 0.25, 0.25)) for ref in tc[i]]
            average_bleu = sum(bleu_scores) / len(bleu_scores)
            score.append(average_bleu)

        # Print average scores
        
        print("Average BLEU:", sum(score)/len(score))

        return 0

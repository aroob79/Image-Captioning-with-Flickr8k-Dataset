import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical


class custom_data_loder(Sequence):
    def __init__(self, df, x_col_name, y_col_name, batch_size, shuffle, features, tokenizer, vocab_size, max_len):
        self.df = df
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.features = features
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self):
        return len(self.df)//self.batch_size

    def on_epoch_end(self):

        if self.shuffle:
            # shuffle the data
            self.df = self.df.sample(
                frac=1, random_state=42).reset_index(drop=True)

    def __getitem__(self, index):
        self.indices = np.arange(len(self.df))
        batch_indices = self.indices[index *
                                     self.batch_size: (index+1) * self.batch_size].tolist()
        img_name = self.df.loc[batch_indices, [self.x_col_name]].values
        captions = self.df.loc[batch_indices, [self.y_col_name]].values

        x1, x2, y = self.__data_generation(img_name, captions)
        return (x1, x2), y

    def __data_generation(self, img_paths, labels):
        x1 = []
        x2 = []
        y = []

        for img_path, label in zip(img_paths, labels):
            img_arr = self.features[img_path[0]][0]
            seq = self.tokenizer.texts_to_sequences(label)[0]
            for i in range(1, len(seq)):
                in_seq = pad_sequences(
                    [seq[:i]], maxlen=self.max_len, padding='post')[0]
                out_seq = to_categorical([seq[i]], num_classes=self.vocab_size)

                x1.append(img_arr)
                x2.append(in_seq)
                y.append(out_seq)

        x1, x2, y = np.array(x1), np.array(x2), np.array(y)
        return x1, x2, y

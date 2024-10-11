import os
from customDataLoader import custom_data_loder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class modelTrain:
    def __init__(self, config_file_class, prep_data_class, num_of_epochs, batch_size):
        self.conf_ = config_file_class
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.data_prep_class = prep_data_class
        self.model_name = os.path.join(
            os.path.curdir, self.conf_.base_net_name, "finalmodel.weights.keras")

    def setModelParameter(self):
        checkpoint = ModelCheckpoint(self.model_name,
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True,
                                     verbose=3)

        earlystopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=7,
                                                    verbose=1,
                                                    factor=0.2,
                                                    min_lr=0.00000001)
        return checkpoint, earlystopping, learning_rate_reduction

    def initModelAndData(self, plot_=False, summary_=False):
        self.finalModel, self.baseModel = self.data_prep_class.imageModel(
            plot_, summary_)
        imgFeature = self.data_prep_class.imageFeature()
        train_df, valid_df = self.data_prep_class.splitDF()
        self.train_data_gen = custom_data_loder(train_df, self.conf_.image_col_name, self.conf_.caption_col_name, self.batch_size, shuffle=True,
                                                features=imgFeature, tokenizer=self.data_prep_class.tokenizer, vocab_size=self.data_prep_class.vocab_size, max_len=self.data_prep_class.max_len)

        self.val_data_gen = custom_data_loder(valid_df,  self.conf_.image_col_name, self.conf_.caption_col_name, self.batch_size, shuffle=True,
                                              features=imgFeature, tokenizer=self.data_prep_class.tokenizer, vocab_size=self.data_prep_class.vocab_size, max_len=self.data_prep_class.max_len)
        self.finalModel.compile(
            loss='categorical_crossentropy', optimizer='adam')

        return self.train_data_gen, self.val_data_gen

    def trainModel(self):
        checkpoint, earlystopping, learning_rate_reduction = self.setModelParameter()

        summ = self.finalModel.fit(self.train_data_gen,
                                   epochs=self.num_of_epochs,
                                   validation_data=self.val_data_gen,
                                   callbacks=[checkpoint, earlystopping, learning_rate_reduction])
        return summ, self.finalModel

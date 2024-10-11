from tensorflow import keras
from tensorflow.keras.applications import ResNet50, DenseNet201, VGG19
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


class buildimageModel:
    def __init__(self, base_net_name='densenet', input_shape=(224, 224,3)):
        self.base_net_name = base_net_name
        self.input_shape = input_shape

    def initBaseNet(self):
        input_ = keras.Input(shape=self.input_shape)
        if self.base_net_name == 'vgg19':
            vgg = VGG19(weights='imagenet', include_top=True,
                        input_tensor=input_)
            base_model = Model(inputs=input_,
                               outputs=vgg.layers[-2].output)

        elif self.base_net_name == 'resnet50':
            resnet = ResNet50(weights='imagenet',
                              include_top=True, input_tensor=input_)
            base_model = Model(inputs=input_,
                               outputs=resnet.layers[-2].output)

        else:
            densenet = DenseNet201(
                weights='imagenet', include_top=True, input_tensor=input_)
            base_model = Model(inputs=input_,
                               outputs=densenet.layers[-2].output)
        model = Model(inputs=input_, outputs=base_model.output)
        return model

    def img_load_and_extrac_feature(self, img_name, img_out_size, model):
        img = load_img(img_name, color_mode='rgb', target_size=(img_out_size[0],img_out_size[1]))
        img = img_to_array(img)
        img = img/255
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)
        return feature

    def getFinalModel(self, inp_, max_len, vocab_size):
        input1 = keras.Input(shape=(inp_,))
        img_x1 = keras.layers.Dense(512, activation='relu')(input1)
        img_x1 = keras.layers.Dense(256, activation='relu')(img_x1)
        img_x2 = keras.layers.Reshape([1, 256])(img_x1)

        # define text layers
        input2 = keras.Input(shape=(max_len, 1))
        text_x1 = ly = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=256, mask_zero=False)(input2)
        text_x1 = keras.layers.Reshape([-1, 256])(text_x1)
        # merge both input
        merge = keras.layers.concatenate([img_x2, text_x1], axis=1)
        merge_x1 = keras.layers.LSTM(256)(merge)

        # add to layers
        add_x1 = keras.layers.add([merge_x1, img_x2])
        x1 = keras.layers.Dense(128, activation='relu')(add_x1)
        x1 = keras.layers.Dense(64, activation='relu')(x1)
        x1 = keras.layers.Dropout(0.3)(x1)
        output = keras.layers.Dense(vocab_size, activation='softmax')(x1)

        final_model = Model(inputs=[input1, input2], outputs=output)

        return final_model

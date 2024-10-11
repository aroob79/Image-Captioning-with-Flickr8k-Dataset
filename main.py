from configaration import config
from data_preparation import prep_data
from model_train import modelTrain
from validate_the_model import validation
from prediction import makePrediction
from matplotlib import pyplot as plt
from textwrap import wrap
import tkinter as tk
from tkinter import filedialog
from utils import readImage

isTrain = False
config = config(img_path=r'E:\python\basic_code\image_captioning\Images',
                dataset_path=r'E:\python\basic_code\image_captioning\captions.txt',
                base_net_name='densenet',
                image_col_name='image',
                caption_col_name='caption',
                inp_shp=(224, 224, 3),
                save_feature=False
                )

if isTrain:
    # prepare the data
    preprocessing = prep_data(config)
    # initate the model and important parameter
    train_model = modelTrain(config, preprocessing,
                             num_of_epochs=50, batch_size=40)
    _ = train_model.initModelAndData()
    # training the model and get the train model
    summ, finalM = train_model.trainModel()

    # validate the model
    validate = validation(config, preprocessing, pre_load_model=finalM)
    validate.validate()
else:
    def select_file():
        # Initialize the root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open file dialog and ask the user to select a file
        file_path = filedialog.askopenfilename(initialdir="C:/",
                                               title="Select a file",
                                               filetypes=(("all files", "*.*"),
                                                          ("text files", "*.txt")))

        # Return the selected file path
        if file_path:
            print(f"File selected: {file_path}")
            return file_path
        else:
            print("No file selected.")
            return None
    selected_file = select_file()
    # read the image
    image1 = readImage(selected_file, config.inp_shp)
    # initiate the prediction class for prediction
    predic_class = makePrediction(config)
    # provide the path of the test image
    text = predic_class.predict(selected_file)
    plt.figure(figsize=(60, 60))

    plt.imshow(image1)
    plt.title("\n".join(wrap(text, 20)))
    plt.axis("off")
    plt.show()

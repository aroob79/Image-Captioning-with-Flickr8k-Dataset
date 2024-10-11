### Image Captioning with Flickr 8k Dataset

This is an implementation of the image captioning task using TensorFlow on the Flickr8k dataset. The model was trained using three base networks: DenseNet, ResNet50, and VGG19.

## Dataset

The dataset should contain a CSV or TXT file with two columns: one for the name of each image and the other for the caption of each image. The images should be stored in a separate folder. For example:

```
image_captioning/
    ├── Images/ (folder)
    ├── captions.txt
```

In this project, there are a total of 8,000 images, and each image has 5 separate captions stored in the `captions.txt` file.  
The dataset can be found at [Kaggle - Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).

## Training and Validation

For training and validation, some parameters need to be set in the config file provided in the `main.py` file. In this case, 85% of the data was used for training and 15% for validation. The model will create a folder for each base network and store the trained weights and other parameters.

For the three base networks, three separate folders will be created in the current directory, which can later be used for prediction.  
The trained weights and parameters can be found [here](https://drive.google.com/drive/u/4/folders/1v_PGIYP4oGpotGKMpfsjNdX2XY3Lgg-y).

## Output
![image](https://github.com/user-attachments/assets/9061a89e-6d75-4db5-909d-8aba83d6303a)   
![image](https://github.com/user-attachments/assets/a53a5466-04bf-45fd-b926-c855a8b7997e)





    

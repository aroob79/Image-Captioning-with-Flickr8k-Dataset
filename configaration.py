class config:
    def __init__(self, img_path: "image folder path ",
                 dataset_path: "path of the data frame ",
                 base_net_img_size=224,
                 base_net_name: "densenet | resnet50 | vgg19" = 'densenet',
                 image_col_name: "name of the columns stored image name in the df" = "image",
                 caption_col_name: "name of the columns stored caption in the df" = 'caption',
                 train_size=0.85,
                 inp_shp: "imput shape of the base net " = (224, 224,3),
                 save_feature=True
                 ):
        # define all parameter that used in the code
        self.dataset_path = dataset_path
        self.base_net_img_size = base_net_img_size
        self.base_net_name = base_net_name
        self.image_col_name = image_col_name
        self.caption_col_name = caption_col_name
        self.img_path = img_path
        self.inp_shp = inp_shp
        self.train_size = train_size 
        self.save_feature=save_feature

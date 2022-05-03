import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class RISE:
    """
    A class retrieving the image from the extracted-features Image database.

    """
    def __init__(self,model = None, features_vectors = [], paths_vectors = []):
        """
        Parameters
        ----------
        model : model
        features_vectors : list
        paths_vectors : list
        """
        self.load_model(model) if model else self.default_model(model)
        self.features_vectors = features_vectors
        self.paths_vectors = paths_vectors


    def load_model(self,model):
        """
        Parameters
        ----------
        model : model
            load the model to the search engine
        """
        self.model = model

    def default_model(self,):
        """Default model is set to ResNet50 from TF library
        """
        self.model = ResNet50(weights = "imagenet", include_top = False,
                              input_shape = (224,224,3), polling = "max")


    def extract_features(self,img_path):
        """
        Parameters
        ----------
        img_path : string, required

        """
        input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        features = self.model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)

        return normalized_features

    def load_features_vectors(self,vectors_features_path, address_vectors_path):
        """
        Load features_vectors from files_path
        """
        self.load
        self.features_vectors_list = pickle.load(open(vectors_features_path,'rb'))
        self.paths_vectors_list = pickle.load(open(address_vectors_path,'rb'))

    def load_images(self,   )

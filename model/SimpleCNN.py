import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

### assuming input (None, 5, 6)

class SimpleCNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.Preprocess = lambda x: tf.expand_dims(x, axis=3)
        self.Reshape = layers.Reshape((5,6), input_shape=(30, ))
        self.CNNs = [
            layers.Conv2D(32, (2,2), activation='sigmoid', padding='same'),
            layers.MaxPool2D(pool_size=(2,2), strides=1),
            layers.BatchNormalization(),
        ]
        self.Flatten = layers.Flatten()
        self.Denses = [
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
        
    def call(self, inputs): 
        x = self.Preprocess(inputs)
        for CNN in self.CNNs: x = CNN(x)
        x = self.Flatten(x)
        for Dense in self.Denses: x = Dense(x)
        return x
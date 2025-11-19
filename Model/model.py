import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

if __name__ == "__main__":
    model = models.Sequential([
      layers.Dense(units=16, activation='relu', input_shape=(3,)),
      layers.Dense(units=16, activation='relu'),
      layers.Dense(units=1, activation='linear') 
])
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
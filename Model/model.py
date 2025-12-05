import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import json
import pandas as pd
import math

INPUT_DIM = 2

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(df['n'])

    train_set = []
    train_labels = []

    for row in df.itertuples():
        pathsize = row.n - 1
        tour_order = row.tourOrder
        start = tour_order[0]
        dest = tour_order[-2]
        start_point = row.Vcoords[start - 1]
        end_point = row.Vcoords[dest - 1]
        sl_distance = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

        train_set.append([pathsize, sl_distance])
        train_labels.append(row.tourCost)
    
    ts_df = pd.DataFrame(train_set, columns=['PathSize', 'Distance'])
    print(ts_df)

    return np.array(train_set), np.array(train_labels)

if __name__ == "__main__":
    x_train, y_train = load_data('./trainingGraphs/5-5000.json')
    print(f"Training set: {x_train}")
    print(f"Training labels: {y_train}")
    x_test, y_test = load_data('./validationGraphs/5-1000_1.json')
    model = models.Sequential([
      layers.Dense(units=16, activation='relu', input_shape=(INPUT_DIM,)),
      layers.Dense(units=16, activation='relu'),
      layers.Dense(units=1, activation='linear') 
    
    ])
    
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest loss:', test_loss)

    model.save('first_model.keras')
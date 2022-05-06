from dataset import *
from dataset.dataset import get_balanced_dataset, process_dataset

import tensorflow as tf
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import seaborn as sns


from config import *

LOG_DIR='origin_ANN_sequential.log'

def main():
    ### STEP 1: PREPARE DATASET
    train_features, train_labels, \
    val_features, val_labels, \
    test_features, test_labels = process_dataset(DATASET_DIR or DATASET_URL)
    log('Complete preparing dataset.')

    ### STEP 2: Build model
    model = Sequential([
        Input(shape=(30,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    log('Complete building model.')
    
    model.build(input_shape = (30,))
    log(model.summary())
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics = ['accuracy', Precision(), Recall()]
    )
    
    model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        CKPT_DIR,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch',
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=10,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )
    
    ### STEP 3: Train
    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=100,
        verbose=1,
        callbacks=[model_checkpoint_callback, early_stopping]
    )
    model.fit(train_features, train_labels)
    log('Complete training.')       
    
    ### STEP 4: Eval
    y_pred = model.predict(test_features)[:, 0]
    print(y_pred)
    print(y_pred.shape)
    y_pred = [round(i) for i in y_pred]
    confusion_m = confusion_matrix(test_labels, y_pred)
    clf_report = classification_report(test_labels, y_pred)
    
    log(f'Confusion Matrix\n{confusion_m}', LOG_DIR)
    log(f'Classification Report\n{clf_report}', LOG_DIR)
    

if __name__ == "__main__":
    main()

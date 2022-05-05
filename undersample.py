from distutils.command.config import config
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import dataset.dataset
import itertools
from sklearn.metrics import confusion_matrix

DATASET="data/creditcard.csv"
# Create a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__=="__main__":
    train,val,test= dataset.dataset.process_dataset(DATASET)
    X_train,y_train = dataset.dataset.get_balanced_dataset(train[0],train[1],False)

    n_inputs = X_train.shape[1]

    undersample_model = Sequential([
        Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])


    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    undersample_model.compile(
        Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[
            "accuracy", precision, recall
        ]
    )
    undersample_model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        batch_size=300, 
        epochs=20, 
        shuffle=True, 
        verbose=2
    )
    predict_x = undersample_model.predict(test[0]) 
    undersample_fraud_predictions = np.argmax(predict_x, axis=1)

    undersample_condensed = confusion_matrix(test[1], undersample_fraud_predictions)
    actual_cm = confusion_matrix(test[1], test[1])
    labels = ['No Fraud', 'Fraud']

    fig = plt.figure(figsize=(16,8))

    print("most vote model")
    plt1=fig.add_subplot(231)
    plot_confusion_matrix(
        undersample_condensed,
        labels,
        title="Undersample (Condensed) \n Confusion Matrix",
        cmap=plt.cm.Oranges
    )
    plt1.title.set_text('most vote model')


    MLPC = MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000)
    MLPC.fit(X_train, y_train)
    predict = MLPC.predict(test[0])
    undersample_fraud_predictions = np.argmax(predict_x, axis=1)

    undersample_condensed = confusion_matrix(test[1], undersample_fraud_predictions)
    actual_cm = confusion_matrix(test[1], test[1])    
    print("97 recall model")
    plt3=fig.add_subplot(232)
    plot_confusion_matrix(undersample_condensed, labels, title="Undersample (Condensed) \n Confusion Matrix", cmap=plt.cm.Oranges)
    plt3.title.set_text('97 recall model')

    plt4=fig.add_subplot(233)
    plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

    plt.show()

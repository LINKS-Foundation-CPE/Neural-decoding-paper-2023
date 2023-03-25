import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import math 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout,TimeDistributed,Bidirectional
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import logging
import argparse

def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y

def lr_scheduler(epoch, lr):
    if epoch == 5:
        return lr/10.
    else:
        return lr

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    parser.add_argument("-m", "--model", help="Keras trained model", default='',
                        type=str)

    args = parser.parse_args()
    data_prefix = '_'.join(os.path.normpath(args.dataset).split(os.sep)[-2:])

    logging.info('\n')
    logging.info('----------------')
    logging.info('Building windows')
    logging.info(data_prefix)
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')
    
    
    trainset_bin_path = os.path.join(args.dataset, 'binary_trainset.npz')
    valset_bin_path = os.path.join(args.dataset, 'binary_valset.npz')
    testset_bin_path = os.path.join(args.dataset, 'binary_testset.npz')
    
    train_bin_set, label_bin_train = load_data(trainset_bin_path)
    val_bin_set, label_bin_val = load_data(valset_bin_path)
    test_bin_set, label_bin_test = load_data(testset_bin_path)
    
    if args.model == '':
        # Primo classificatore Grasp / no Grasp
        print('\n')
        model = Sequential()
        model.add(Bidirectional(LSTM(40, return_sequences = True, dropout=0.8, kernel_regularizer='l2', recurrent_regularizer='l2'), input_shape = (train_bin_set.shape[1], train_bin_set.shape[2])))
        model.add(Bidirectional(LSTM(units = 40, return_sequences = False, dropout=0.8, kernel_regularizer='l2', recurrent_regularizer='l2')))
        model.add(Dense(label_bin_train.shape[1], activation = "softmax"))
        model.summary()

        opt = keras.optimizers.Adam(learning_rate=0.001)
        callback = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=0.0001)
        callback1 = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

        model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = "accuracy") # binary perchè sono due classi
        model.fit(train_bin_set, label_bin_train, epochs = 20, batch_size = 256, validation_data=(val_bin_set, label_bin_val), callbacks=[callback,callback1])

        model.save(data_prefix+'_binary_model')
    else:
        model = keras.models.load_model(args.model)

    predicted_bin_value = model.predict(test_bin_set)
    predicted_bin_value = predicted_bin_value.argmax(axis=1)
    label_bin_test = label_bin_test.argmax(axis=1)
    
    #confusion matrix
    
    cm1 = confusion_matrix(label_bin_test,predicted_bin_value)
        
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm1, annot=cm1, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('binary_classes_matrix_BRNN.png')
    plt.close()
    print('Binary classes report')
    print(classification_report(label_bin_test, predicted_bin_value))
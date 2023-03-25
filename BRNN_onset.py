import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense,LSTM,Bidirectional
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import argparse

def load_data(filepath):
    '''
    Load pre-processed data
    '''
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    y = windows_dataset['y']
    return X, y

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    parser.add_argument("-m", "--model", help="Keras pre-trained model for validation only", default='',
                        type=str)

    args = parser.parse_args()
    data_prefix = '_'.join(os.path.normpath(args.dataset).split(os.sep)[-2:])
    
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
import tensorflow.keras as keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense,LSTM,Bidirectional
from sklearn.metrics import classification_report
import os
import numpy as np
import argparse
from plotting_utils import *

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
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension",   default='data/ZRec50_Mini_40_binned_spiketrains/lookback_10_lookahead_0',
                        type=str)
    parser.add_argument("-m", "--model", help="Keras trained model", default='',
                        type=str)

    args = parser.parse_args()
    data_prefix = '_'.join(os.path.normpath(args.dataset).split(os.sep)[-2:])

    print('\n')
    print('----------------')
    print('Building windows')
    print(data_prefix)
    print('----------------')
    print('\n')
    print('Loading data...')

    
    trainset_multi_path = os.path.join(args.dataset, 'multi_trainset.npz')
    valset_multi_path = os.path.join(args.dataset, 'multi_valset.npz')
    testset_multi_path = os.path.join(args.dataset, 'multi_testset.npz')
    
    train_multi_set, label_multi_train = load_data(trainset_multi_path)
    val_multi_set, label_multi_val = load_data(valset_multi_path)
    test_multi_set, label_multi_test = load_data(testset_multi_path)

    if args.model == '':
        model = Sequential()
        model.add(Bidirectional(LSTM(40, return_sequences = False, dropout=0.8, kernel_regularizer='l2', recurrent_regularizer='l2'), 
                                input_shape = (train_multi_set.shape[1], train_multi_set.shape[2])))
#         model.add(Bidirectional(LSTM(units = 40, return_sequences = False, dropout=0.8, kernel_regularizer='l2', recurrent_regularizer='l2')))
        model.add(Dense(label_multi_train.shape[1], activation = "softmax"))
        model.summary()

        opt = keras.optimizers.Adam(learning_rate=0.001)

        callback = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=0.00005)
        callback2 = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)

        model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])  #categorical perchè sono n classi

        tf.keras.utils.plot_model(model, show_layer_names=False, to_file='model.png', show_shapes=True, dpi=300)

        model.fit(train_multi_set, label_multi_train, epochs = 100, batch_size = 256, validation_data=(test_multi_set, label_multi_test), callbacks=[callback, callback2], use_multiprocessing=True)
  
        model.save(data_prefix+'_multi_model')
    
    else:
        model = keras.models.load_model(args.model)
    
    predicted_multi_value = model.predict(test_multi_set)
    predicted_multi_value = predicted_multi_value.argmax(axis=1)
    label_multi_test = label_multi_test.argmax(axis=1)
    
    #confusion matrix
    print('All classes report')
    print(classification_report(label_multi_test, predicted_multi_value))

    conf_matrix, my_labels = my_confusion_matrix(predicted_multi_value, label_multi_test, return_labels=True)
    errors_distribution = distance_from_diagonal(conf_matrix)

    # Displaying
    display_testing_phase_results(conf_matrix, label_multi_test, my_labels, errors_distribution, label_multi_train.shape[1], network_name=data_prefix[:6])
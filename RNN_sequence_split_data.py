import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse

 #Map identical objects into the same classes
def obj_mapping(id):
    if id == 11:
        return 41
    if id == 13:
        return 34
    if id == 14:
        return 44
    if id == 16:
        return 25
    if id == 12:
        return 54    
    return id

def prepare_detection_dataset(dirpath,df,lookback, lookahead):
    '''
    Prepare dataset for movement onset detection task
    Extract sub-sequences with a sliding window from trials
    Associate to each sequence either 0 (rest) or 1 (grasp)
    '''
    X_bin = []
    y_bin = []
    y = []

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                t.append(np_matrix[:, [(i+j)]])
            X_bin.append(t)
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                y_bin.append(1)
                y.append(obj_mapping(obj_ids[idx]))
            else: 
                y_bin.append(0)
                y.append(-1)

    data_set_bin = np.array(X_bin)
    y_label_bin = np.array(y_bin)
    y_label = np.array(y)    
    
    label_encoder_bin = LabelEncoder()
    label_encoder_bin.fit(y_label_bin)
    integer_encoded_bin = label_encoder_bin.transform(y_label_bin).reshape(len(y_label_bin), 1)
    onehot_encoder_bin = OneHotEncoder(sparse=False)
    onehot_encoder_bin.fit(integer_encoded_bin)
    label_bin = onehot_encoder_bin.transform(integer_encoded_bin)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)

    return data_set_bin, label_bin, label, y_label

def prepare_classification_dataset(dirpath,df,lookback, lookahead):
    X = []
    y = []

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                t.append(np_matrix[:, [(i+j)]])
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                X.append(t)
                y.append(obj_mapping(obj_ids[idx]))
               
    data_set = np.array(X)
    y_label = np.array(y)    
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)

    return data_set, label, y_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    parser.add_argument("-o", "--outdir", help="Output data directory", default='./data',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)

    args = parser.parse_args()
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_file = os.path.join(args.dataset,data_prefix+'.csv')

    print('\n')
    print('----------------')
    print('Building windows')
    print(args.dataset)
    print('----------------')
    print('\n')
    print('Loading data...')

    subdirectory = 'lookback_' + str(args.lookback) + '_' + 'lookahead_' + str(args.lookahead)
    if not os.path.exists(os.path.join(args.outdir,data_prefix,subdirectory)):
        os.makedirs(os.path.join(args.outdir,data_prefix,subdirectory))

    binned_samples_df = pd.read_csv(csv_file)
    hold = binned_samples_df['Hold']
    rew = binned_samples_df['Rew']
    img = binned_samples_df['img_id']
    obj = binned_samples_df['obj_id']

    print('Remove under-represented classes')
    obj_to_remove = binned_samples_df.groupby('obj_id').Start.count().loc[lambda p: p < 3]
    for obj_r in obj_to_remove.keys():
        print('Removed object: ' + str(obj_r))
        binned_samples_df = binned_samples_df[binned_samples_df.obj_id!=obj_r]
    
    train_val_df, test_df = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df.obj_id)    
    
    print('Binary train')
    X_train_bin, y_train_bin, y_train_full, obj_id_full = prepare_detection_dataset(args.dataset, train_df, args.lookback, args.lookahead)
    print('Binary validation')
    X_val_bin, y_val_bin, y_val_full, obj_id_val_full = prepare_detection_dataset(args.dataset, val_df, args.lookback, args.lookahead)
    print('Binary test')
    X_test_bin, y_test_bin, y_test_full, obj_id_test_full = prepare_detection_dataset(args.dataset, test_df, args.lookback, args.lookahead)

    print('Multiclass train')
    X_train_multi, y_train_multi, obj_id = prepare_classification_dataset(args.dataset, train_df, args.lookback, args.lookahead)
    print('Multiclass validation')
    X_val_multi, y_val_multi, obj_id_val = prepare_classification_dataset(args.dataset, val_df, args.lookback, args.lookahead)
    print('Multiclass test')
    X_test_multi, y_test_multi, obj_id_test = prepare_classification_dataset(args.dataset, test_df, args.lookback, args.lookahead)
    
    print('\n')
    print('Save binary dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_bin, y=y_train_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_valset.npz'), 'bw') as valfile:
        np.savez(valfile, X=X_val_bin, y=y_val_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_bin, y=y_test_bin)
            
    print('\n')
    print('Save multiclass dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_multi, y=y_train_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_trainset_objid.npz'), 'bw') as ref_obj_id:
        np.savez(ref_obj_id, X=X_train_multi, y=obj_id)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_valset.npz'), 'bw') as valfile:
        np.savez(valfile, X=X_val_multi, y=y_val_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_valset_objid.npz'), 'bw') as ref_obj_id:
        np.savez(ref_obj_id, X=X_val_multi, y=obj_id_val)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_multi, y=y_test_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_testset_objid.npz'), 'bw') as ref_obj_id:
        np.savez(ref_obj_id, X=X_test_multi, y=obj_id_test)

    print('\n')
    print(data_prefix+'Completed')

# Neural-decoding-paper-2023

To execute the pipeline, the original recordin session data (MRec40.neo.mat) must be placed in the current directory.

Use `run.sh` to execute all the steps at once.

### Step 1 - time discretization and trial-metadata association

This step extract the trials from the entire dataset, applies time discretisation and and associates each trial with metadata (experiment phases timestamp, object id).

    python time_binning_trials.py -d MRec40.neo.mat -o ./pre_processed --bin-size 40

### Step 2 - sequence creation

This step splits the trials in training, validation and test set, then create sequences for RNN training and associates labels for both movement onset detection and grasp classification. Lookback represents the length of the sequences extracted from trials.

    RNN_sequence_split_data.py -d pre_processed/MRec40_40_binned_spiketrains -o ./training_ready --lookback 12

### Step 3.1 - training of movement onset detector

This step trains a binary classification model for the movement onset detection.

    python BRNN_onset.py -d ./training_ready/MRec40_40_binned_spiketrains/lookback_12_lookahead_0/

### Step 3.2 -  trainibng of grasp classificator

This step trains a multi-class classification model for the grasped object detection.

    python BRNN_classification.py -d ./training_ready/MRec40_40_binned_spiketrains/lookback_12_lookahead_0/
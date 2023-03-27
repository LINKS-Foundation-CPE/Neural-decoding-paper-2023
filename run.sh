#!/bin/bash

# python time_binning_trials.py -d MRec40.neo.mat -o ./pre_processed --bin-size 40
# python RNN_sequence_split_data.py -d pre_processed/MRec40_40_binned_spiketrains -o ./training_ready --lookback 12

python BRNN_onset.py -d ./training_ready/MRec40_40_binned_spiketrains/lookback_12_lookahead_0/
python BRNN_classification.py -d ./training_ready/MRec40_40_binned_spiketrains/lookback_12_lookahead_0/
import os
import json
import pickle
import numpy as np
from copy import copy
import torch
from filelock import FileLock
#from torch.utils import data

import bci_experiments.constants as consts
from mlbackend.data.dataset import  EegDataset
from mlbackend.data.dataset import generate_partitions
from mlbackend.preprocessing.process import time_delay_embedding

filenames = {
    '11': 's11-gammasys-home-impaired.json',
    '13': 's13-gammasys-home-impaired.json',
    '15': 's15-gammasys-home-impaired.json',
    '16': 's16-gammasys-home-impaired.json',
    '20': 's20-gammasys-gifford-unimpaired.json',
    '21': 's21-gammasys-gifford-unimpaired.json',
    '22': 's22-gammasys-gifford-unimpaired.json',
    '23': 's23-gammasys-gifford-unimpaired.json',
    '24': 's24-gammasys-gifford-unimpaired.json',
    '25': 's25-gammasys-gifford-unimpaired.json',
    '26': 's26-gammasys-gifford-unimpaired.json',
    '27': 's27-gammasys-gifford-unimpaired.json',
    '28': 's28-gammasys-gifford-unimpaired.json',
}


# ----------------------------------------------
# FOR LETTER PROTOCOL ---> MIN WINDOW SIZE = 208
# ----------------------------------------------

# find min window size

def find_min_window_size():
    global_min_window_size = np.inf

    for id, file in filenames.items():
        data = json.load(open(os.path.join(consts.DATASET_LOCATION, consts.DATASET_CSU_GAMMA_SYS, file), 'r'))

        for experiment in data:
            protocol = experiment['protocol']
            target_letter = protocol[-1]

            # only collect data for letter protocol for the time being
            # I do not understand what grid protocol was
            if 'letter' in protocol:
                for key, trial in experiment['eeg'].items():
                    eeg = np.array(trial).T

                    # find beginning of each stimulus
                    starts = np.where(np.diff(np.abs(eeg[:, -1])) > 0)[0]

                    stimuli = [chr(int(n)) for n in np.abs(eeg[starts + 1, -1])]

                    # identify target segments
                    target_segments = np.array(stimuli) == target_letter

                    # find window sizes
                    min_window_size = np.min(np.diff(starts))
                    if global_min_window_size > min_window_size:
                        global_min_window_size = min_window_size

    return global_min_window_size


def collect_all_data_for_subject(file, window_size, preprocessing):

    X = None
    T = None
    data = json.load(open(file, 'r'))

    for experiment in data:
        protocol = experiment['protocol']
        target_letter = protocol[-1]

        # only collect data for letter protocol for the time being
        # I do not understand what grid protocol was
        if 'letter' in protocol:
            for key, trial in experiment['eeg'].items():
                eeg = np.array(trial).T

                # find beginning of each stimulus
                starts = np.where(np.diff(np.abs(eeg[:, -1])) > 0)[0]

                stimuli = [chr(int(n)) for n in np.abs(eeg[starts + 1, -1])]

                # identify target segments
                target_segments = np.array(stimuli) == target_letter

                # get starting indices for all the segments
                indices = np.array([np.arange(s, s + window_size) for s in starts])
                segments = eeg[indices, :8]

                segments = np.moveaxis(segments, 1, 2)
                if X is None:
                    X = segments
                    T = target_segments
                else:
                    X = np.vstack((X, segments))
                    T = np.hstack((T, target_segments))

    targets = np.array([1] * T.shape[0])
    non_target_idx = np.where(T == False)[0]
    targets[non_target_idx] = 0

    for preprocessing_method, args in preprocessing:
        args['data'] = X
        args['targets'] = targets
        X, targets = preprocessing_method(args)

    return torch.FloatTensor(copy(X)), torch.LongTensor(targets)


def get_data_loaders_for_subject(location, filename, args):
    with FileLock(os.path.join(location, 'data.lock')):
        file = os.path.join(location, filename)

        data, targets = collect_all_data_for_subject(file, args.segment_window_size, args.preprocessing)
        if args.expand_dims == True:
            data = np.expand_dims(data, axis=1)

        train_idx, valid_idx, test_idx = generate_partitions(data.shape[0], args.folds, args.fold_id)

        train_x = data[train_idx, :]
        train_labels = targets[train_idx]
        train_data = EegDataset(train_x, train_labels)

        valid_x = data[valid_idx, :]
        valid_labels = targets[valid_idx]
        valid_data = EegDataset(valid_x, valid_labels)

        test_x = data[test_idx, :]
        test_labels = targets[test_idx]
        test_data = EegDataset(test_x, test_labels)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        
        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

        return train_queue, valid_queue, test_queue


def create_partitions(subjects, dataset_location, segment_window_size, preprocessing, partition_file_name):
    for subject, filename in subjects.items():
        file = os.path.join(dataset_location, filename)
        _, targets = collect_all_data_for_subject(file, segment_window_size, preprocessing)
        classes = np.unique(targets)
        training_partition = [] 
        validation_partition = []
        testing_partition = []
        for target in classes:
            idxs = np.where(targets == target)[0]
            np.random.shuffle(idxs)
            train, valid, test = np.split(idxs, [int(.6 * idxs.shape[0]), int(.8 * idxs.shape[0])])
            training_partition.extend(list(train))
            validation_partition.extend(list(valid))
            testing_partition.extend(list(test))
        
        training_partition = np.array(training_partition)
        validation_partition = np.array(validation_partition)
        testing_partition = np.array(testing_partition)

        assert np.intersect1d(training_partition, validation_partition).shape[0] == 0, \
                                'common elements in training_partition and validation_partition'

        assert np.intersect1d(testing_partition, validation_partition).shape[0] == 0, \
                        'common elements in testing_partition and validation_partition'
        
        assert np.intersect1d(training_partition, testing_partition).shape[0] == 0, \
        'common elements in training_partition and testing_partition'

        partitions = [training_partition, validation_partition, testing_partition]
        partition_file_path = os.path.join(dataset_location, partition_file_name + '_%s.pickle'%str(subject))
        with open(os.path.join(dataset_location, partition_file_path), 'wb') as f:
            pickle.dump(partitions, f)
        

def print_dataset_stats(location, subjects, window_size):
    class_labels = {0:'Non Target', 1: 'Target'}
    for id, filename in subjects.items():
        file = os.path.join(location, filename)
        data, targets = collect_all_data_for_subject(file, window_size, {})
        class_string = ''
        labels, counts = np.unique(targets, return_counts=True)
        for l, c in zip(labels, counts):
            class_string+= '\t\t%s: %d'%(class_labels[l], c)
        np.unique(targets, return_counts=True)
        
        print('Subject :',id, class_string)

if __name__ == '__main__':
    dataset_location = os.path.join(consts.DATASET_LOCATION, consts.DATASET_CSU_GAMMA_SYS)
    subjects  = {
        '11': 's11-gammasys-home-impaired.json',
        '13': 's13-gammasys-home-impaired.json',
        '15': 's15-gammasys-home-impaired.json',
        '16': 's16-gammasys-home-impaired.json',
        '20': 's20-gammasys-gifford-unimpaired.json',
        '21': 's21-gammasys-gifford-unimpaired.json',
        '22': 's22-gammasys-gifford-unimpaired.json',
        '23': 's23-gammasys-gifford-unimpaired.json',
        '24': 's24-gammasys-gifford-unimpaired.json',
        '25': 's25-gammasys-gifford-unimpaired.json',
        '26': 's26-gammasys-gifford-unimpaired.json',
        '27': 's27-gammasys-gifford-unimpaired.json',
        '28': 's28-gammasys-gifford-unimpaired.json',
    }

    segment_window_size = 208
    preprocessing = []
    partition_file_name = 'raw_partition'


    print_dataset_stats(dataset_location, subjects, segment_window_size)
    #create_partitions(subjects, dataset_location, segment_window_size, preprocessing, partition_file_name)




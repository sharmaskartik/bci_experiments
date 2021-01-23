import torch
import pickle
import os
import torch.nn as nn

import bci_experiments.constants as consts
from bci_experiments.dataset_preprocessing.high_gamma_movement_dataset import \
            load_bbci_data

from arguments import get_args_for_experiment

from mlbackend.factory.model import SchirrmeisterDeepModel
from mlbackend.factory.optimizer  import AdamFactory
from mlbackend.factory.scheduler import LRSchedulerFactory

from mlbackend.experiments import Experiment
from mlbackend.data.dataset import EegDataset
from mlbackend.viz.plotter import PlotStatsForMultipleSubjects

from mlbackend.preprocessing.process import time_delay_embedding

def load_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return EegDataset(dataset.X, dataset.y)

def load_data_from_mat(dataset_location, file_name, args):
    #load datasets
    filepath = os.path.join(dataset_location, 'train', file_name)
    dataset = load_bbci_data(filepath, args.low_cut_hz, debug=False)
    X, Y = time_delay_embedding(dataset.X, dataset.y, args.time_delay_window_size)
    train_data  = EegDataset(X, Y)

    filepath = os.path.join(dataset_location, 'test', file_name)
    dataset = load_bbci_data(filepath, args.low_cut_hz, debug=False)
    X, Y = time_delay_embedding(dataset.X, dataset.y, args.time_delay_window_size)
    valid_data  = EegDataset(X, Y)


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    return train_queue, valid_queue


def load_data_from_pickle(dataset_location, file_name, args):
    #load datasets
    filepath = os.path.join(dataset_location, 'train', file_name)
    train_data  = load_data(filepath)


    filepath = os.path.join(dataset_location, 'test', file_name)
    valid_data  = load_data(filepath)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    return train_queue, valid_queue

def main():

    dataset_location = os.path.join(consts.DATASET_LOCATION, consts.DATASET_HIGH_GAMMA)
    args = get_args_for_experiment(results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__))), save='rep',
                                   multi_subject_experiment=True)
    # subjects = {3:'1.pickle', 2:'1.pickle', 1:'1.pickle'}
    # data_loader = load_data_from_pickle

    subjects = {}
    for i in range(1,15):
        subjects[i] = '%d.mat' % i
    data_loader = load_data_from_mat

    args.low_cut_hz = 0
    args.reps_per_subject = 5
    args.num_targets = 4
    args.resume = ''

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model_factory = SchirrmeisterDeepModel()
    optimizer_factory = AdamFactory()
    scheduler_factory = LRSchedulerFactory()

    exp = Experiment(dataset_location, subjects, data_loader,  model_factory, optimizer_factory, scheduler_factory, criterion, args)
    stat = exp.run()
    path_to_stats = os.path.join(args.resume, 'stat.pickle')
    plotter = PlotStatsForMultipleSubjects(path_to_stats, block=False, alpha=0.5)
    plotter.plot()


if __name__ == '__main__':
    main()

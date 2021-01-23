from matplotlib.pyplot import plot

import os
import torch
import torch.nn as nn
import numpy as np
import bci_experiments.constants as consts
from bci_experiments.dataset_preprocessing.csu_gamma_sys import \
            get_data_loaders_for_subject
import copy
from bci_experiments.experiments.csu_p300.arguments import get_args_for_hyperparams_experiment

from mlbackend.factory.model import CSUp300ModelFactory
from mlbackend.factory.model import EEGNetModelFactory
from mlbackend.factory.optimizer  import AdamFactory
from mlbackend.factory.scheduler import LRSchedulerFactory

from mlbackend.experiments import HyperparameterExperiment
from mlbackend.viz.plotter import PlotStatsForMultipleSubjects

from mlbackend.preprocessing.process import decimate, time_delay_embedding
from mlbackend.preprocessing.filter import butter_filter



def main():

    dataset_location = os.path.join(consts.DATASET_LOCATION, consts.DATASET_CSU_GAMMA_SYS)
    args = get_args_for_hyperparams_experiment(results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__))), save='rep',
                                   multi_subject_experiment=True)

    args.num_targets = 2
    args.resume = ''
    args.partition_file_name = 'raw_partition'
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

    
    subjects  = {
        '11': 's11-gammasys-home-impaired.json',
        '13': 's13-gammasys-home-impaired.json',
    }


    model_factories = []
    args.expand_dims = True
    model_args = {}
    model_args['C'] = 8
    model_args['F1'] = 8
    model_args['D'] = 2
    model_args['kernel_size'] = (1, 128)
    model_args['drop_p'] = 0.5
    model_factories.append([EEGNetModelFactory(), model_args])

    learning_rates = [10**np.random.uniform(-3, -7) for _ in range(1)]
    epochs = [6]

    weight_decays = [10**np.random.uniform(-5, 0) for _ in range(2)]
    optimizer_factories = []
    for weight in weight_decays:
        optim_args = {}
        optim_args['weight_decay'] = weight
        optimizer_factories.append([AdamFactory(), optim_args])


    decays = [1]
    gammas = [.99]
    shed_args ={}
    scheduler_factories = []
    for decay in decays:
        for gamma in gammas:
            shed_args['decay'] = decay
            shed_args['gamma'] = gamma
            scheduler_factories.append([LRSchedulerFactory(), shed_args])

    time_delay_window_sizes = [0]

    i =0
    hyperparams = {}
    for model in model_factories:
        for optimizer in optimizer_factories:
            for scheduler in scheduler_factories:
                for learning_rate in learning_rates:
                    for epoch in epochs:
                        for window_size in time_delay_window_sizes:
                            params = {}
                            params['model'] = model
                            params['optimizer'] = optimizer
                            params['scheduler'] = scheduler
                            params['learning_rate'] = learning_rate
                            params['epochs'] = epoch
                            params['time_delay_window_size'] = window_size

                            hyperparams[i] = params 
                            i += 1





    data_loader = get_data_loaders_for_subject

    ##################################
    # DATASET PREPROCESSING ARGUMENTS
    ##################################

    preprocessing = []
    
    # decimate_args = {}
    # decimate_args['downsample_factor'] = 2
    # preprocessing.append([decimate, decimate_args])

    # time_delay_embedding_args = {}
    # time_delay_embedding_args['window_size'] = window_size
    # time_delay_embedding_args['jump'] = time_delay_embedding_jump
    # preprocessing.append([time_delay_embedding, time_delay_embedding_args])

    # butter_args = {}
    # butter_args['freqs'] = [45]
    # butter_args['fs'] = 256
    # butter_args['order'] = 9
    # butter_args['btype']  = 'lp'
    # preprocessing.append([butter_filter, butter_args])
    args.preprocessing = preprocessing





    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    from mlbackend.util.util import bsr
    metric = bsr
    exp = HyperparameterExperiment(dataset_location, subjects, data_loader, criterion, metric, hyperparams, args)
    exp.run()



if __name__ == '__main__':
    main()


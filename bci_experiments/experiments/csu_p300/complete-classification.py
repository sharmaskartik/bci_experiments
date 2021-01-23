from matplotlib.pyplot import plot
import torch
import pickle
import os
import torch.nn as nn
import ray

import bci_experiments.constants as consts
from bci_experiments.dataset_preprocessing.csu_gamma_sys import \
            get_data_loaders_for_subject

from arguments import get_args_for_experiment

from mlbackend.factory.model import CSUp300ModelFactory
from mlbackend.factory.model import EEGNetModelFactory
from mlbackend.factory.optimizer  import AdamFactory
from mlbackend.factory.scheduler import LRSchedulerFactory

from mlbackend.experiments import Experiment
from mlbackend.viz.plotter import PlotStatsForMultipleSubjects

from mlbackend.preprocessing.process import decimate, time_delay_embedding
from mlbackend.preprocessing.filter import butter_filter

from mlbackend.experiments import Experiment

def main():

    dataset_location = os.path.join(consts.DATASET_LOCATION, consts.DATASET_CSU_GAMMA_SYS)
    args = get_args_for_experiment(results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__))), save='rep',
                                   multi_subject_experiment=True)
    # subjects = {3:'1.pickle', 2:'1.pickle', 1:'1.pickle'}
    # data_loader = load_data_from_pickle

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
        '27': 's27-gammasys-gifford-unimpaired.json',
        '28': 's28-gammasys-gifford-unimpaired.json',
    }
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


    args.reps_per_subject = 5
    args.num_targets = 2
    args.resume = '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/train-multi-subject-2020-11-17--16-14-32'

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # model_args = {}
    # model_args['init_channels'] = args.init_channels
    # model_args['num_targets'] = args.num_targets
    # model_args['dropout_prob'] = args.dropout_prob
    # setattr(args, 'model_args', model_args)
    # args.expand_dims = False
    # model_factory = CSUp300ModelFactory()

    model_args = {}
    model_args['C'] = 8
    model_args['F1'] = 8
    model_args['D'] = 2
    model_args['kernel_size'] = (1, 128)
    model_args['drop_p'] = 0.5
    args.model_args = model_args
    model_factory = EEGNetModelFactory()
    args.expand_dims = True

    
    optimizer_factory = AdamFactory()
    scheduler_factory = LRSchedulerFactory()

    from mlbackend.util.util import bsr
    metric = bsr
    #exp = Experiment(dataset_location, subjects, data_loader,  model_factory, optimizer_factory, scheduler_factory, criterion, metric, args)
    #stat = exp.run()
    # path_to_stats = os.path.join(args.resume, 'stat.pickle')
    # plotter = PlotStatsForMultipleSubjects(path_to_stats, block=False, alpha=0.5)
    # plotter.plot()
    print('hello')

if __name__ == '__main__':
    main()
    # plotter = PlotStatsForMultipleSubjects(
    #     '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/temp/stat.pickle'
    #     , block=False, show=False, alpha=0.5)
    # plotter.plot()




    #code to check plotting for multiple experiment

    # list_of_files = ['/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/train-multi-subject-2020-11-12--06-40-48/stat.pickle',
    # '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/train-multi-subject-2020-11-12--08-13-10/stat.pickle',
    # '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/train-multi-subject-2020-11-12--08-13-10/stat.pickle']
    # from mlbackend.viz.plotter import PlotStatsForMultipleExperiments
    # colors = ['r', 'g', 'b']
    # save = '/s/chopin/l/grad/kartikay/code/machine_learning/workspace/bci/bci_experiments/bci_experiments/experiments/csu-p300/logs/comparison'
    # plotter = PlotStatsForMultipleExperiments(list_of_files, save, colors = colors, alpha=0.3)
    # plotter.plot()
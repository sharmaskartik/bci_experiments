import torch
import pickle
import os
import torch.nn as nn

import bci_experiments.constants as consts
from bci_experiments.dataset_preprocessing.high_gamma_movement_dataset import \
            load_bbci_data

from mlbackend.model_zoo.schirrmeister import DeepModel
from arguments import get_args_for_experiment
from mlbackend.experiments import SingleSubjectExperiment
from mlbackend.data.dataset import EegDataset

def load_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return EegDataset(dataset.X, dataset.y)

def main():

    dataset_location = os.path.join(consts.DATASET_LOCATION, consts.DATASET_HIGH_GAMMA)

    subject_id = 1
    low_cut_hz = 0

    # #load datasets
    # filename = os.path.join(dataset_location, 'train', '%d.mat'%subject_id)
    # dataset = load_bbci_data(filename, low_cut_hz, debug=False)
    #
    # filename = os.path.join(dataset_location, 'test', '%d.mat'%subject_id)
    # dataset = load_bbci_data(filename, low_cut_hz, debug=False)
    # valid_data  = EegDataset(dataset.X, dataset.y)
    args = get_args_for_experiment(results_dir = os.path.dirname(os.path.abspath(__file__)), save = 'subject_{%d}'%subject_id)
    import pdb; pdb.set_trace()

    valid_data  = load_data('test.pickle')
    train_data  = load_data('train.pickle')

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    x, l = next(iter(train_queue))
    model = DeepModel(x.shape[2], 25, 4, args.dropout_prob)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    exp = SingleSubjectExperiment(model, train_queue, valid_queue, optimizer, scheduler, criterion, args)
    exp.run()

if __name__ == '__main__':
    main()

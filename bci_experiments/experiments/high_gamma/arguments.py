import argparse
import os


def get_args_for_experiment(results_dir, save, multi_subject_experiment = False):
    parser = argparse.ArgumentParser("Darts")
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--save', type=str, default=save, help='experiment name')
    parser.add_argument('--results_dir', type = str, default = os.path.join(results_dir, 'logs')
                                , help = 'directory where the results are stored on disk')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--time_delay_window_size', type=int, default=500, help='size of the window in time delay embedding')
    parser.add_argument('--multi_subject_experiment', type=bool, default=multi_subject_experiment, help='Is the experiment\
         ran over multiple subject or single subject. This changes how the results directory is created')

    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--data', type=str, default='../data/', help='location of the data corpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

    parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')

    parser.add_argument('--gamma', type=float, default=0.99, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')

    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')

    parser.add_argument('--init_channels', type=int, default=25, help='num of init channels')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')

    parser.add_argument('--dropout_prob', type=float, default=0.5, help='drop out probability')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--n_models_to_save', type=int, default=10, help='Number of top models to save based on AUC')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loaders')

    args = parser.parse_args()
    return args
    import pdb; pdb.set_trace()

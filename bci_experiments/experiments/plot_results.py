import argparse
import os
import pickle

from magic.data.stats import MultiTrainingStats
from magic.viz.plotter import PlotStatsForMultipleSubjects
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("plots")
parser.add_argument('--path', required = True, type=str, metavar='PATH', help='path to stats.pickle')
args = parser.parse_args()

# with open(os.path.join(args.path, 'stat.pickle'), 'rb') as f:
#     ser_stat = pickle.load(f)
#
plt.ion()
#
# stat = MultiTrainingStats(ser_stat)
#import pdb; pdb.set_trace()
plotter = PlotStatsForMultipleSubjects(args.path, block=False, alpha=0.5)
plotter.plot()

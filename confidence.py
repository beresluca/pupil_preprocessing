"""
Pupil preprocessing step 2
Getting rid of samples with low confidence levels
- with cutoff point as a parameter

Usage:
python3 confidence.py --input_dir "INPUTDIR" --cutoff "CUTOFF"
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from filter_3d import filter_3d
import matplotlib
from matplotlib import pyplot as plt

# input_dir = "/home/lucab/pupildata_CommGame/pupil_R/pair16_Gondor_freeConv/"
# cutoff = 0.80

def conf(input_dir, cutoff):
    tmp = filter_3d(input_dir)
    pldata = pd.DataFrame(tmp)
    time = pldata.timestamp - pldata.timestamp[0]

    matplotlib.pyplot.scatter(time, pldata.diameter, c="lightgreen", s=1, label="raw")
    #plt.show()

    # insert NaN to diameter and position columns where confidence is low
    pldata.diameter = pldata.diameter.where(pldata.confidence >= cutoff)
    pldata.norm_pos_x = pldata.norm_pos_x.where(pldata.confidence >= cutoff)
    pldata.norm_pos_y = pldata.norm_pos_y.where(pldata.confidence >= cutoff)

    print(pldata[0:16])
    # check how much data was replaced with NaN
    n_nan = sum(np.isnan(x) for x in pldata.diameter)
    print("Missing {}% of the data after confidence thresholding." .format(round((n_nan/len(pldata)) * 100)))

    return pldata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='path to the raw pupil labs recording dir')
    parser.add_argument('cutoff', help='confidence cutoff point (float), samples with EQUAL or HIGHER '
                                       'confidence levels will be kept only', type=float)
    args = parser.parse_args()

    # check if input directory is valid
    if not os.path.isdir(args.input_dir):
        print('Invalid input dir: {}'.format(args.input_dir))
        sys.exit()
    else:
        # run function
        conf(args.input_dir, args.cutoff)



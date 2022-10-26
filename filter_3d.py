"""
Pupil data preprocessing script to load and filter 3D data from the pupil_data.tsv created by pl_preprocessing.py
Ideally, first step in the preprocessing pipeline.

Usage:
python3 filter_3d.py --input_dir "INPUTDIR"

input_dir: str, path to session-level data directory containing pupil_data.tsv
"""
import sys
import os
import argparse
from os.path import join
import pandas as pd

#input_dir = "/home/lucab/pupildata_CommGame/pupil_R/pair16_Mordor_freeConv/"
sys.path.append("/home/lucab/PycharmProjects/pupil_preprocessing")

def filter_3d(input_dir):

    tsv_file = join(input_dir, "pupil_data.tsv")
    pl_tsv = pd.read_csv(tsv_file, sep='\t', skiprows=lambda x: x % 2)
    #print(pl_tsv)
    return pl_tsv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='path to the raw pupil labs recording dir')
    args = parser.parse_args()

    # check if input directory is valid
    if not os.path.isdir(args.input_dir):
        print('Invalid input dir: {}'.format(args.input_dir))
        sys.exit()
    else:
        # run script
        filter_3d(args.input_dir)




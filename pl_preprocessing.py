"""
Updated version of the pl_preprocessing.py script from https://github.com/dcnieho/GlassesTestCodeData
(under /preprocessing/1_to_common_format/pupillabs/pl_preprocessing.py)

Updates were necessary due to changed pupil-lab data formats. Current version also extracts pupil diameter and
transforms timestamps into unix time for synchronization with further devices.

Current version tested under python 3.8.

The module contains helper functions for extracting timestamps and gaze information from pupil-lab's own data formats
to easy-to-handle csv files + a __main__ running these functions on specified data set.

Usage as script:

python3 pl_preprocessing.py --input_dir "INPUTDIR" --output_dir "OUTPUTDIR" --pid "PID"

Parameters:
input_dir:      str, path to data dir (subject directory created by pupil)
output_dir:     str, path for output dir
pid:            str, ID for data set (participant / session / etc)

Returns:
None, preprocessed data is saved to output_root
Output_root will contain:
    - frame_timestamps.tsv: table of timestamps for each frame in the world
    - gaze_data_world.tsv: gaze data, where all gaze coordinates are represented w/r/t the world camera
    - pupil_data.tsv: pupil data, with all pupil diameter and raw position coordinates, from all frames irrespective
    of gaze detection
"""


import sys
import os
# import shutil
import argparse
from datetime import datetime
from os.path import join
import numpy as np
import pandas as pd
import csv
import json
from itertools import chain
# module from pupil (.../pupil_src/shared_modules/file_methods.py)
sys.path.append('/media/lucab/data_hdd/lucab/PycharmProjects/pupil_preprocessing/preprocessing_code/1_to_common_format/pupillabs/')
import file_methods as fm



def preprocess_data(input_dir, output_root, pid):
    """
    Run all preprocessing steps for pupil lab data

    Parameters:
    input_dir:       str, path to data dir (subject directory created by pupil)
    output_root:     str, path for output dir
    pid:             str, ID for data set (participant / session / etc)

    Returns:
    None, preprocessed data is saved to output_root
    """

    # Read info.player.json containing basic info about recording / data set
    info_file = join(input_dir, 'info.player.json')
    with open(info_file) as f:
        info_dict = json.load(f)  # output is dictionary

    # extract useful info from info dict
    pupil_starttime = info_dict['start_time_synced_s']  # start time in seconds according to pupil's time epoch
    unix_starttime = info_dict['start_time_system_s']  # start time in seconds in unix time
    recording_date = datetime.fromtimestamp(unix_starttime)

    # user feedback
    print('From info.player.json: \n',
          'Recording name: ', info_dict['recording_name'], '\n',
          'Recording started at: ', recording_date.strftime('%Y.%m.%d %H:%M:%S:%f'), '\n',
          'Recording start in pupil\'s own epoch: ', str(pupil_starttime)
          )

    # create the output directory (if necessary)
    output_dir = join(output_root, pid)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('\nOutput dir is', output_dir)

    # Format the gaze data
    print('\nFormatting gaze data...')
    gaze_data_world, frame_timestamps = format_gaze_data(input_dir)

    # write the gaze_data to a tsv file
    print('\nWriting gaze data to tsv file...')
    csv_file = join(output_dir, 'gaze_data_world.tsv')
    export_range = slice(0, len(gaze_data_world))
    with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
        csv_writer.writerow(['{}\t{}\t{}\t{}\t{}'.format('timestamp',
                                                         'frame_idx',
                                                         'confidence',
                                                         'norm_pos_x',
                                                         'norm_pos_y')])
        for g in list(chain(*gaze_data_world[export_range])):
            data = ['{:.6f}\t{:d}\t{:.2f}\t{:.3f}\t{:.3f}'.format(g['timestamp']-pupil_starttime+unix_starttime,
                                                                  g['frame_idx'],
                                                                  g['confidence'],
                                                                  g['norm_pos'][0],
                                                                  1-g['norm_pos'][1])]  # translate y coord to origin in top-left
            csv_writer.writerow(data)

    # Read in pupil data
    pupil_data = read_pupil_data(input_dir)
    # write pupil data to a tsv file
    print('\nWriting pupil data to tsv file...')
    csv_file = join(output_dir, 'pupil_data.tsv')
    export_range = slice(0, len(pupil_data))
    with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
        csv_writer.writerow(['{}\t{}\t{}\t{}\t{}\t{}'.format('timestamp',
                                                         'diameter',
                                                         'confidence',
                                                         'method',
                                                         'norm_pos_x',
                                                         'norm_pos_y')])
        for g in pupil_data:
            if g['method'] == 'pye3d 0.1.1 real-time':
                data = ['{:.6f}\t{:.3f}\t{:.2}\t{}\t{:.3f}\t{:.3f}'.format(g['timestamp']-pupil_starttime+unix_starttime,
                                                                           g['diameter_3d'],
                                                                           g['confidence'],
                                                                           g['method'],
                                                                           g['norm_pos'][0],
                                                                           1-g['norm_pos'][1])]  # translate y coord to origin in top-left
            else:
                data = ['{:.6f}\t{:.3f}\t{:.2}\t{}\t{:.3f}\t{:.3f}'.format(g['timestamp']-pupil_starttime+unix_starttime,
                                                                           g['diameter'],
                                                                           g['confidence'],
                                                                           g['method'],
                                                                           g['norm_pos'][0],
                                                                           1-g['norm_pos'][1])]  # translate y coord to origin in top-left
            csv_writer.writerow(data)



    # write the world camera frame timestamps to a tsv file
    frameNum = np.arange(1, frame_timestamps.shape[0]+1)
    frame_ts_df = pd.DataFrame({'frameNum': frameNum, 'timestamp': frame_timestamps})
    frame_ts_df.to_csv(join(output_dir, 'frame_timestamps.tsv'), sep='\t', float_format='%.4f', index=False)

    # # Compress and Move the world camera movie to the output
    # print('\nCopying world recording movie...')
    # if not 'worldCamera.mp4' in os.listdir(output_dir):
    #     # compress
    #     print('compressing world camera video')
    #     cmd_str = ' '.join(['ffmpeg', '-i', join(input_dir, 'world.mp4'), '-pix_fmt', 'yuv420p', join(input_dir, 'worldCamera.mp4')])
    #     os.system(cmd_str)
    #
    #     # move the file to the output directory
    #     shutil.move(join(input_dir, 'worldCamera.mp4'), join(output_dir, 'worldCamera.mp4'))
    return


def format_gaze_data(input_dir):
    """
    Function to
    (1) Load gaze data and corresponding timestamps ('gaze.pldata' and 'gaze_timestamps.npy')
    (2) Get the gaze location w/r/t world camera (normalized locations)
    (3) Sync gaze data with the world_timestamps array (see correlate_data function for details)

    Parameters:
    input_dir:       Str, path to data dir (subject directory created by pupil)

    Returns:
    gaze_by_frame:   List of lists, where each list contains gaze frame dictionary(ies) corresponding to
                     world frame timestamps. That is, gaze_by_frame[idx] is a list containing all gaze frames
                     (as dicts) corresponding to (happening before / around) world camera frame [idx].
                     Each gaze frame dictionary is derived from the pldata data serialized dictionaries and have keys
                     'topic', 'norm_pos', 'confidence', 'timestamp' (in pupil's own time) and 'frame_idx'.
                     The last key, 'frame_idx' corresponds to the appropriate world camera frame index.
    frame_timestamps:    Numpy array, holds the capture times for world camera frames in ms,
                         relative to the first gaze frame data.  Note that both the epoch start and the unit (ms)
                         are different than other timestamps in the pupil output!
    """

    # load gaze location data
    gaze_data = fm.load_pldata_file(input_dir, 'gaze')

    # After unpacking, pldata files contain data in a collections.deque of 'serialized dict'
    # types - all in all, similar to a memory mapped list of dictionaries (?).
    # Thing is, we can transform it into a list of default dictionaries, one per frame.
    # Note however that we access a protected member of the class here, not good practice in general(?)
    gaze_data = [gaze_data.data[i]._deep_copy_dict() for i in range(len(gaze_data.data))]
    # get rid of the 'base_data' part we don't need (it contains the raw info from which gaze is derived)
    for i in range(len(gaze_data)):
        gaze_data[i].pop('base_data', None)

    # load world camera frame timestamps
    timestamps_path = join(input_dir, 'world_timestamps.npy')
    frame_timestamps = np.load(timestamps_path)

    # align gaze with world camera timestamps
    gaze_by_frame = correlate_data(gaze_data, frame_timestamps)

    # make frame_timestamps relative to the first data timestamp
    start_timeStamp = gaze_by_frame[0][0]['timestamp']
    frame_timestamps = (frame_timestamps - start_timeStamp) * 1000  # convert to ms

    return gaze_by_frame, frame_timestamps


def read_pupil_data(input_dir):
    """
    Reads pupil data frames.

    Parameters:
    input_dir:       Str, path to data dir (subject/session directory created by pupil)

    Returns:
    pupil_data:  List of dicts, with each dictionary containing pupil and eye model parameters.
    """

    # load gaze location data
    pupil_data = fm.load_pldata_file(input_dir, 'pupil')

    # After unpacking, pldata files contain data in a collections.deque of 'serialized dict'
    # types - all in all, similar to a memory mapped list of dictionaries (?).
    # Thing is, we can transform it into a list of default dictionaries, one per frame.
    # Note however that we access a protected member of the class here, not good practice in general(?)
    pupil_data = [pupil_data.data[i]._deep_copy_dict() for i in range(len(pupil_data.data))]

    return pupil_data


def correlate_data(data, timestamps):
    """
    Sorts gaze data frames to world camera frames. Takes a data list and a timestamps list and makes a new list
    with the length of the number of timestamps. Each element of the output a list that will have 0, 1 or more
    associated data points. Finally we add an index field to the datum with the associated index

    Parameters:
    data:            List of dictionaries where each dict has at least key 'timestamp'
                     with double-precision float as value
    timestamps:      Timestamps list to correlate (sort) data to

    Returns:
    data_by_frame:   List of lists, with each list containing dictionaries from parameter 'data',
                     with the extra field 'frame_idx' added to it.
    """

    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    data.sort(key=lambda d: d['timestamp'])

    while True:
        try:
            datum = data[data_index]
            # we can take the midpoint between two frames in time: More appropriate for SW timestamps
            ts = (timestamps[frame_idx] + timestamps[frame_idx+1]) / 2.
            # or the time of the next frame: More appropriate for Sart Of Exposure Timestamps (HW timestamps).
            # ts = timestamps[frame_idx+1]
        except IndexError:
            # we might loose a data point at the end but we dont care
            break

        if datum['timestamp'] <= ts:
            datum['frame_idx'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index += 1
        else:
            if frame_idx == 0:
                datum['frame_idx'] = frame_idx
                data_by_frame[frame_idx].append(datum)
            frame_idx += 1

    return data_by_frame


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='path to the raw pupil labs recording dir')
    parser.add_argument('output_dir', help='output directory root. Raw data will be written to recording specific dirs within this directory')
    parser.add_argument('pid', help='participant ID')
    args = parser.parse_args()

    # check if input directory is valid
    if not os.path.isdir(args.input_dir):
        print('Invalid input dir: {}'.format(args.input_dir))
        sys.exit()
    else:

        # run preprocessing on this data
        preprocess_data(args.input_dir, args.output_dir, args.pid)

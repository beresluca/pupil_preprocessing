"""
Pupil preprocessing step 3
objectives:
- plot data points to detect the remaining outliers
- plot confidence levels, check sampling frequency to detect anomalies

"""
import math
import sys
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib
import scipy
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from scipy.interpolate import interp1d
from scipy.stats import zscore
from confidence import conf
from scipy import signal
from scipy.signal import medfilt, butter, lfilter, detrend, filtfilt
from scipy import fftpack

input_dir = "/media/lucab/data_hdd/lucab/pupildata_CommGame/pupil_R/pair25_Gondor_BG1/"
cutconf_at = 0.80
pl_data = conf(input_dir, cutconf_at)

def missing_data(data):
    count_nan = sum(np.isnan(x) for x in data)
    print("Missing {}% of the data at the moment." .format(round((count_nan/len(data)) * 100)))

# convert timestamps to readable format
time = pl_data.timestamp - pl_data.timestamp[0]

mean_plsize = np.nanmean(pl_data.diameter)
median_plsize = np.nanmedian(pl_data.diameter)
extr_plsize = [np.nanmin(pl_data.diameter), np.nanmax(pl_data.diameter)]
sd_plsize = np.nanstd(pl_data.diameter)
print(mean_plsize, median_plsize, sd_plsize)
print(extr_plsize)


# filter data by deviation from the median
lower_threshold = median_plsize - 3 * sd_plsize
higher_threshold = median_plsize + 3 * sd_plsize
print(lower_threshold, higher_threshold)

pl_data.diameter = pl_data.diameter.where(pl_data.diameter <= higher_threshold)
pl_data_thresh = pl_data.diameter.where(pl_data.diameter >= lower_threshold)
matplotlib.pyplot.plot(time,
                       pl_data_thresh, c="grey", label="SD thresholded")
# windowed way of SD filtering? big windows (5-10 s) átfedés nélkül először, van-e nagy trend?

newdata = np.array(pl_data_thresh)
plt.grid(True)
plt.legend(loc="upper left")
#plt.show()

# dilation speed outliers ???
def dialation_speed(data, thres_val=2):
    diff_dia = abs(np.diff(data, append=np.nan))
    matplotlib.pyplot.plot(time, diff_dia, c="blue", label="difference distribution")
    # thres_val is the difference value between two successive samples that the data should not exceed
    # a value of 2 seems to remove most of the spikes, smaller than that might remove valid data (?)
    data[(abs(np.diff(data, append=np.nan)) > thres_val)] = np.nan
    matplotlib.pyplot.plot(time, data, c="red", label="DS outliers removed")
    #plt.show()


dialation_speed(newdata, thres_val=2)
# let's see how much data we lost so far
missing_data(newdata)

# the smaller the min_diff value, the shorter the sections of missing data we interpolate
def interpolate(data, min_diff=3, max_samples=1000, method="lin", padding=1, n_samples=4):
    ind = np.array(np.where(np.isnan(data))).flatten()  # find nan-s in the data aka missing values
    # splits data according to the difference of nan indexes (min_diff), we use this to create sections to interpolate over
    missing_ind = np.split(ind, np.where(np.diff(ind) > min_diff)[0]+1)
    data_noEB = np.copy(data)  # copy data to interpolate over
    # for i in blink_ind_to_remove:
    for i in missing_ind:
        if i.size == 0:
            continue
        if i.size > max_samples:
            continue
        # create a vector of data and sample numbers before and after the blink
        befores = np.arange((i[0] - (n_samples + padding)), (i[0] - padding))
        afters = np.arange(i[-1] + (1 + padding), i[-1] + (1 + n_samples + padding))
        # this if statement is a contingency for when the blinks occur at the end of the dataset. it deletes the blink rather than interpolating
        if any(afters > len(data) - 1):
            data_noEB = data_noEB[0:i[0] - 1]
        else:
        # this is the actual interpolation part. you create your model dataset to interpolate over
            x = np.append(befores, afters)
            y = np.append(data[befores], data[afters])
        # then interpolate it
        if method == "lin":
            li = interp1d(x, y)  # scipy docs says nans present in input values results in undefined behavior..
            # create indices for the interpolated data, so you can return it to the right segment of the data
            xs = range(i[0] - padding, i[-1] + (1 + padding))
            np.put(data_noEB, xs, li(xs))
        if method == "cubic":
            cubic = interp1d(x, y, kind='cubic')
            # create indices for the interpolated data, so you can return it to the right segment of the data
            xs = range(i[0] - padding, i[-1] + (1 + padding))
            np.put(data_noEB, xs, cubic(xs))

    return data_noEB


data_noEB = interpolate(newdata, method="lin")
matplotlib.pyplot.plot(time, data_noEB, c="pink", label="interpolated")
plt.grid(True)
plt.legend(loc="upper left")
plt.show()

missing_data(data_noEB)

median_filtered_pldata = scipy.signal.medfilt(data_noEB, 5)
matplotlib.pyplot.plot(time, median_filtered_pldata, c="black", label="median filtered")
#plt.legend(loc="upper left")
#plt.show()


# lowpass butterworth filtering function
def butter_lowpass(cutoff, fs, order=5):
    # get nyquist frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # this is another filtering function from scipy
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=10, fs=120, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    # and another filtering function from scipy
    data_lp = lfilter(b, a, data)  # this results in a phase-shifted time series
    data_lp_zerophase = filtfilt(b, a, data)
    return data_lp_zerophase


# why does it return NaNs? should we interpolate blinks first?
pl_dia_low_pass = butter_lowpass_filter(median_filtered_pldata, cutoff=10, order=5)
matplotlib.pyplot.plot(time, pl_dia_low_pass, c="orange", label="10 Hz low pass")
plt.legend(loc="lower left")
plt.grid()
plt.show()

# pl_dia_low_pass_dt = detrend(pl_dia_low_pass)
# matplotlib.pyplot.plot(time,
#                        pl_dia_low_pass_dt, c="black")
# plt.show()


plt.hist(pl_data.diameter, bins=50)
plt.show()

# getting sampling frequency (difference vector of timestamps)
# diff = []
# for i in range(len(pl_data.timestamp)-1):
#    diff.append(pl_data.timestamp[i + 1] - pl_data.timestamp[i])


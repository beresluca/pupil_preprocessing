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
import numpy as np
import pandas as pd
import matplotlib
import scipy
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import zscore
from confidence import conf
from scipy import signal
from scipy.signal import medfilt, butter, lfilter, detrend
from scipy import fftpack

input_dir = "/media/lucab/data_hdd/lucab/pupildata_CommGame/pupil_R/pair25_Mordor_freeConv/"
cutconf_at = 0.70
pl_data = conf(input_dir, cutconf_at)

mean_plsize = np.nanmean(pl_data.diameter)
median_plsize = np.nanmedian(pl_data.diameter)
extr_plsize = [np.nanmin(pl_data.diameter), np.nanmax(pl_data.diameter)]
sd_plsize = np.nanstd(pl_data.diameter)

time = pl_data.timestamp - pl_data.timestamp[0]

# filter data by deviation from the median
lower_threshold = median_plsize - 3 * sd_plsize
higher_threshold = median_plsize + 3 * sd_plsize
# print(lower_threshold, higher_threshold)

pl_data.diameter = pl_data.diameter.where(pl_data.diameter <= higher_threshold)
pl_data_thresh = pl_data.diameter.where(pl_data.diameter >= lower_threshold)
matplotlib.pyplot.plot(time,
                       pl_data_thresh, c="grey", label="SD thresholded")
# windowed way of SD filtering? big windows (5-10 s) átfedés nélkül először, van-e nagy trend?

newdata = np.array(pl_data_thresh)

# plt.show()

# dilation speed outliers ???
dial_speed_thres = 10
diff_dia = np.diff(newdata)
matplotlib.pyplot.plot(time[0:-1], diff_dia, c="blue", label="difference distribution")
#plt.show()




# interpolating over bad data
# accepts a data matrix, "val" is the zscore threshold the function uses to find bad data
# n_samples is the number of samples on either side of the noise that the function will use to interpolate
# padding is the number of samples interpolated on either side of the noise to avoid interpolation spikes
def lin_interpolate(data, threshold='nan', val=-2, padding=5, n_samples=10):
    if threshold == 'zscore':
        if val < 0:
            ind = np.array(np.where(zscore(data) <= val)).flatten()  # find samples recorded during blinks (2 SDs below mean, unless val is different)
        elif val > 0:
            ind = np.array(np.where(zscore(data) >= val)).flatten()  # find samples recorded during blinks (2 SDs above mean, unless val is different)
        blink_ind = np.split(ind, np.where(np.diff(ind) > 15)[0] + 1)  # split indexed samples into groups of blinks
        data_noEB = np.copy(data)  # copy data to interpolate over
    elif threshold == 'nan':
        ind = np.array(np.where(np.isnan(data))).flatten()  # find samples recorded during blinks (data is NaN) (flatten kell-e?)
        print(ind)
        blink_ind = np.split(ind, np.where(np.diff(ind) > 15)[0] + 1)  # split indexed samples into groups of blinks
        #print(blink_ind)
        # delete blinks that are longer than 60 samples (~ 500 ms) ??
        diff_ts = [(blink_ind[i][-1] - blink_ind[i][0]) for i in range(len(blink_ind))]
        #blink_ind_to_remove = [i for i in range(len(blink_ind)) if diff_ts[i] > 60]
        blink_ind_to_remove = []
        for i in range(len(blink_ind)):
            if diff_ts[i] > 60:
                blink_ind_to_remove.append(blink_ind[i])
        #print(blink_ind_to_remove)
        data_noEB = np.copy(data)  # copy data to interpolate over
    # loop through each group of blinks
    # for i in blink_ind_to_remove:
        for blinks in blink_ind:
            if blinks.size == 0:
                continue
            # if (blink_ind_to_remove[i]) in blink_ind:
            #     print('found 1')
            #     continue
            # create a vector of data and sample numbers before and after the blink
            befores = np.arange((blinks[0] - (n_samples + padding)), (blinks[0] - padding))
            afters = np.arange(blinks[-1] + (1 + padding), blinks[-1] + (1 + n_samples + padding))
            # this if statement is a contingency for when the blinks occur at the end of the dataset. it deletes the blink rather than interpolating
            if any(afters > len(data) - 1):
                data_noEB = data_noEB[0:blinks[0] - 1]
            else:
                # this is the actual interpolation part. you create your model dataset to interpolate over
                x = np.append(befores, afters)
                y = np.append(data[befores], data[afters])
                # then interpolate it
                li = interp1d(x, y)
                # would cubic spline work?
                cubic = scipy.interpolate.CubicSpline(x, y, bc_type="clamped")
                # looks strange, spikes appear (?)

                # create indices for the interpolated data, so you can return it to the right segment of the data
                xs = range(blinks[0] - padding, blinks[-1] + (1 + padding))
                # I'm actually not sure that you need these two variables anymore, but they're still in here for some reason.
                #x_stitch = np.concatenate((x[0:n_samples], xs, x[n_samples:]))
                #y_stitch = np.concatenate((y[0:n_samples], li(xs), y[n_samples:]))
                # put the interpolated vector into the data
                #np.put(data_noEB, xs, li(xs))
                np.put(data_noEB, xs, cubic(xs))
        return data_noEB


data_noEB = lin_interpolate(newdata, threshold='nan', val=-2, padding=5, n_samples=10)
matplotlib.pyplot.plot(time, data_noEB, c="pink", label="interpolated")
plt.grid(True)
plt.legend(loc="upper left")
plt.show()

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
    data_lp10 = lfilter(b, a, data)
    return data_lp10


# why does it return NaNs? should we interpolate blinks first?
pl_dia_low_pass = butter_lowpass_filter(median_filtered_pldata, cutoff=10, order=5)
matplotlib.pyplot.plot(time, pl_dia_low_pass, c="orange", label="10 Hz low pass")

pl_dia_low_pass_20 = butter_lowpass_filter(median_filtered_pldata, cutoff=20, order=5)
matplotlib.pyplot.plot(time, pl_dia_low_pass, c="lightblue", label="20 Hz low pass")
plt.legend(loc="lower left")
plt.grid()
plt.show()

# pl_dia_low_pass_dt = detrend(pl_dia_low_pass)
# matplotlib.pyplot.plot(time,
#                        pl_dia_low_pass_dt, c="black")
# plt.show()


plt.hist(pl_data.diameter, bins=50)
#plt.show()

# getting sampling frequency (difference vector of timestamps)
# diff = []
# for i in range(len(pl_data.timestamp)-1):
#    diff.append(pl_data.timestamp[i + 1] - pl_data.timestamp[i])

# plt.hist(diff, bins=20)
# plt.show()

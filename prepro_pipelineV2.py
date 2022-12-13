"""

Complete preprocessing pipeline for analyzing Pupil Labs pupil data
- using the whole recording
- input is the pupil_data.tsv file from one recording

Steps:
1. Confidence thresholding
2. Filter by SD from median
3. Filtering "dialation speed" outliers - can be done more than once
4. Interpolating (linear or cubic spline)
5. Median filter
6. Butterworth low-pass filter

"""

import numpy as np
import pandas as pd
import matplotlib
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from confidence import conf
from scipy import signal
from scipy.signal import medfilt, butter, lfilter, detrend, filtfilt


# FUNCTIONS

# removing dilation speed outliers
# relying on the guidelines and code from Kret & Sjak-Shie, 2019
def dilation_speed(data, thres_val=3):
    diffs = np.diff(data, prepend=np.nan) / np.diff(time, prepend=np.nan)
    diffs_back = np.diff(data, append=np.nan) / np.diff(time, append=np.nan)
    dil_speed = np.stack((diffs, diffs_back), axis=1)
    max_dil_speed = np.nanmax(abs(dil_speed), axis=1)
    med_d = np.nanmedian(max_dil_speed)
    mad = np.nanmedian(abs(max_dil_speed - med_d))
    mad_threshold = med_d + (thres_val * mad)
    print("MAD threshold value set at: {}" .format(mad_threshold))
    data[max_dil_speed >= mad_threshold] = np.nan

    return data


def missing_data(data):
    count_nan = sum(np.isnan(x) for x in data)
    print("Missing {}% of the data at the moment." .format(round(count_nan/len(data) * 100, 4)))


# the smaller the min_diff value, the shorter the sections of missing data we interpolate
def interpolate(data, min_diff=4, max_samples=10000, method="lin", padding=1, n_samples=2):
    ind = np.array(np.where(np.isnan(data))).flatten()  # find nan-s in the data aka missing values
    # splits data according to the difference of nan indexes (min_diff), we use this to create sections to interpolate over
    missing_ind = np.split(ind, np.where(np.diff(ind) > min_diff)[0]+1)
    data_copy = np.copy(data)  # copy data to interpolate over
    # for i in blink_ind_to_remove:
    for i in missing_ind:
        if i.size == 0:
            continue
        if i.size > max_samples:  # still here but right now does not do anything, we interpolate every gap (even long ones)
            continue
        # create a vector of data and sample numbers before and after the blink
        befores = np.arange((i[0] - (n_samples + padding)), (i[0] - padding))
        afters = np.arange(i[-1] + (1 + padding), i[-1] + (1 + n_samples + padding))
        # this if statement is a contingency for when the blinks occur at the end of the dataset. it deletes the blink rather than interpolating
        if any(afters > len(data) - 1):
            data_noEnds = data_copy[0:i[0] - 1]
            #print(len(data_noEnds))
            end_segment = len(data_copy) - len(data_noEnds)
            data_noEB = np.append(data_noEnds, np.repeat(np.nan, end_segment))
            #print(len(data_noEB))
        else:
            data_noEB = data_copy
        # this is the actual interpolation part. you create your model dataset to interpolate over
            x = np.append(befores, afters)
            y = np.append(data[befores], data[afters])
        # then interpolate it
            if method == "lin":
                li = interp1d(x, y)
                # create indices for the interpolated data, so you can return it to the right segment of the data
                xs = range(i[0] - padding, i[-1] + (1 + padding))
                np.put(data_noEB, xs, li(xs))
            if method == "cubic":
                cubic = interp1d(x, y, kind='cubic')
                # create indices for the interpolated data, so you can return it to the right segment of the data
                xs = range(i[0] - padding, i[-1] + (1 + padding))
                np.put(data_noEB, xs, cubic(xs))

    return data_noEB


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
    data_lp = lfilter(b, a, data)
    data_lp_zerophase = filtfilt(b, a, data)  # this is zero phase, applies the filter once forward and once backward (doubles the original order?)
    return data_lp_zerophase


# PARAMETERS
input_dir = "/media/lucab/data_hdd/lucab/pupildata_CommGame/pair72_Mordor_freeConv/"
cutconf_at = 0.80  # cutoff for confidence values
n_SD = 2  # number of SD-s to cut from median (applied for the whole recording as one of the first steps)
dil_speed_thres = 6  # threshold value for the dilation speed function to use (multiplier of the median)
filt_order = 5  # n-th order median filter


# PREPROCESS DATA

# Apply confidence thresholding
pl_data = conf(input_dir, cutconf_at)

# convert timestamps to readable format
time = pl_data.timestamp - pl_data.timestamp[0]

mean_plsize = np.nanmean(pl_data.diameter)
median_plsize = np.nanmedian(pl_data.diameter)
extr_plsize = [np.nanmin(pl_data.diameter), np.nanmax(pl_data.diameter)]
sd_plsize = np.nanstd(pl_data.diameter)
print("Median pl size: {}" .format(median_plsize))
print("SD: {}" .format(sd_plsize))
# since we have pupil size data in mm, should we cut the very extreme values?
# e.g. outside the 1-9 range


# Filter data by deviation from the median

lower_threshold = median_plsize - n_SD * sd_plsize
higher_threshold = median_plsize + n_SD * sd_plsize
print("Lower SD threshold: {}" .format(lower_threshold))
print("Higher SD threshold: {}" .format(higher_threshold))

pl_data.diameter = pl_data.diameter.where(pl_data.diameter <= higher_threshold)  # replaces with nan-s the ones outside the threshold
pl_data_thresh = pl_data.diameter.where(pl_data.diameter >= lower_threshold)
matplotlib.pyplot.plot(time, pl_data_thresh, c="grey", label="SD thresholded")
# windowed way of SD filtering? big windows (5-10 s)

newdata = np.array(pl_data_thresh)
plt.grid(True)
plt.legend(loc="upper left")
#plt.show()


# Apply dilation speed filtering

print("Filtering dilation speed outliers...")
data_dil1 = dilation_speed(newdata, dil_speed_thres)
matplotlib.pyplot.plot(time, data_dil1, c="red", label="DS outliers removed")
# let's see how much data we lost so far
missing_data(data_dil1)
data_dil2 = dilation_speed(data_dil1, thres_val=5)
matplotlib.pyplot.plot(time, data_dil2, c="blue", label="DS outliers removed (2nd)")
missing_data(data_dil2)
#plt.show()


# Interpolate over missing data

data_interp = interpolate(data_dil2, min_diff=3, padding=1, n_samples=1, method="lin")  # usually needs some adjusting
# data_interp2 = interpolate(data_interp, method="cubic")
print("Interpolation done.")

# plot things so far
matplotlib.pyplot.plot(time, data_interp, c="violet", label="interpolated")
plt.grid(True)
plt.legend(loc="upper left")
#plt.show()

missing_data(data_interp)  # how much do we miss now?

# Apply median filter

med_filt_pldata = scipy.signal.medfilt(data_interp, filt_order)  # automatically zero-pads the data according to filter order
matplotlib.pyplot.plot(time, med_filt_pldata, c="black", label="median filtered")
plt.legend(loc="upper left")
plt.grid()
#plt.show()


# NaN values in input results in all NaN values...
# interpolation took care of most of them, but we might have some left around the edges of the data,
# so we'll just skip those
nans_left = np.argwhere(np.isnan(data_interp))
if len(nans_left) > 0:
    pl_dia_low_pass = butter_lowpass_filter(med_filt_pldata[np.logical_not(np.isnan(med_filt_pldata))],
                                            cutoff=10,
                                            order=5)
    matplotlib.pyplot.plot(time[np.logical_not(np.isnan(med_filt_pldata))], pl_dia_low_pass,
                           c="orange", label="10 Hz low pass")
    plt.legend(loc="best")
    plt.grid()
    plt.show()  # show the smoothed data
else:
    pl_dia_low_pass = butter_lowpass_filter(med_filt_pldata, cutoff=10, order=5)
    matplotlib.pyplot.plot(time, pl_dia_low_pass, c="orange", label="10 Hz low pass")
    plt.legend(loc="best")
    plt.grid()
    plt.show()  # show the smoothed data


# detrending causes (erroneous) shift in pupil diameters (?)
# pldata_detrended = detrend(med_filt_pldata[np.logical_not(np.isnan(med_filt_pldata))])
# time = time[0:len(pldata_detrended)]
# matplotlib.pyplot.plot(time, pldata_detrended, c="blue", label="Detrended")
# plt.legend(loc="lower left")
# plt.grid()
# plt.show()

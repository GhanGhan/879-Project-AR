import numpy as np
from scipy.stats import pearsonr
from scipy.signal import medfilt
from scipy.stats import pearsonr
from scipy.signal import butter
from scipy.signal import hanning
from scipy import signal
from pandas import read_csv
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ///////////////////////////////// LOAD FROM 'WINDOWED_#' DIRECTORY
# /// Time and Window constants
k = 64  # length of axis window
offset = k//2
M = 3  # number of points for moving average filter
NF = 39  # number of features
T = 1/52.0  # Time between samples
# // Frequency constants
fc = 0.5
fn = 26.0
f = fc/fn
# print('cut off', f)
b, a = butter(4, f, 'low', analog=False)
hannW = hanning(k)
# /// Normalization Contants
MSM = 600.0
MDC = 600.0
MAC = 8.0

SDSM = 70.0
SDDC = 7.5
SDAC = 70.0

NSM = 450.0
NDC = 25
NAC = 450.0

RSM = 25.0
RDC = 3.0
RAC = 25.0


directory = 'windowed_' + str(k) + '/'
print('Directory', directory)
data_1 = np.load(directory + '/subject_data_1.npy')
label_1 = np.load(directory + '/subject_label_1.npy')
data_2 = np.load(directory + '/subject_data_2.npy')
label_2 = np.load(directory + '/subject_label_2.npy')
data_3 = np.load(directory + '/subject_data_3.npy')
label_3 = np.load(directory + '/subject_label_3.npy')
data_4 = np.load(directory + '/subject_data_4.npy')
label_4 = np.load(directory + '/subject_label_4.npy')
data_5 = np.load(directory + '/subject_data_5.npy')
label_5 = np.load(directory + '/subject_label_5.npy')
data_6 = np.load(directory + '/subject_data_6.npy')
label_6 = np.load(directory + '/subject_label_6.npy')
data_7 = np.load(directory + '/subject_data_7.npy')
label_7 = np.load(directory + '/subject_label_7.npy')
data_8 = np.load(directory + '/subject_data_8.npy')
label_8 = np.load(directory + '/subject_label_8.npy')
data_9 = np.load(directory + '/subject_data_9.npy')
label_9 = np.load(directory + '/subject_label_9.npy')
data_10 = np.load(directory + '/subject_data_10.npy')
label_10 = np.load(directory + '/subject_label_10.npy')
data_11 = np.load(directory + '/subject_data_11.npy')
label_11 = np.load(directory + '/subject_label_11.npy')
data_12 = np.load(directory + '/subject_data_12.npy')
label_12 = np.load(directory + '/subject_label_12.npy')
data_13 = np.load(directory + '/subject_data_13.npy')
label_13 = np.load(directory + '/subject_label_13.npy')
data_14 = np.load(directory + '/subject_data_14.npy')
label_14 = np.load(directory + '/subject_label_14.npy')
data_15 = np.load(directory + '/subject_data_15.npy')
label_15 = np.load(directory + '/subject_label_15.npy')

print('Data has been loaded')

# Acquiring Sub array reference
print('Shape of motion data: ', data_1.shape)  # (2538, 384)
dr = data_1[0, :]
xr = data_1[0, 0:k]
yr = data_1[0, k:2*k]
zr = data_1[0, 2*k:3*k]
dr[0:3*k] = data_2[0, 0:3*k]
print('Shape of xvalues: ', xr.shape)
print('Shape of yvalues: ', yr.shape)
print('Shape of z: ', zr.shape)
print('Shape of rowvalue: ', dr.shape)  # (128,)

# PRE-PROCESSING: all methods assume that data has been windowed already
# will return the x, y and z mean for each row element
# -------Filter and Normalization


# Filter Data using Moving Average Filter on each row
def butter_filter(data):
    filtered = np.full_like(data, 1)
    rows = len(data)

    for i in range(rows):
        x = data[i, 0: k]
        y = data[i, k: 2 * k]
        z = data[i, 2 * k: 3 * k]

        # get means
        mx = np.mean(x)
        my = np.mean(y)
        mz = np.mean(z)
        # zero center signal
        xc = x - mx
        yc = y - my
        zc = z - mz
        # apply Hanning window
        xh = np.multiply(xc, hannW)
        yh = np.multiply(yc, hannW)
        zh = np.multiply(zc, hannW)
        # apply Butter-worth
        filtered[i, 0:k] = signal.lfilter(b, a, xh)
        filtered[i, 0:k] = medfilt(filtered[i, 0:k])
        filtered[i, k:2*k] = signal.lfilter(b, a, yh)
        filtered[i, k:2 * k] = medfilt(filtered[i, k:2 * k])
        filtered[i, 2*k:3*k] = signal.lfilter(b, a, zh)
        filtered[i, 2 * k:3 * k] = medfilt(filtered[i, 2 * k:3 * k])
        # add mean back
        filtered[i, 0:k] = filtered[i, 0:k] + mx
        filtered[i, k:2 * k] = filtered[i, k:2*k] + my
        filtered[i, 2 * k:3 * k] = filtered[i, 2 * k:3 * k] +mz

    return filtered


# Smooth data py applying median filter to remove spikes and moving average filter to remove nois
def smooth(data):
    rows = len(data)
    normal = np.full_like(data, 1)

    ma = np.ones((M,)) / M
    for i in range(rows):
        # Remove Spikes
        normal[i, 0:k] = medfilt(data[i, 0:k])
        normal[i, k:2*k] = medfilt(data[i, k:2*k])
        normal[i, 2*k:3*k] = medfilt(data[i, 2*k:3*k])

        # apply moving average to take out noise
        normal[i, 0:k] = np.convolve(normal[i, 0:k], ma, mode='same')
        normal[i, k:2 * k] = np.convolve(normal[i, k:2 * k], ma, mode='same')
        normal[i, 2 * k:3 * k] = np.convolve(normal[i, 2 * k:3 * k], ma, mode='same')

    return normal

# ------- Feature Functions


# Calculates the mean value
def mean_acc(x, y, z):
    meanx = np.mean(x)
    meany = np.mean(y)
    meanz = np.mean(z)

    return meanx, meany, meanz


# Calculates the Pearson Correlation Coefficients between all the axis
def pear_corr(x, y, z):

    xy, _ = pearsonr(x, y)
    yz, _ = pearsonr(y, z)
    zx, _ = pearsonr(z, x)
    return xy, yz, zx


# Calculates the Standard Deviation of each axis
def stdev_acc(x, y, z):
    stdx = np.std(x)
    stdy = np.std(y)
    stdz = np.std(z)

    return stdx, stdy, stdz


# Calculate fft of x, y and z
def acc_fft(data_row):
    x_fft = np.fft.fft(data_row[0: k])
    y_fft = np.fft.fft(data_row[k: 2*k])
    z_fft = np.fft.fft(data_row[2*k: 3*k])
    return x_fft, y_fft, z_fft


# Calculate Signal energy
def energy(xk):
    e_fft = np.sum(np.abs(xk[1: k//2])**2)/(k-1)
    return e_fft


# Calculate the Magnitude
def pyth_mag(data_row):
    x = data_row[0: k]
    y = data_row[k: 2 * k]
    z = data_row[2 * k: 3 * k]
    mag = np.zeros(k)
    for i in range(k):
        mag[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)

    mean = np.mean(mag)
    std = np.std(mag)
    fftM = np.fft.fft(mag)
    e_fft = energy(fftM)

    speed = velocity(mag, mean)
    rmsM = np.sqrt(np.sum(speed[1: k]**2)/k)

    minmaxM = np.amax(mag) - np.amin(mag)

    return mean, std, e_fft, rmsM, minmaxM


# Calculate RMS velocity
def rms_velocity(x, y, z):
    meanx = np.mean(x)
    meany = np.mean(y)
    meanz = np.mean(z)
    speedx = velocity(x, meanx)
    speedy = velocity(y, meany)
    speedz = velocity(z, meanz)

    rmsx = np.sqrt(np.sum(speedx[1: k]**2)/k)
    rmsy = np.sqrt(np.sum(speedy[1: k]**2)/k)
    rmsz = np.sqrt(np.sum(speedz[1: k]**2)/k)

    return rmsx, rmsy, rmsz

    # Calculate minmax
def minmax(x, y, z):
    minmaxX = np.amax(x) - np.amin(x)
    minmaxY = np.amax(y) - np.amin(y)
    minmaxZ = np.amax(z) - np.amin(z)

    return minmaxX, minmaxY, minmaxZ


def velocity(values, mean):
    speed = np.zeros(k)
    values = values - mean

    for i in range(1,k):
        speed[i] = speed[i-1] + T*values[i-1]
    return speed


# Acquires all of the features
def get_features(data):
    rows = len(data)
    feature_data = np.zeros((rows, NF))
    motionSM = smooth(data)
    motionDC = butter_filter(motionSM)
    motionAC = motionSM - motionDC
    for i in range(rows):
        # Not Filtered Values
        d_row = motionSM[i, :]
        xSM = d_row[0: k]
        ySM = d_row[k: 2 * k]
        zSM = d_row[2 * k: 3 * k]

        mxSM, mySM, mzSM = mean_acc(xSM, ySM, zSM)  # mean values of xyz axis
        sdxSM, sdySM, sdzSM = stdev_acc(xSM, ySM, zSM)  # st dev of xyz axis
        mmxSM, mmySM, mmzSM = minmax(xSM, ySM, zSM)  # MinMax values of x, y, z

        rmxSM, rmySM, rmzSM = rms_velocity(xSM, ySM, zSM)  # rms velocity of x y and z

        corrxy, corryz, corrzx = pear_corr(xSM, ySM, zSM)

        # LPF Values
        d_row = motionDC[i, :]
        xDC = d_row[0: k]
        yDC = d_row[k: 2 * k]
        zDC = d_row[2 * k: 3 * k]

        mxDC, myDC, mzDC = mean_acc(xDC, yDC, zDC)  # mean values of xyz axis
        sdxDC, sdyDC, sdzDC = stdev_acc(xDC, yDC, zDC)  # st dev of xyz axis
        mmxDC, mmyDC, mmzDC = minmax(xDC, yDC, zDC)  # MinMax values of x, y, z

        rmxDC, rmyDC, rmzDC = rms_velocity(xDC, yDC, zDC)  # rms velocity of x y and z

        # AC Values
        d_row = motionAC[i, :]
        xAC = d_row[0: k]
        yAC = d_row[k: 2 * k]
        zAC = d_row[2 * k: 3 * k]

        mxAC, myAC, mzAC = mean_acc(xAC, yAC, zAC)  # mean values of xyz axis
        sdxAC, sdyAC, sdzAC = stdev_acc(xAC, yAC, zAC)  # st dev of xyz axis
        mmxAC, mmyAC, mmzAC = minmax(xAC, yAC, zAC)  # MinMax values of x, y, z

        rmxAC, rmyAC, rmzAC = rms_velocity(xAC, yAC, zAC)  # rms velocity of x y and z

        feature_data[i, :] = [mxSM, mySM, mzSM,         # 0-2           # No Filter
                              sdxSM, sdySM, sdzSM,      # 3-5
                              mmxSM, mmySM, mmzSM,      # 6-8
                              rmxSM, rmySM, rmzSM,      # 9-11
                              mxDC, myDC, mzDC,         # 12-14                # DC
                              sdxDC, sdyDC, sdzDC,      # 15-17
                              mmxDC, mmyDC, mmzDC,      # 18-20
                              rmxDC, rmyDC, rmzDC,      # 21-23
                              mxAC, myAC, mzAC,         # 24-26                   # AC
                              sdxAC, sdyAC, sdzAC,      # 27-29
                              mmxAC, mmyAC, mmzAC,      # 28-32
                              rmxAC, rmyAC, rmzAC,      # 33-35
                              corrxy, corryz, corrzx]   # 36-39            # rms velocity of x y and z))
        for j in range(36):
            if j < 3:
                feature_data[i, j] = feature_data[i, j]/MSM - 3          # SM
            elif j < 6:
                feature_data[i, j] = feature_data[i, j] / SDSM - 1.5
            elif j < 9:
                feature_data[i, j] = feature_data[i, j] / NSM - 1.4
            elif j < 12:
                feature_data[i, j] = feature_data[i, j] / RSM - 0.1
            elif j < 15:
                feature_data[i, j] = feature_data[i, j] / MDC - 2.9       # DC
            elif j < 18:
                feature_data[i, j] = feature_data[i, j] / SDDC
            elif j < 21:
                feature_data[i, j] = feature_data[i, j] / NDC
            elif j < 24:
                feature_data[i, j] = feature_data[i, j] / RDC
            elif j < 27:
                feature_data[i, j] = feature_data[i, j] / MAC + 0.9     # AC
            elif j < 30:
                feature_data[i, j] = feature_data[i, j] / SDAC - 1.5
            elif j < 33:
                feature_data[i, j] = feature_data[i, j] / NAC - 1.4
            elif j < 36:
                feature_data[i, j] = feature_data[i, j] / RAC - 0.3
            # elif j == 36:
            #     feature_data[i, j] = feature_data[i, j] / RSM
            # elif j == 37:
            #     feature_data[i, j] = feature_data[i, j] / MDC
            # elif j == 38:
            #     feature_data[i, j] = feature_data[i, j] / NDC


    return feature_data


# ------- Get features of all data
features_1 = get_features(data_1)
print('Shape of feature1 vector is', features_1.shape)
features_2 = get_features(data_2)
print('Shape of feature2 vector is', features_2.shape)
features_3 = get_features(data_3)
print('Shape of feature3 vector is', features_3.shape)
features_4 = get_features(data_4)
print('Shape of feature4 vector is', features_4.shape)
features_5 = get_features(data_5)
print('Shape of feature5 vector is', features_5.shape)
features_6 = get_features(data_6)
print('Shape of feature6 vector is', features_6.shape)
features_7 = get_features(data_7)
print('Shape of feature7 vector is', features_7.shape)
features_8 = get_features(data_8)
print('Shape of feature8 vector is', features_8.shape)
features_9 = get_features(data_9)
print('Shape of feature9 vector is', features_9.shape)
features_10 = get_features(data_10)
print('Shape of feature10 vector is', features_10.shape)
features_11 = get_features(data_11)
print('Shape of feature11 vector is', features_11.shape)
features_12 = get_features(data_12)
print('Shape of feature12 vector is', features_12.shape)
features_13 = get_features(data_13)
print('Shape of feature13 vector is', features_13.shape)
features_14 = get_features(data_14)
print('Shape of feature14 vector is', features_14.shape)
features_15 = get_features(data_15)
print('Shape of feature15 vector is', features_15.shape)


# ///////////////////////////////// SAVE FEATURES MUTHAFUCKA
directory = 'features_'+ str(NF) + '_W' + str(k) + 'N/'
print('Directory', directory)
print('Window Size', k)
print('Number of Features', NF)


np.save(directory + '/subject_data_1', features_1)
np.save(directory + '/subject_label_1', label_1)
np.save(directory + '/subject_data_2', features_2)
np.save(directory + '/subject_label_2', label_2)
np.save(directory + '/subject_data_3', features_3)
np.save(directory + '/subject_label_3', label_3)
np.save(directory + '/subject_data_4', features_4)
np.save(directory + '/subject_label_4', label_4)
np.save(directory + '/subject_data_5', features_5)
np.save(directory + '/subject_label_5', label_5)
np.save(directory + '/subject_data_6', features_6)
np.save(directory + '/subject_label_6', label_6)
np.save(directory + '/subject_data_7', features_7)
np.save(directory + '/subject_label_7', label_7)
np.save(directory + '/subject_data_8', features_8)
np.save(directory + '/subject_label_8', label_8)
np.save(directory + '/subject_data_9', features_9)
np.save(directory + '/subject_label_9', label_9)
np.save(directory + '/subject_data_10', features_10)
np.save(directory + '/subject_label_10', label_10)
np.save(directory + '/subject_data_11', features_11)
np.save(directory + '/subject_label_11', label_11)
np.save(directory + '/subject_data_12', features_12)
np.save(directory + '/subject_label_12', label_12)
np.save(directory + '/subject_data_13', features_13)
np.save(directory + '/subject_label_13', label_13)
np.save(directory + '/subject_data_14', features_14)
np.save(directory + '/subject_label_14', label_14)
np.save(directory + '/subject_data_15', features_15)
np.save(directory + '/subject_label_15', label_15)
# Data
df = pd.DataFrame(features_1)
df.to_csv(directory + '/subject_data_1.csv')
df = pd.DataFrame(features_2)
df.to_csv(directory + '/subject_data_2.csv')
df = pd.DataFrame(features_3)
df.to_csv(directory + '/subject_data_3.csv')
df = pd.DataFrame(features_4)
df.to_csv(directory + '/subject_data_4.csv')#
df = pd.DataFrame(features_5)
df.to_csv(directory + '/subject_data_5.csv')
df = pd.DataFrame(features_6)
df.to_csv(directory + '/subject_data_6.csv')
df = pd.DataFrame(features_7)
df.to_csv(directory + '/subject_data_7.csv')
df = pd.DataFrame(features_8)
df.to_csv(directory + '/subject_data_8.csv')#
df = pd.DataFrame(features_9)
df.to_csv(directory + '/subject_data_9.csv')
df = pd.DataFrame(features_10)
df.to_csv(directory + '/subject_data_10.csv')
df = pd.DataFrame(features_11)
df.to_csv(directory + '/subject_data_11.csv')
df = pd.DataFrame(features_12)
df.to_csv(directory + '/subject_data_12.csv')#
df = pd.DataFrame(features_13)
df.to_csv(directory + '/subject_data_13.csv')
df = pd.DataFrame(features_14)
df.to_csv(directory + '/subject_data_14.csv')
df = pd.DataFrame(features_15)
df.to_csv(directory + '/subject_data_15.csv')
# Labels
df = pd.DataFrame(label_1)
df.to_csv(directory + '/subject_label_1.csv')
df = pd.DataFrame(label_2)
df.to_csv(directory + '/subject_label_2.csv')
df = pd.DataFrame(label_3)
df.to_csv(directory + '/subject_label_3.csv')
d_labels = pd.DataFrame(label_4)
df.to_csv(directory + '/subject_label_4.csv')#
df = pd.DataFrame(label_5)
df.to_csv(directory + '/subject_label_5.csv')
df = pd.DataFrame(label_6)
df.to_csv(directory + '/subject_label_6.csv')
df = pd.DataFrame(label_7)
df.to_csv(directory + '/subject_label_7.csv')
df = pd.DataFrame(label_8)
df.to_csv(directory + '/subject_label_8.csv')#
df = pd.DataFrame(label_9)
df.to_csv(directory + '/subject_label_9.csv')
df = pd.DataFrame(label_10)
df.to_csv(directory + '/subject_label_10.csv')
df = pd.DataFrame(label_11)
df.to_csv(directory + '/subject_label_11.csv')
df = pd.DataFrame(label_12)
df.to_csv(directory + '/subject_label_12.csv')#
d_labels = pd.DataFrame(label_13)
df.to_csv(directory + '/subject_label_13.csv')
df = pd.DataFrame(label_14)
df.to_csv(directory + '/subject_label_14.csv')
df = pd.DataFrame(label_15)
df.to_csv(directory + '/subject_label_15.csv')

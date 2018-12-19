import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ///////////////////////////////// LOAD FROM 'raw_values' DIRECTORY
directory = 'raw_values/'
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

# ///////////////////////////////// APPLY WINDOW
# k: size of the sub-array (in rows)
# ol: offset length

# For data reformatting
def new_motion_perfect_data(arr, k, ol):
    j = 1
    print('New motion perfect: Data')
    final = []
    while j < len(arr) and ol*(j-1)+k < len(arr):
        a1 = arr[ol*(j-1): ol*(j-1)+k, 0]
        a2 = arr[ol*(j-1): ol*(j-1)+k, 1]
        a3 = arr[ol*(j-1): ol*(j-1)+k, 2]
        b = np.concatenate((a1, a2, a3), axis=None)
        if j == 1:
            final = b
        else:
            final = np.vstack((final, b))

        j = j+1

    return final


# For label reformatting
def new_motion_perfect_label(arr, k, ol):
    j = 1
    final = []
    a = [100]
    print('New motion perfect: Label')
    while j < len(arr) and ol*(j-1)+k < len(arr):
        a[0] = np.median(arr[ol*(j-1): ol*(j-1)+k])
        j = j+1
        if j == 1:
            final = a
        else:
            final = np.concatenate((final, a))

    return final

# On actual data set
window = 64
offset = window//2
print('Window Size', window)
print('Offset Size', offset)
w_Data_1 = new_motion_perfect_data(data_1, window, offset)
print('Size of windowed data', w_Data_1.shape)
w_Label_1 = new_motion_perfect_label(label_1, window, offset)
w_Data_2 = new_motion_perfect_data(data_2, window, offset)
w_Label_2 = new_motion_perfect_label(label_2, window, offset)
w_Data_3 = new_motion_perfect_data(data_3, window, offset)
w_Label_3 = new_motion_perfect_label(label_3, window, offset)
w_Data_4 = new_motion_perfect_data(data_4, window, offset)
w_Label_4 = new_motion_perfect_label(label_4, window, offset)
w_Data_5 = new_motion_perfect_data(data_5, window, offset)
w_Label_5 = new_motion_perfect_label(label_5, window, offset)
w_Data_6 = new_motion_perfect_data(data_6, window, offset)
w_Label_6 = new_motion_perfect_label(label_6, window, offset)
w_Data_7 = new_motion_perfect_data(data_7, window, offset)
w_Label_7 = new_motion_perfect_label(label_7, window, offset)
w_Data_8 = new_motion_perfect_data(data_8, window, offset)
w_Label_8 = new_motion_perfect_label(label_8, window, offset)
w_Data_9 = new_motion_perfect_data(data_9, window, offset)
w_Label_9 = new_motion_perfect_label(label_9, window, offset)
w_Data_10 = new_motion_perfect_data(data_10, window, offset)
w_Label_10 = new_motion_perfect_label(label_10, window, offset)
w_Data_11 = new_motion_perfect_data(data_11, window, offset)
w_Label_11 = new_motion_perfect_label(label_11, window, offset)
w_Data_12 = new_motion_perfect_data(data_12, window, offset)
w_Label_12 = new_motion_perfect_label(label_12, window, offset)
w_Data_13 = new_motion_perfect_data(data_13, window, offset)
w_Label_13 = new_motion_perfect_label(label_13, window, offset)
w_Data_14 = new_motion_perfect_data(data_14, window, offset)
w_Label_14 = new_motion_perfect_label(label_14, window, offset)
w_Data_15 = new_motion_perfect_data(data_15, window, offset)
w_Label_15 = new_motion_perfect_label(label_15, window, offset)
# ///////////////////////////////// SAVE WINDOWED 
# ----------Numpy arrays
directory = 'windowed_' + str(window) + '/'
np.save(directory + '/subject_data_1', w_Data_1)
np.save(directory + '/subject_label_1', w_Label_1)
np.save(directory + '/subject_data_2', w_Data_2)
np.save(directory + '/subject_label_2', w_Label_2)
np.save(directory + '/subject_data_3', w_Data_3)
np.save(directory + '/subject_label_3', w_Label_3)
np.save(directory + '/subject_data_4', w_Data_4)
np.save(directory + '/subject_label_4', w_Label_4)
np.save(directory + '/subject_data_5', w_Data_5)
np.save(directory + '/subject_label_5', w_Label_5)
np.save(directory + '/subject_data_6', w_Data_6)
np.save(directory + '/subject_label_6', w_Label_6)
np.save(directory + '/subject_data_7', w_Data_7)
np.save(directory + '/subject_label_7', w_Label_7)
np.save(directory + '/subject_data_8', w_Data_8)
np.save(directory + '/subject_label_8', w_Label_8)
np.save(directory + '/subject_data_9', w_Data_9)
np.save(directory + '/subject_label_9', w_Label_9)
np.save(directory + '/subject_data_10', w_Data_10)
np.save(directory + '/subject_label_10', w_Label_10)
np.save(directory + '/subject_data_11', w_Data_11)
np.save(directory + '/subject_label_11', w_Label_11)
np.save(directory + '/subject_data_12', w_Data_12)
np.save(directory + '/subject_label_12', w_Label_12)
np.save(directory + '/subject_data_13', w_Data_13)
np.save(directory + '/subject_label_13', w_Label_13)
np.save(directory + '/subject_data_14', w_Data_14)
np.save(directory + '/subject_label_14', w_Label_14)
np.save(directory + '/subject_data_15', w_Data_15)
np.save(directory + '/subject_label_15', w_Label_15)
# ----------CSV files
# Data
df = pd.DataFrame(w_Data_1)
df.to_csv(directory + '/subject_data_1.csv')
df = pd.DataFrame(w_Data_2)
df.to_csv(directory + '/subject_data_2.csv')
df = pd.DataFrame(w_Data_3)
df.to_csv(directory + '/subject_data_3.csv')
df = pd.DataFrame(w_Data_4)
df.to_csv(directory + '/subject_data_4.csv')#
df = pd.DataFrame(w_Data_5)
df.to_csv(directory + '/subject_data_5.csv')
df = pd.DataFrame(w_Data_6)
df.to_csv(directory + '/subject_data_6.csv')
df = pd.DataFrame(w_Data_7)
df.to_csv(directory + '/subject_data_7.csv')
df = pd.DataFrame(w_Data_8)
df.to_csv(directory + '/subject_data_8.csv')#
df = pd.DataFrame(w_Data_9)
df.to_csv(directory + '/subject_data_9.csv')
df = pd.DataFrame(w_Data_10)
df.to_csv(directory + '/subject_data_10.csv')
df = pd.DataFrame(w_Data_11)
df.to_csv(directory + '/subject_data_11.csv')
df = pd.DataFrame(w_Data_12)
df.to_csv(directory + '/subject_data_12.csv')#
df = pd.DataFrame(w_Data_13)
df.to_csv(directory + '/subject_data_13.csv')
df = pd.DataFrame(w_Data_14)
df.to_csv(directory + '/subject_data_14.csv')
df = pd.DataFrame(w_Data_15)
df.to_csv(directory + '/subject_data_15.csv')
# Labels
df = pd.DataFrame(w_Label_1)
df.to_csv(directory + '/subject_label_1.csv')
df = pd.DataFrame(w_Label_2)
df.to_csv(directory + '/subject_label_2.csv')
df = pd.DataFrame(w_Label_3)
df.to_csv(directory + '/subject_label_3.csv')
df = pd.DataFrame(w_Label_4)
df.to_csv(directory + '/subject_label_4.csv')#
df = pd.DataFrame(w_Label_5)
df.to_csv(directory + '/subject_label_5.csv')
df = pd.DataFrame(w_Label_6)
df.to_csv(directory + '/subject_label_6.csv')
df = pd.DataFrame(w_Label_7)
df.to_csv(directory + '/subject_label_7.csv')
df = pd.DataFrame(w_Label_8)
df.to_csv(directory + '/subject_label_8.csv')#
df = pd.DataFrame(w_Label_9)
df.to_csv(directory + '/subject_label_9.csv')
df = pd.DataFrame(w_Label_10)
df.to_csv(directory + '/subject_label_10.csv')
df = pd.DataFrame(w_Label_11)
df.to_csv(directory + '/subject_label_11.csv')
df = pd.DataFrame(w_Label_12)
df.to_csv(directory + '/subject_label_12.csv')#
df = pd.DataFrame(w_Label_13)
df.to_csv(directory + '/subject_label_13.csv')
df = pd.DataFrame(w_Label_14)
df.to_csv(directory + '/subject_label_14.csv')
df = pd.DataFrame(w_Label_15)
df.to_csv(directory + '/subject_label_15.csv')


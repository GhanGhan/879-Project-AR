# Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# ///////////////////////// LOAD ACCELERATION DATA AND TABLES
num_features = 39  # Number of features
window = 64
sets = 15

num_trees = 200
max_nodes = 1000
RSEED = 50
depth = 100

directory = 'features_' + str(num_features) + '_W' + str(window) + 'N/'
print('Directory: ', directory)
print('Window Size', window)
print('Number of Features', num_features)

names = ['Working at PC', 'Standing Up, Walking and Going updown stairs', 'Standing', 'Walking',
         'Going UpDown Stairs', 'Walking and Talking with Someone', 'Talking while Standing']

# ///////////////////////// LOAD ACCELERATION DATA AND TABLES
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
# ///////////////////////// TRAINING DATA SETS AND ASSOCIATED TEST SETS
# Collect into Array of arrays
# Append Motion Data
motion_data = []
motion_data.append(data_1)
motion_data.append(data_2)
motion_data.append(data_3)
motion_data.append(data_4)
motion_data.append(data_5)
motion_data.append(data_6)
motion_data.append(data_7)
motion_data.append(data_8)
motion_data.append(data_9)
motion_data.append(data_10)
motion_data.append(data_11)
motion_data.append(data_12)
motion_data.append(data_13)
motion_data.append(data_14)
motion_data.append(data_15)
# Append Motion Labels
motion_label = []
motion_label.append(label_1)
motion_label.append(label_2)
motion_label.append(label_3)
motion_label.append(label_4)
motion_label.append(label_5)
motion_label.append(label_6)
motion_label.append(label_7)
motion_label.append(label_8)
motion_label.append(label_9)
motion_label.append(label_10)
motion_label.append(label_11)
motion_label.append(label_12)
motion_label.append(label_13)
motion_label.append(label_14)
motion_label.append(label_15)

# Accuracies
TN_Accuracy = np.empty([sets, 1])
TT_Accuracy = np.empty([sets, 1])
# Precision
T_Precision = np.empty([sets, 1])
# Recall
T_Recall = np.empty([sets, 1])
# F1-Score
T_F1 = np.empty([sets, 1])
# Average Confusion Matrix
con_mat = []
# leave one out cross validation
test = 1
for i in range(15):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    add_train = False
    for j in range(15):
        if test != j+1:
            if add_train == False:
                train_data = motion_data[j]
                train_label = motion_label[j]
                add_train = True
                # print('Subject placed in train set', j + 1)
                continue
            train_data = np.concatenate((train_data, motion_data[j]))
            train_label = np.concatenate((train_label, motion_label[j]))
            # print('Subject placed in train set', j + 1)
        else:
            test_data = motion_data[j]
            test_label = motion_label[j]
            # print('Testing on subject: ', test)
            # print('Subject placed in test set', j+1)
    # print('Shape of train_data:', train_data.shape)
    # print('Shape of train_label: ', train_label.shape)
    # print('Shape of test_data:', test_data.shape)
    # print('Shape of test_label: ', test_label.shape)
    # ////////////////////////////////////////////////////Validation will take place here

    # Ensure labels are integers
    test_label = test_label.astype(dtype=np.int64)
    train_label = train_label.astype(dtype=np.int64)
    test = i + 1
    print("Validation will take place here, iteration", test)
    # ///////////////////////// Random Forest
    model = RandomForestClassifier(n_estimators=num_trees,
                                   random_state=RSEED,
                                  # criterion='gini',
                                  # max_depth=depth,
                                   max_features='log2',
                                  # class_weight='balanced_subsample',
                                   n_jobs=-1)
    if i == 0:
        print(model.get_params())

    #////////////////////////// TRAIN Random Forest
    model.fit(train_data, train_label)
    # Training predictions (to determine performance)
    train_predictions = model.predict(train_data)
    train_accuracy = accuracy_score(train_label, train_predictions, normalize=True)
    print("Train Accuracy: ", train_accuracy)

    # Testing predictions (to determine performance)
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy_score(test_label, test_predictions, normalize=True)
    test_precision = precision_score(test_label, test_predictions, names, average='micro')
    test_recall = recall_score(test_label, test_predictions, names, average='micro')
    test_F1 = f1_score(test_label, test_predictions, names, average='micro')

    print("Test Accuracy: ", test_accuracy)
    if i == 0:
        con_mat = confusion_matrix(test_label, test_predictions)
    else:
        con_mat = con_mat + confusion_matrix(test_label, test_predictions)

    TN_Accuracy[i] = train_accuracy
    TT_Accuracy[i] = test_accuracy
    T_Precision[i] = test_precision
    T_Recall[i] = test_recall
    T_F1[i] = test_F1


print('Confusion matrix')
con_mat = con_mat/float(sets)
print(con_mat)

print('test_accuracy')
print(TT_Accuracy)
print('Average test accuracy', np.sum(TT_Accuracy)/15.0)

print('train accuracy')
print(TN_Accuracy)

T_Accuracy = np.concatenate((TN_Accuracy, TT_Accuracy), axis=1)
df = pd.DataFrame(T_Accuracy)
df.to_csv('RFL.csv')
df = pd.DataFrame(con_mat)
df.to_csv('RFL_confusion_matrix.csv')

# From top to bottom, Accuracy,  Precision, Recall, F1
D = 15.0
stats = [np.sum(TT_Accuracy)/D, np.sum(T_Precision)/D, np.sum(T_Recall)/D, np.sum(T_F1)/D]
df = pd.DataFrame(stats)
df.to_csv('RFLStats.csv')


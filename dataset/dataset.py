import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from colorama import Fore
from under_sampling import Method, get_undersampled_dataset

raw_df = pd.read_csv("./data/creditcard_2000.csv")

neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Raw data stat =========================')
print('Total: {}\nPositive: {} ({:.2f}% of total)'
    .format(total, pos, 100 * pos / total))

cleaned_df = raw_df.copy()
cleaned_df.pop('Time')
eps = 0.001
cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount') + eps)

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# print('Training labels shape:', train_labels.shape)
# print('Validation labels shape:', val_labels.shape)
# print('Test labels shape:', test_labels.shape)
# print('Training features shape:', train_features.shape)
# print('Validation features shape:', val_features.shape)
# print('Test features shape:', test_features.shape)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

# Sampling take place here
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]
pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

print(Fore.RED + 'Before sampling =======================' + Fore.RESET)
print('Pos features shape:', pos_features.shape)
print('Neg features shape:', neg_features.shape)

train_features, train_labels = \
    get_undersampled_dataset(Method.RANDOM, train_features, train_labels)
bool_train_labels = train_labels != 0

pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]
pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

print(Fore.GREEN + 'After sampling ========================' + Fore.RESET)
print('Pos features shape:', pos_features.shape)
print('Neg features shape:', neg_features.shape)

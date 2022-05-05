import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from .under_sampling import UndersampleMethod, undersample_with
from .oversampling import OversampleMethod, oversample_with

def process_dataset(dataset, test_ratio=0.2, val_ratio=0.2, min_clip=-5, max_clip=5):
    df = pd.read_csv(dataset)

    # neg, pos = np.bincount(df['Class'])
    # total = neg + pos
    # print('Raw data stat =========================')
    # print('Total: {}\nPositive: {} ({:.2f}% of total)'
    #     .format(total, pos, 100 * pos / total))

    eps = 0.001
    df['Log Time'] = np.log(df.pop('Time') + eps)
    df['Log Amount'] = np.log(df.pop('Amount') + eps)

    train_df, test_df = train_test_split(df, test_size=test_ratio)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio)

    train_labels = np.array(train_df.pop('Class'))
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

    train_features = np.clip(train_features, min_clip, max_clip)
    val_features = np.clip(val_features, min_clip, max_clip)
    test_features = np.clip(test_features, min_clip, max_clip)

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def get_balanced_dataset(train_features, train_labels, oversample=True):
    return \
        oversample_with(OversampleMethod.SYNTHETIC, train_features, train_labels) if oversample else \
        undersample_with(UndersampleMethod.CONDENSED, train_features, train_labels)


def _print_sampling_stat(train_features, train_labels):
    bool_train_labels = train_labels != 0
    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]
    print('Pos features shape:', pos_features.shape)
    print('Neg features shape:', neg_features.shape)

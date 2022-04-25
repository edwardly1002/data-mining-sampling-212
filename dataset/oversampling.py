import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import StandardScaler

DATASET = "../data/creditcard_2000.csv"
RESULTDIR= "./oversampling/"


class OversampleMethod:
    RANDOM = RandomOverSampler()
    SYNTHETIC= SMOTE()


def oversample_with(strategy, X, y):
    return strategy.fit_resample(X, y)

if __name__ =="__main__":
    df = pd.read_csv(DATASET)
    
    # SSK or train_test_split
    eps=0.01
    df['Amount'] = np.log(df['Amount'].values.reshape(-1,1)+eps)
    df['Time'] = np.log(df['Time'].values.reshape(-1,1)+eps)
    
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    
    # drop class 
    train_labels = np.array(train_df.pop('Class'))
    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))
    
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    train_features=StandardScaler().fit_transform(train_features)
    train_features, train_labels = oversample_with(OversampleMethod.SYNTHETIC, train_df, train_labels)
    
    # test to file log
    print(train_features,train_labels)

EPOCH = 1000
BATCH_SIZE = 32
STEPS_PER_EPOCH = None
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATASET_DIR = "data/creditcard.csv"

DEBUG = True
def log(s):
    from datetime import datetime
    if DEBUG: print(f'[DEBUG] {str(datetime.now())}:\t{s}')
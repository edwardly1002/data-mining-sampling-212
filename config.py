EPOCH = 1000
BATCH_SIZE = 32
STEPS_PER_EPOCH = None
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATASET_DIR = "data/creditcard.csv"

DEBUG = True
LOG_DIR = None
def log(s):
    from datetime import datetime
    if DEBUG and not LOG_DIR: print(f'[DEBUG] {str(datetime.now())}:\t{str(s)}')
    elif DEBUG and LOG_DIR: 
        with open(LOG_DIR, 'a') as f:
            f.write(f'[DEBUG] {str(datetime.now())}:\t{str(s)}\n')
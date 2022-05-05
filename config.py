EPOCH = 1000
BATCH_SIZE = 32
STEPS_PER_EPOCH = None
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATASET_DIR = "data/creditcard.csv"

DEBUG = True
def log(s, log_dir=None):
    from datetime import datetime
    if DEBUG and not log_dir: print(f'[DEBUG] {str(datetime.now())}:\t{str(s)}')
    elif DEBUG and log_dir: 
        with open(log_dir, 'a') as f:
            f.write(f'[DEBUG] {str(datetime.now())}:\t{str(s)}\n')
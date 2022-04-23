from dataset import *
from model.SimpleCNN import SimpleCNN

import tensorflow as tf
from config import *

def main():
    ### STEP 1: PREPARE DATASET
    train_dataset = (
        tf.data.Dataset.from_generator(
            ### generator, 
            # output_signature = (
            #     tf.TensorSpec(shape=(), dtype=tf.float32), 
            #     tf.TensorSpec(shape=(), dtype=tf.float32,)  
            # )
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        # .prefetch(tf.data.experimental.AUTOTUNE)
    )

    ### STEP 2: Build model
    model = SimpleCNN()
    model.build(input_shape = ())
    model.summary()
    model.compile(
        # optimizer=,
        # loss = BinaryCrossentropy(), 
        # metrics = ['accuracy', precision_m, recall_m, f1_m]
    )
    
    model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        # ckpt_dir,
        # monitor='val_f1',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch',
    )
    
    ### STEP 3: TRAIN
    history = model.fit(
        train_dataset,
        # validation_data=test_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCH,
        verbose=1,
        callbacks=[model_checkpoint_callback]
    )

if __name__ == "__main__":
    main()

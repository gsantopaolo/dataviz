import os
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from wandb.integration.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint


# Retrieve API key from environment variable and log in to W&B.
api_key = os.environ.get("WANDB_API_KEY")
if api_key is None:
    raise ValueError("WANDB_API_KEY environment variable is not set.")
wandb.login(key=api_key)

def load_and_preprocess_data(batch_size=128):
    """
    Loads the MNIST dataset and creates training and testing pipelines.

    Returns:
        ds_train: Training dataset with normalization, shuffling, and batching.
        ds_test: Testing dataset with normalization and batching.
        ds_info: Dataset metadata (contains info like number of examples).
    """
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

def create_model():
    """
    Creates a basic feed-forward neural network model for MNIST.

    Returns:
        A compiled tf.keras.Sequential model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(wandb.config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def train_model(model, ds_train, ds_test, epochs=6):
    """
    Trains the model on the training dataset and validates it on the test dataset.
    Uses the WandbCallback (with graph logging disabled) to log metrics and model data to W&B.
    """
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        callbacks=[WandbCallback(log_graph=False)]
    )


def main():
    """
    Main function that orchestrates the data loading, model creation, and training process.
    It now starts by logging into W&B and initializing a run.
    """
    # Initialize a new W&B run with project and hyperparameter config.
    wandb.init(project="mnist-tf", config={
        "learning_rate": 0.001,
        "epochs": 12,
        "batch_size": 128,
    })

    # Load and preprocess the MNIST dataset.
    ds_train, ds_test, ds_info = load_and_preprocess_data(batch_size=wandb.config.batch_size)

    # Create the model.
    model = create_model()

    # Train the model.
    # train_model(model, ds_train, ds_test, epochs=wandb.config.epochs)
    model.fit(
        ds_train,
        epochs=wandb.config.epochs,
        validation_data=ds_test,
        callbacks=[
            WandbMetricsLogger(),
            WandbModelCheckpoint("models.keras")
        ]
    )

    # Finish the run.
    wandb.finish()

if __name__ == '__main__':
    main()

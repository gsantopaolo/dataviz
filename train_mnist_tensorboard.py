import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Define log directory and metadata file
LOG_DIR = os.path.join(os.getcwd(), 'logs', 'mnist')

def load_and_preprocess_data(batch_size=128):
    """
    Loads the MNIST dataset and creates training and testing pipelines.

    Returns:
        ds_train: Training dataset with normalization, shuffling, and batching.
        ds_test: Testing dataset with normalization and batching.
        ds_info: Dataset metadata (contains info like number of examples).
    """
    # Load the MNIST dataset as supervised (image, label) tuples along with dataset info.
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,  # Shuffle file order (useful for large datasets)
        as_supervised=True,  # Returns (image, label) tuples
        with_info=True,  # Includes metadata about the dataset
    )

    def normalize_img(image, label):
        """
        Normalizes images: converts from uint8 to float32 and scales to [0, 1].
        """
        return tf.cast(image, tf.float32) / 255., label

    # Prepare the training dataset pipeline:
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()  # Cache the dataset in memory for faster access
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  # Prefetch to improve latency

    # Prepare the testing dataset pipeline:
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()  # Cache for performance
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info


def create_model():
    """
    Creates a basic feed-forward neural network model for MNIST.

    Returns:
        A compiled tf.keras.Sequential model.
    """
    model = tf.keras.models.Sequential([
        # Flatten the 28x28 image into a 784-dimensional vector.
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        # Dense layer with 128 neurons and ReLU activation.
        tf.keras.layers.Dense(128, activation='relu'),
        # Output layer with 10 neurons (one per MNIST digit).
        tf.keras.layers.Dense(10)
    ])

    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


def train_model(model, ds_train, ds_test, epochs=6):
    """
    Trains the model on the training dataset and validates it on the test dataset.

    Args:
        model: The compiled Keras model.
        ds_train: Preprocessed training dataset.
        ds_test: Preprocessed testing dataset.
        epochs: Number of training epochs.
    """
    # Creates log dir if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a TensorBoard callback to monitor training progress
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,  # Directory where TensorBoard logs (scalars, histograms, graphs) will be stored.
        histogram_freq=1  # Logs weight and activation histograms every epoch.
    )

    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        callbacks=[tensorboard_callback]  # Add TensorBoard callback
    )


def main():
    """
    Main function that orchestrates the data loading, model creation, and training process.
    """
    # Load and preprocess the MNIST dataset.
    ds_train, ds_test, ds_info = load_and_preprocess_data()

    # Create the model.
    model = create_model()

    # Train the model with the prepared datasets.
    train_model(model, ds_train, ds_test, 20)

    # Save the trained model
    model.save("model.keras")  # Saves in Keras format
    # or
    # model.save("model.h5")  # Saves in HDF5 format

    print("Training complete. Launch TensorBoard with:")
    print(f"tensorboard --logdir={LOG_DIR}")


# Entry point of the script
if __name__ == '__main__':
    main()

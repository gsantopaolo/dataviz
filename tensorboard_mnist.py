import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins import projector

# Disable eager execution so we can use TF1-style summary writer.
tf.compat.v1.disable_eager_execution()

# Define log directory and metadata file
LOG_DIR = os.path.join(os.getcwd(), 'mnist-tensorboard', 'log-1')
os.makedirs(LOG_DIR, exist_ok=True)
metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Load MNIST test dataset with as_supervised=True (returns image, label pairs)
ds_test = tfds.load('mnist', split='test', as_supervised=True)

# Collect a fixed number of examples for visualization
images = []
labels = []
for image, label in tfds.as_numpy(ds_test.take(10000)):
    images.append(image.reshape(-1))  # flatten 28x28 image
    labels.append(label)
images = np.array(images)

# Create a TensorFlow variable for the embeddings
embedding_var = tf.Variable(images, name='embedding')

# Write out the metadata file (add header and 10,000 label lines -> 10001 total lines)
with open(metadata_path, 'w') as f:
    f.write("label\n")
    for label in labels:
        f.write(f"{label}\n")

# Save the checkpoint for the embedding variable
saver = tf.compat.v1.train.Saver([embedding_var])
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.save(sess, os.path.join(LOG_DIR, "embedding.ckpt"))

    # Configure the projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Use an absolute path for the metadata
    embedding.metadata_path = os.path.abspath(metadata_path)

    # Use a TF1 summary writer
    writer = tf.compat.v1.summary.FileWriter(LOG_DIR, sess.graph)
    projector.visualize_embeddings(writer, config)
    writer.close()

print("Setup complete. Launch TensorBoard with:")
print(f"tensorboard --logdir={LOG_DIR}")

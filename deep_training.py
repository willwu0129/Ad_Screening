
# This neural network architecture takes the vectorized articles and their labels as input.
# Then one-dimension vertorized input is fed into a multi-layer model.
# The parameters for training process is saved along the training path (/tmp/mnist_logs). They can be viewed on tensorboard.

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


import tensorflow as tf
import art_vectorizing as av
import data_helper as dh
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')
flags.DEFINE_float('test_portion', 0.1, "This decides how large the test set is")
flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 200)")
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_of_classes", 2, "Two categories: yes or no")
flags.DEFINE_integer("h1_size", 10, "The size of the first hidden layer")

def train():
    # Import data
    x_raw, y_raw = av.vectorizing('D')


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_raw)))
    x_shuffled = x_raw[shuffle_indices]
    y_shuffled = y_raw[shuffle_indices]
    test_set_size = int(len(y_raw) * FLAGS.test_portion)
    x_train, x_dev = x_shuffled[:-test_set_size], x_shuffled[-test_set_size:]
    y_train, y_dev = y_shuffled[:-test_set_size], y_shuffled[-test_set_size:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Start a session
    sess = tf.InteractiveSession()


    # Create a weight variable with appropriate initialization.
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    # Create a bias variable with appropriate initialization.
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    # Attach a lot of summaries to a Tensor.
    def variable_summaries(var, name):
        with tf.name_scope('summaries'):
            with tf.name_scope('mean'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
            with tf.name_scope('max'):
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
            with tf.name_scope('min'):
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)


    # Reusable code for making a simple neural net layer
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            tf.histogram_summary(layer_name + '/activations', activations)
            return activations


    # Create a multilayer model.
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, len(x_raw[0])], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_of_classes], name='y-input')
    # First hidden layer
    hidden1 = nn_layer(x, len(x_raw[0]), FLAGS.h1_size, 'layer1')
    # Add dropout for the first hidden layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
    # Output layer
    y = nn_layer(dropped, FLAGS.h1_size, FLAGS.num_of_classes, 'layer2', act=tf.nn.softmax)
    with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y)
    with tf.name_scope('total'):
        cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)
    # Accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
    # Define training step
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        train_step = optimizer.apply_gradients(grads_and_vars)


    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()

    # Generate batches
    batch_generator = dh.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    # Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
    def feed_dict(train, X=x_dev, Y=y_dev):
        if train or FLAGS.fake_data:
            k = FLAGS.dropout
        else:
            k = 1.0
        return {x: X, y_: Y, keep_prob: k}

    # Start the training and evaluating processes
    ind = 1
    for batch in batch_generator:
        x_batch, y_batch = zip(*batch)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, acc, _ = sess.run([merged, accuracy, train_step], feed_dict=feed_dict(True, x_batch, y_batch),
                                   options=run_options, run_metadata=run_metadata)
        print('Accuracy at step for training %s: %s' % (ind, acc))
        train_writer.add_summary(summary, ind)
        if ind % FLAGS.evaluate_every == 0:
            train_writer.add_run_metadata(run_metadata, 'step%03d' % ind)
            print('Adding run metadata for %d' % ind)
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, ind)
            print('Accuracy at step for test %s: %s' % (ind, acc))
        ind += 1
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

# This part shows how to wire the CNN structure up in tensorflow. The behaviours of CNN is defined in TextCNN

from tensorflow.contrib import learn
import numpy as np
import data_helper as dh
import tensorflow as tf
from TextCNN import TextCNN
import os
import time
import datetime
import data_helper
import csv

tf.logging.set_verbosity(tf.logging.INFO)


# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "50,100,150", "Comma-separated filter sizes (default: '10,20,30')")
tf.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 800, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 8, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_of_classes", 2, "Two categories: yes or no")
tf.flags.DEFINE_float("test_portion", 0.1, "This decides how large the test set is")
tf.flags.DEFINE_float("art_length_percentile", 95, "This decides the length of the article length ")
tf.flags.DEFINE_string("category", "D", "This decides which category it belongs to")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================
# loading data ...
if not os.path.isfile(os.path.join(os.getcwd(), 'training', FLAGS.category, 'x_train.csv')):
    articles, y, art_size = dh.art_seg_new(FLAGS.category)
    art_lens = np.array([len(a.split(" ")) for a in articles])
    articles_new = []
    y_new = []
    max_document_length = max(art_lens)
    if max_document_length < 1000:
        articles_new = articles
        y_new = y
    else:
        max_document_length = int(np.percentile(art_lens, FLAGS.art_length_percentile))

        for ind in range(len(art_lens)):
            if art_lens[ind] < max_document_length:
                articles_new.append(articles[ind])
                y_new.append(y[ind])

        y_new = np.array(y_new)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_new = list(vocab_processor.fit_transform(articles_new))
    x_new = np.array(x_new)

    vs = len(vocab_processor.vocabulary_)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_new)))
    x_shuffled = x_new[shuffle_indices]
    y_shuffled = y_new[shuffle_indices]
    test_set_size = int(len(y_new)*FLAGS.test_portion)
    x_train, x_dev = x_shuffled[:-test_set_size], x_shuffled[-test_set_size:]
    y_train, y_dev = y_shuffled[:-test_set_size], y_shuffled[-test_set_size:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'x_train.csv'), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(x_train)
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'x_dev.csv'), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(x_dev)
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'y_train.csv'), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(y_train)
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'y_dev.csv'), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(y_dev)
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'dic_size.csv'), "wb") as f:
        writer = csv.writer(f)
        writer.writerows([[vs]])

else:
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'x_train.csv'), 'rb') as f:
        reader = csv.reader(f)
        x_train = list(reader)
        x_train = np.array([np.array(map(int, item)) for item in x_train])
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'x_dev.csv'), 'rb') as f:
        reader = csv.reader(f)
        x_dev = list(reader)
        x_dev = np.array([np.array(map(int, item)) for item in x_dev])
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'y_train.csv'), 'rb') as f:
        reader = csv.reader(f)
        y_train = list(reader)
        y_train = np.array([np.array(map(int, item)) for item in y_train])
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'y_dev.csv'), 'rb') as f:
        reader = csv.reader(f)
        y_dev = list(reader)
        y_dev = np.array([np.array(map(int, item)) for item in y_dev])
    max_document_length = len(x_train[0])
    with open(os.path.join(os.getcwd(), 'training', FLAGS.category, 'dic_size.csv'), "rb") as f:
        reader = csv.reader(f)
        vs = list(reader)
        vs = int(vs[0][0])

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            art_length=max_document_length,
            num_classes=FLAGS.num_of_classes,
            vocab_size=vs,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helper.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))



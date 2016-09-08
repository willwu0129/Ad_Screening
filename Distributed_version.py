
# this is a distributed version of CNN model. In this example, the ps_host is the one holding all the variables and parameters during the training
# worker_host is the one who actually does the computational work.
# The way neural network works is the same as CNN.

from tensorflow.contrib import learn
import numpy as np
import data_helper as dh
import tensorflow as tf
from TextCNN import TextCNN
import os
import time
import datetime
import data_helper
# Parameters
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "10.2.58.50:2225", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "10.2.58.202:2225", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim", 16, "Dimensionality of character embedding (default: 16)")
tf.app.flags.DEFINE_string("filter_sizes", "50,100,150", "Comma-separated filter sizes (default: '50,100,150')")
tf.app.flags.DEFINE_integer("num_filters", 16, "Number of filters per filter size (default: 16)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 1)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularizaion lambda (default: 1)")
# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 800, "Batch Size (default: 800)")
tf.app.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 30)")
tf.app.flags.DEFINE_integer("evaluate_every", 3, "Evaluate model on dev set after this many steps (default: 3)")
tf.app.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 2000)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_integer("num_of_classes", 2, "Two categories: yes or no")
tf.app.flags.DEFINE_float("test_portion", 0.1, "This decides how large the test set is")
tf.app.flags.DEFINE_string("category", "P", "This denoted the category used to perform classification")

FLAGS = tf.app.flags.FLAGS

# Display flags
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Data Preparation
            # ==================================================
            # loading data ...

            articles, y, art_size = dh.art_seg_new(FLAGS.category)
            max_document_length = max([len(a.split(" ")) for a in articles])
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            x = list(vocab_processor.fit_transform(articles))
            x = np.array(x)

            # Randomly shuffle data
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            test_set_size = int(len(articles) * FLAGS.test_portion)
            x_train, x_dev = x_shuffled[:-test_set_size], x_shuffled[-test_set_size:]
            y_train, y_dev = y_shuffled[:-test_set_size], y_shuffled[-test_set_size:]
            print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

            # Training
            # ==================================================
            cnn = TextCNN(
                art_length=max_document_length,
                num_classes=FLAGS.num_of_classes,
                vocab_size=len(vocab_processor.vocabulary_),
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
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, tf.get_default_graph())

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, tf.get_default_graph())

            # Initialize all variables
            init_op = tf.initialize_all_variables()

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
               os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, "vocab"))

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

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=out_dir,
                                 init_op=init_op,
                                 summary_op=train_summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
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

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
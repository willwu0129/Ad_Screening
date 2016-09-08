# This is an interesting way of saving the data and feeding it directly to the model. BUt in the model, i don't use it.

import os
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(x_data, y_data, path, name):
    filename = os.path.join(path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    num_examples = len(x_data)
    for index in range(num_examples):
        temp_x = " ".join(str(elem) for elem in x_data[index])
        temp_y = " ".join(str(elem) for elem in y_data[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            'doc_vec': _bytes_feature(temp_x),
            'label_vec': _bytes_feature(temp_y)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    x_data = [[1, 2, 3], [1, 3, 5]]
    y_data = [[1, 0], [0, 1]]
    convert_to(x_data, y_data, os.getcwd(), "formal_save")
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), 'formal_save.tfrecords')])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'doc_vec': tf.FixedLenFeature([], tf.string),
            'label_vec': tf.FixedLenFeature([], tf.string),
        })
    print(features.get('doc_vec'))
    print("")





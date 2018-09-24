import numpy as np
import tensorflow as tf
from datetime import datetime
import time

tf.logging.set_verbosity(tf.logging.INFO)
fmt = '%H:%M:%S'


def get_current_time():
    time.ctime()
    return time.strftime(fmt)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=16, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


train_data = np.genfromtxt(fname='trainData.csv', delimiter=',', dtype=np.float32, skip_header=1)
train_labels = np.genfromtxt(fname='trainLabels.csv', dtype=np.int32, skip_header=1)
i = len(train_labels) - 1
while i >= 0:
    if train_labels[i] == 4: train_labels[i] = 0
    elif train_labels[i] == 8: train_labels[i] = 2
    i -= 1

# eval_data = np.genfromtxt(fname='kaggleTestSubset.csv', delimiter=',', dtype=np.float32, skip_header=1)
# eval_labels = np.genfromtxt(fname='kaggleTestSubsetLabels.csv', dtype=np.int32, skip_header=1)
# i = len(eval_labels) - 1
# while i >= 0:
#     if eval_labels[i] == 4: eval_labels[i] = 0
#     elif eval_labels[i] == 8: eval_labels[i] = 2
#     i -= 1

eval_data = np.genfromtxt(fname='testData.csv', delimiter=',', dtype=np.float32, skip_header=1)

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn)  # model_dir="/tmp/mnist_convnet_model")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=10,
    num_epochs=None,
    shuffle=True)
first = get_current_time()      # starting time of training
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=15000)
second = get_current_time()     # end time of training
print("Time taken to train(sec):", datetime.strptime(second, fmt) - datetime.strptime(first, fmt))

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    # y=eval_labels,
    num_epochs=1,
    shuffle=False)

second = get_current_time()     # start time of prediction
# eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
# third = get_current_time()      # end time of prediction
# print("Time taken to predict(sec):", datetime.strptime(third, fmt) - datetime.strptime(second, fmt))
# print(eval_results)

train_results = mnist_classifier.predict(input_fn=eval_input_fn)
third = get_current_time()      # end time of prediction

handle = open('result.csv', 'w')
handle.write('ID,Label\n')
i = 0
for x in train_results:
    i += 1
    if x['classes'] == 0: handle.write(str(i) + ',' + '4\n')
    elif x['classes'] == 2: handle.write(str(i) + ',' + '8\n')
    else: handle.write(str(i) + ',' + str(x['classes']) + '\n')
handle.close()

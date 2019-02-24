import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from sklearn.utils import shuffle
from import_data import import_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"



INPUT_NODE = 36

OUTPUT_NODE = 19

LAYER1_NODE = 10



def get_weight_variable(shape, regularizer):

    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):

    with tf.variable_scope('layer1'):



        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)

        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)



    with tf.variable_scope('layer2'):

        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)

        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + biases



    return layer2


def train(mnist):

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.next_batch()
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()



class image_data:
    def __init__(self, path):
        self.data = import_data.import_data(path)
        self.data = shuffle(self.data)
        self.num_examples = 10000
        # 选出前多少个样本点
        self.data=self.data[:-self.num_examples]
        self.y=self.data.pop('o_label')
        self.x=self.data
        self.i=1
        self.batch_size=100

    def next_batch(self):
        batch=self.x[(self.i-1)*self.batch_size:self.i*self.batch_size]
        batch_label=self.y[(self.i-1)*self.batch_size:self.i*self.batch_size]
        self.i+=1
        return batch,batch_label

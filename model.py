from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class L2AED:
    def __init__(self, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.lr = tf.placeholder(tf.float32)
        if FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.dim_hidden = FLAGS.num_filters
            self.forward = self.forward_conv
            self.construct_weights = self.construct_conv_weights
            self.construct_weights_attend = self.construct_attend_weights
            self.forward_attend = self.forward_attention_net
            self.merge = self.merge_by_channel
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
                self.featuremap_size = 5
            else:
                self.channels = 1
                self.featuremap_size = 3
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
            if FLAGS.K_shot == 1:
                self.attend_channels = FLAGS.num_classes
            else:
                self.attend_channels = FLAGS.K_shot
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                weights_attend = self.weights_attend
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                self.weights_attend = weights_attend = self.construct_weights_attend()

            outputbs = []
            lossesb = []
            accuraciesb = []

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp

                embedding_a = self.forward(inputa, weights, prefix, reuse=reuse)
                embedding_b = self.forward(inputb, weights, prefix, reuse=True)

                embedding_merge = []

                if FLAGS.K_shot == 1:
                    embedding_a_class = tf.transpose(embedding_a, [3,1,2,0])
                else:
                    embedding_a_class = tf.transpose(tf.gather(embedding_a, tf.range(FLAGS.K_shot)), [3,1,2,0])

                embedding_a_class_weights = self.forward_attend(embedding_a_class, weights_attend, prefix, reuse=reuse) 
                embedding_merge_class = self.merge(embedding_a_class, embedding_a_class_weights, self.attend_channels)
                embedding_merge.append(embedding_merge_class)

                for j in range(1,FLAGS.num_classes):
                    if FLAGS.K_shot == 1:
                        embedding_a_1, embedding_a_2 = tf.gather(embedding_a, [0]), tf.gather(embedding_a, tf.range(1,FLAGS.num_classes))
                        embedding_a = tf.concat([embedding_a_2, embedding_a_1], 0)
                        embedding_a_class = embedding_a
                    else:
                        embedding_a_class = tf.gather(embedding_a, tf.range(FLAGS.K_shot)+j*FLAGS.K_shot)

                    embedding_a_class = tf.transpose(embedding_a_class, [3,1,2,0])
                    embedding_a_class_weights = self.forward_attend(embedding_a_class, weights_attend, prefix, reuse=True)
                    embedding_a_class = self.merge(embedding_a_class, embedding_a_class_weights, self.attend_channels)
                    embedding_merge.append(embedding_a_class)

                embedding_merge = tf.convert_to_tensor(embedding_merge)


                if 'train' in prefix:
                    num_query_b = FLAGS.num_classes*FLAGS.num_query
                else:
                    num_query_b = FLAGS.num_classes*FLAGS.num_query_val

                # reshape the embeddings
                embedding_merge = tf.reshape(embedding_merge, [FLAGS.num_classes,-1])
                embedding_b = tf.reshape(embedding_b, [num_query_b,-1])

                distance_list = []
                for i in range(num_query_b):
                    embedding_test = tf.gather(embedding_b,[i])
                    embedding_test_expand = tf.tile(embedding_test, [self.dim_output,1])
                    distance = tf.norm(embedding_merge - embedding_test_expand, axis=-1)
                    distance_list.append(distance)

                task_outputbs = []


                # classification loss
                task_outputbs = tf.negative(distance_list)
                self.task_outputbs = task_outputbs
                task_lossesb = self.loss_func(task_outputbs, labelb)
                task_accuraciesb = []
                task_accuraciesb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs), 1), tf.argmax(labelb, 1))
                task_output = [task_lossesb]
                task_output.extend([task_accuraciesb])

                return task_output

            # to initialize the batch norm vars
            unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32]
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb, accuraciesb = result


        ## Performance & Optimization
        if 'train' in prefix:
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb) / tf.to_float(FLAGS.meta_batch_size)]
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb) / tf.to_float(FLAGS.meta_batch_size)]

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.gvs = gvs = optimizer.compute_gradients(tf.reshape(self.total_losses2,[]))
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb) / tf.to_float(FLAGS.meta_batch_size)]
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb) / tf.to_float(FLAGS.meta_batch_size)]

        ## Summaries
        # tf.summary.scalar(prefix+'classification loss', tf.reshape(total_losses2,[]))
        # tf.summary.scalar(prefix+'classification accuracy', tf.reshape(total_accuracies2,[]))


    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))

        return weights

    def construct_attend_weights(self):
        weights_attend = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3
        dim_hidden = 32

        weights_attend['attend_conv1'] = tf.get_variable('attend_conv1', [k, k, self.attend_channels, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights_attend['attend_b1'] = tf.Variable(tf.zeros([dim_hidden]))
        weights_attend['attend_conv2'] = tf.get_variable('attend_conv2', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights_attend['attend_b2'] = tf.Variable(tf.zeros([dim_hidden]))
        weights_attend['attend_w3'] = tf.get_variable('attend_w3', [self.featuremap_size*self.featuremap_size*dim_hidden, self.attend_channels], initializer=fc_initializer)
        weights_attend['attend_b3'] = tf.Variable(tf.zeros([self.attend_channels]), name='b3')

        return weights_attend


    def forward_conv(self, inp, weights, prefix, reuse=False, scope=''):
        if 'train' in prefix:
            dropout = True
        else:
            dropout = False
        if FLAGS.datasource == 'miniimagenet':
            last_pooling = True
        else:
            last_pooling = False

        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0', pooling=True)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1', pooling=True, drop=dropout)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2', pooling=True, drop=dropout)
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3', pooling=last_pooling)

        return hidden4

    def forward_attention_net(self, inp, weights_attend, prefix, reuse=False, scope=''):
        if 'train' in prefix:
            dropout = True
        else:
            dropout = False

        hidden1 = conv_block(inp, weights_attend['attend_conv1'], weights_attend['attend_b1'], reuse, scope+'4', pooling=False, drop=dropout) # [32,5,5,32]
        hidden2 = conv_block(hidden1, weights_attend['attend_conv2'], weights_attend['attend_b2'], reuse, scope+'5', pooling=False, drop=dropout) # [32,5,5,32]
        hidden2 = tf.reshape(hidden2, [-1, np.prod([int(dim) for dim in hidden2.get_shape()[1:]])]) 
        merge_weight = tf.matmul(hidden2, weights_attend['attend_w3']) + weights_attend['attend_b3']
        if FLAGS.K_shot != 1:
            merge_weight = tf.nn.softmax(merge_weight)

        return merge_weight

    def merge_by_channel(self, embedding, merge_weights, attend_channels):
        # merge embeddings based on attention weights
        merged_embedding = []
        for i in range(FLAGS.num_filters):
            embedding_channel = tf.reshape(tf.transpose(tf.gather(embedding, [i]), [0,3,1,2]), [attend_channels,-1])
            weights_channel = tf.gather(merge_weights, [i]) 
            merged_channel = tf.matmul(weights_channel, embedding_channel)
            merged_embedding.append(merged_channel)

        merged_embedding = tf.transpose(tf.reshape(merged_embedding, [FLAGS.num_filters,1,self.featuremap_size,self.featuremap_size]), [1,2,3,0])

        return merged_embedding

import csv
import numpy as np
# np.set_printoptions(precision=4,suppress=True,threshold=np.inf)
import pickle
import random
import tensorflow as tf


from data_generator import DataGenerator
from model import L2AED
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'miniimagenet', 'omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('metatrain_iterations', 70000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('K_shot', 5, 'number of training examples per class')
flags.DEFINE_integer('num_query', 15, 'number of query examples per class for meta-training (e.g. 15 for miniImageNet, 5 for omniglot)')
flags.DEFINE_integer('num_query_val', 15, 'number of query examples per class for meta-validation or meta-testing (e.g. 15 for miniImageNet, 5 for omniglot)')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_K_shot', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_integer('test_seed', 6, 'Control the sampling of a batch of testing tasks')

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    PRINT_INTERVAL = 100
    VAL_INTERVAL = 1000
    LR_CUT_STEP = 20000

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    losses, accuracies = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    best_accuracy = 0

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        lr_cut = FLAGS.lr / (2**(itr // LR_CUT_STEP))
        feed_dict = {model.lr: lr_cut}

        input_tensors = [model.metatrain_op]
        if itr % SUMMARY_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_losses2, model.total_accuracies2])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            losses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            accuracies.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(losses)) + ', ' + str(np.mean(accuracies))
            print(print_str)
            losses, accuracies = [], []


        if (itr!=0) and itr % VAL_INTERVAL == 0:
            metaval_accuracies = []
            for _ in range(600):
                feed_dict = {}
                input_tensors = [model.metaval_total_accuracies2]
                result = sess.run(input_tensors, feed_dict)
                metaval_accuracies.append(result)
            metaval_accuracies = np.array(metaval_accuracies)
            means = np.mean(metaval_accuracies, 0)

            if means > best_accuracy:
                saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr), write_meta_graph = False)
                best_accuracy = means
            # elif itr % 10000 ==0:
            #     saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr), write_meta_graph = False)

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

def test(model, saver, sess, exp_string, data_generator):
    if FLAGS.datasource == 'omniglot':
        NUM_TEST_POINTS = 1000
    else:
        NUM_TEST_POINTS = 600

    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)
    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):

        feed_dict = {}
        feed_dict = {model.lr : 0.0}

        result = sess.run([model.metaval_total_accuracies2], feed_dict)
        metaval_accuracies.append(result)


    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))



def main():
    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
        assert FLAGS.meta_batch_size == 1
        assert FLAGS.K_shot == 1
        data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
    else:
        if FLAGS.train:
            data_generator = DataGenerator(FLAGS.K_shot+FLAGS.num_query, FLAGS.meta_batch_size)
        else:
            data_generator = DataGenerator(FLAGS.K_shot+FLAGS.num_query_val, FLAGS.meta_batch_size)

    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    tf_data_load = True
    num_classes = data_generator.num_classes

    if FLAGS.train: 
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        inputa, labela, inputb, labelb = [],[],[],[]


        for i in range(FLAGS.num_classes):
            inputa.append(tf.slice(image_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query),0], [-1,FLAGS.K_shot,-1]))
            labela.append(tf.slice(label_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query),0], [-1,FLAGS.K_shot,-1]))
            inputb.append(tf.slice(image_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query)+FLAGS.K_shot,0], [-1,FLAGS.num_query,-1]))
            labelb.append(tf.slice(label_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query)+FLAGS.K_shot,0], [-1,FLAGS.num_query,-1]))
        inputa = tf.concat(inputa, 1)
        labela = tf.concat(labela, 1)
        inputb = tf.concat(inputb, 1)
        labelb = tf.concat(labelb, 1)
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    random.seed(FLAGS.test_seed)
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    inputa, labela, inputb, labelb = [],[],[],[]
    for i in range(FLAGS.num_classes):
            inputa.append(tf.slice(image_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query_val),0], [-1,FLAGS.K_shot,-1]))
            labela.append(tf.slice(label_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query_val),0], [-1,FLAGS.K_shot,-1]))
            inputb.append(tf.slice(image_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query_val)+FLAGS.K_shot,0], [-1,FLAGS.num_query_val,-1]))
            labelb.append(tf.slice(label_tensor, [0,i*(FLAGS.K_shot+FLAGS.num_query_val)+FLAGS.K_shot,0], [-1,FLAGS.num_query_val,-1]))
    inputa = tf.concat(inputa, 1)
    labela = tf.concat(labela, 1)
    inputb = tf.concat(inputb, 1)
    labelb = tf.concat(labelb, 1)
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = L2AED(dim_input, dim_output)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_K_shot == -1:
        FLAGS.train_K_shot = FLAGS.K_shot

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_K_shot)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)


    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator)

if __name__ == "__main__":
    main()

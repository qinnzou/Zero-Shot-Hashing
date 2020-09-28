import os
import tensorflow as tf
import numpy as np
from resnet import *
import random
import math
import scipy.io as sio
from evaluation import *
import argparse
from loss import *
from PIL import Image
from utils import *
from evaluation_metrics import *
import time

parser = argparse.ArgumentParser()

parser.add_argument("--train_unseen_lst", default="./data/nus/train_imglst.txt")
parser.add_argument("--test_unseen_lst", default="./data/nus/query_imglst.txt")
parser.add_argument("--valid_unseen_lst", default="./data/nus/database_imglst.txt")
parser.add_argument("--train_unseen_label", default="./data/nus/train_labels.npy")
parser.add_argument("--test_unseen_label", default="./data/nus/query_labels.npy")
parser.add_argument("--valid_unseen_label", default="./data/nus/database_labels.npy")

parser.add_argument("--train_seen_lst", default="./data/voc2012/train.txt")
parser.add_argument("--test_seen_lst", default="./data/voc2012/test.txt")
parser.add_argument("--valid_seen_lst", default="./data/voc2012/validation.txt")
parser.add_argument("--train_seen_label", default="./data/voc2012/voc2012_train_label.npy")
parser.add_argument("--test_seen_label", default="./data/voc2012/voc2012_test_label.npy")
parser.add_argument("--valid_seen_label", default="./data/voc2012/voc2012_valid_label.npy")

parser.add_argument("--unseen_img_dir", default="/dataset/NUS-WIDE/all_images/")
parser.add_argument("--seen_img_dir", default="/dataset/voc2012/im256/")

parser.add_argument("--unseen_word2vec", default="./data/word2vec/nus18_word2vec_300d.npy")
parser.add_argument("--seen_word2vec", default="./data/word2vec/pascal17_word2vec_300d.npy")

parser.add_argument("--num_epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")
parser.add_argument("--hash_bits", type=int, default=36, help="number of hash bits")
parser.add_argument("--nb_labels_seen", type=int, default=17, help="number of seen calss")
parser.add_argument("--nb_labels_unseen", type=int, default=18, help="number of unseen calss")

parser.add_argument("--keep_prob", type=float, default=0.5, help="dropout rate")
parser.add_argument("--alpha", type=float, default=1, help="weight on regularizer term")
parser.add_argument("--belta", type=float, default=1., help="threshold to limit the range of value")
parser.add_argument("--gama", type=float, default=10., help="weight on pairwise similar or dissimilar")
parser.add_argument("--img_size", type=int, default=224, help="image size of input")
parser.add_argument("--delen", type=int, default=300, help="feature length in shared space")

parser.add_argument("--visual_lr", type=float, default=1e-6, help="initial learning rate for adam")
parser.add_argument("--hash_lr", type=float, default=1e-3, help="initial learning rate for adam")

parser.add_argument("--decay_step", type=int, default=500, help="number of steps to dacay lreaning rate")
parser.add_argument("--decay_rate", type=float, default=0.5, help="decaying rate")

parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--output_dir", default='./models/resnet50-12b-nus-unseen-top3-300d', help="where to put output files")
parser.add_argument("--log", default='./log/resnet50-24b-nus-unseen-top3-300d-log.txt')
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--resnet_depth", type=int, default=50)
parser.add_argument("--skip_layer", default="fc")
parser.add_argument("--device", default="7")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mean_value = np.array([123, 117, 104]).reshape((1, 3))
skip_layer = ['fc8', 'visual_fc1', 'semantic_fc1']


def print_to_file(txtfile, content):
    with open(txtfile, 'a') as f:
        f.write(content + '\n')
        print(content)


def exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=False):
    with tf.name_scope("exponential_decay_with_warmup"):
        linear_increase=learning_rate_base*tf.cast(global_step/warmup_step,tf.float32)
        exponential_decay=tf.train.exponential_decay(learning_rate_base,
                                                     global_step-warmup_step,
                                                     learning_rate_step,
                                                     learning_rate_decay,
                                                     staircase=staircase)
        learning_rate=tf.cond(global_step<=warmup_step,lambda:linear_increase,lambda:exponential_decay)
        return learning_rate




if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_seen_x, train_seen_y = data_load(args.train_seen_lst, args.train_seen_label, args.img_size, args.seen_img_dir,
                                           mean_value)
    test_seen_x, test_seen_y = data_load(args.test_seen_lst, args.test_seen_label, args.img_size, args.seen_img_dir,
                                           mean_value)
    valid_seen_x, valid_seen_y = data_load(args.valid_seen_lst, args.valid_seen_label, args.img_size, args.seen_img_dir,
                                           mean_value)

    train_seen_y_all = np.hstack((train_seen_y, np.zeros(dtype=np.float32, shape=[train_seen_y.shape[0], args.nb_labels_unseen])))
    test_seen_y_all = np.hstack((test_seen_y, np.zeros(dtype=np.float32, shape=[test_seen_y.shape[0],  args.nb_labels_unseen])))
    valid_seen_y_all = np.hstack((valid_seen_y, np.zeros(dtype=np.float32, shape=[valid_seen_y.shape[0],  args.nb_labels_unseen])))

    train_unseen_x, train_unseen_y = data_load(args.train_unseen_lst, args.train_unseen_label, args.img_size, args.unseen_img_dir,
                                           mean_value)
    test_unseen_x, test_unseen_y = data_load(args.test_unseen_lst, args.test_unseen_label, args.img_size, args.unseen_img_dir,
                                           mean_value)
    valid_unseen_x, valid_unseen_y = data_load(args.valid_unseen_lst, args.valid_unseen_label, args.img_size, args.unseen_img_dir,
                                           mean_value)

    train_unseen_y_all = np.hstack((np.zeros(dtype=np.float32, shape=[train_unseen_y.shape[0], args.nb_labels_seen]), train_unseen_y))
    test_unseen_y_all = np.hstack((np.zeros(dtype=np.float32, shape=[test_unseen_y.shape[0],  args.nb_labels_seen]), test_unseen_y))
    valid_unseen_y_all = np.hstack((np.zeros(dtype=np.float32, shape=[valid_unseen_y.shape[0],  args.nb_labels_seen]), valid_unseen_y))

    nb_train_seen = train_seen_x.shape[0]
    batch_size = args.batch_size
    nb_labels_seen = args.nb_labels_seen  # 17
    nb_labels_unseen = args.nb_labels_unseen  # 18

    semantic_mat_seen = np.load(args.seen_word2vec)
    semantic_mat_unseen = np.load(args.unseen_word2vec)
    dim_vector = semantic_mat_seen.shape[1]
    semantic_mat = np.concatenate((semantic_mat_seen, semantic_mat_unseen), axis=0)
    semantic_mat = semantic_mat / np.tile(
        np.sqrt(np.sum(np.square(semantic_mat), axis=1, keepdims=True)), [1, dim_vector])

    sementic_batch_size = 6
    highest_map = 0
    top_nums = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    semantic_embedding_updated = semantic_mat_seen
    semantic_embedding_updated = semantic_embedding_updated / np.tile(
        np.sqrt(np.sum(np.square(semantic_embedding_updated), axis=1, keepdims=True)), [1, dim_vector])
    # semantic_embedding_updated = np.random.randn(nb_labels_seen, delen)
    visual_embedding_updated = np.random.randn(nb_train_seen, args.delen)
    # visual_embedding_updated = np.zeros(dtype=np.float32, shape=(nb_train_seen, delen))

    global_step1 = tf.Variable(0, name="global_step1", trainable=False)
    global_step2 = tf.Variable(0, name="global_step2", trainable=False)

    is_training = tf.placeholder(tf.bool, shape=())
    dropout_rate = tf.cond(is_training, lambda: 0.5, lambda: 1.0)
    # dropout_rate = tf.cond(is_training, lambda: 0.8, lambda: 1.0)
    input_image = tf.placeholder(dtype=tf.float32, shape=(None, args.img_size, args.img_size, 3))
    input_labels = tf.placeholder(dtype=tf.float32, shape=(None, None), name='train_labels')
    input_vectors = tf.placeholder(dtype=tf.float32, shape=(None, dim_vector), name='seen_word_vector')
    batch_input_labels = tf.placeholder(dtype=tf.float32, shape=(None, sementic_batch_size), name='batch_train_labels')

    semantic_embedding = tf.placeholder(dtype=tf.float32, shape=(nb_labels_seen, args.delen), name='semantic_embedding')
    decay_step = tf.placeholder(dtype=tf.int32,shape=())

    steps = int(math.floor(nb_train_seen / args.batch_size))

    train_fuse_x = np.vstack((train_seen_x, train_unseen_x))
    steps3 = int(math.floor(train_fuse_x.shape[0] / batch_size))


    model = ResNetModel(input_image, is_training, depth=args.resnet_depth, num_classes=args.hash_bits)
    hash_layer = model.prob
    with tf.variable_scope('visual_fc1'):
        visual_fc1 = fc(hash_layer, args.delen)

    

    c_matrix1 = tf.matmul(visual_fc1, semantic_embedding, transpose_a=False, transpose_b=True)

    regularizer_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    visual_rank_loss = ranking_loss(c_matrix1, input_labels) + regularizer_loss
    hash_loss = hash_focal_loss(hash_layer, input_labels, args.hash_bits, args.alpha)
   
    # List of trainable variables of the layers we want to train
    var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in skip_layer]
    var_list1 = [v for v in tf.trainable_variables() if v not in var_list2]


    ema = tf.train.ExponentialMovingAverage(0.997)
    tf.add_to_collection('resnet_update_ops1', ema.apply([hash_loss]))
    batchnorm_updates1 = tf.get_collection('resnet_update_ops1')




    # learning rate
    hash_lr = tf.train.exponential_decay(args.hash_lr, global_step2, decay_step, args.decay_rate, staircase=True)

    hash_opt1 = tf.train.AdamOptimizer(hash_lr * 0.01)
    hash_opt2 = tf.train.AdamOptimizer(hash_lr)
   # hash_opt3 = tf.train.AdamOptimizer(hash_lr * 2)

    # apply different grads for two type layers
    hash_grads = tf.gradients(hash_loss, var_list1 + var_list2)
    hash_grads1 = hash_grads[:len(var_list1)]
    hash_grads2 = hash_grads[len(var_list1):]
    hash_train_op1 = hash_opt1.apply_gradients(zip(hash_grads1, var_list1))
    hash_train_op1 = tf.group(*(batchnorm_updates1 + [hash_train_op1]))
    hash_train_op2 = hash_opt2.apply_gradients(zip(hash_grads2, var_list2), global_step=global_step2)
    hash_train_op2 = tf.group(*(batchnorm_updates1 + [hash_train_op2]))
  
    with tf.control_dependencies(batchnorm_updates1):
        hash_train_op = tf.group(hash_train_op1, hash_train_op2)
    visual_var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in ['visual_fc1']]
    visual_var_list1 = [v for v in tf.trainable_variables() if v not in visual_var_list2]


    ema = tf.train.ExponentialMovingAverage(0.997)
    tf.add_to_collection('resnet_update_ops', ema.apply([visual_rank_loss]))
    batchnorm_updates = tf.get_collection('resnet_update_ops')
  
    # learning rate
    visual_lr = tf.train.exponential_decay(args.visual_lr, global_step1, decay_step, args.decay_rate, staircase=True)

    visual_opt1 = tf.train.AdamOptimizer(visual_lr * 0.0)
    visual_opt2 = tf.train.AdamOptimizer(visual_lr)
    

    # apply different grads for two type layers
    visual_grads = tf.gradients(visual_rank_loss, visual_var_list1 + visual_var_list2)
    visual_grads1 = visual_grads[:len(visual_var_list1)]
    visual_grads2 = visual_grads[len(visual_var_list1):]
    visual_train_op1 = visual_opt1.apply_gradients(zip(visual_grads1, visual_var_list1))
    visual_train_op1 = tf.group(*(batchnorm_updates + [visual_train_op1]))
    visual_train_op2 = visual_opt2.apply_gradients(zip(visual_grads2, visual_var_list2), global_step=global_step1)
    visual_train_op2 = tf.group(*(batchnorm_updates + [visual_train_op2]))
    

   # visual_train_op = tf.group(visual_train_op1, visual_train_op2,batchnorm_updates_op)
    with tf.control_dependencies(batchnorm_updates):
        visual_train_op = tf.group(visual_train_op1, visual_train_op2)

    with tf.Session(config=config) as sess:

        saver = tf.train.Saver(tf.global_variables())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if args.checkpoint is not None:
            ckpt = tf.train.get_checkpoint_state(args.checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                checkpoint = tf.train.latest_checkpoint(args.checkpoint)
                print('Restoring model from {}'.format(checkpoint))
                saver.restore(sess, checkpoint)
            else:
                model.load_original_weights(sess, skip_layers=skip_layer)
               
                print('1.Using pre-trained model')
        else:
            # Load the pretrained weights into the non-trainable layer
            model.load_original_weights(sess, skip_layers=skip_layer)
            
            print('2.Using pre-trained model')

        hashcode_unseen_test = np.zeros([test_unseen_x.shape[0], args.hash_bits], np.float32)
        begin_inds = 0
        end_inds = args.batch_size
        while end_inds < test_unseen_x.shape[0]:
            tmp_hashcode = sess.run(hash_layer, feed_dict={input_image: test_unseen_x[begin_inds:end_inds, :, :, :], is_training:False})
            hashcode_unseen_test[begin_inds:end_inds, :] = tmp_hashcode
            begin_inds = begin_inds + args.batch_size
            end_inds = end_inds + args.batch_size
        hashcode_unseen_test[begin_inds:test_unseen_x.shape[0], :] = sess.run(hash_layer,
                    feed_dict={input_image: test_unseen_x[begin_inds:test_unseen_x.shape[0], :, :, :], is_training:False})
        hashcode_unseen_test = np.sign(hashcode_unseen_test)


        hashcode_unseen_val = np.zeros([valid_unseen_x.shape[0], args.hash_bits], np.float32)
        begin_inds = 0
        end_inds = args.batch_size
        while end_inds < valid_unseen_x.shape[0]:
            tmp_hashcode = sess.run(hash_layer, feed_dict={input_image: valid_unseen_x[begin_inds:end_inds, :, :, :], is_training:False})
            hashcode_unseen_val[begin_inds:end_inds, :] = tmp_hashcode
            begin_inds = begin_inds + args.batch_size
            end_inds = end_inds + args.batch_size
        hashcode_unseen_val[begin_inds:valid_unseen_x.shape[0], :] = sess.run(hash_layer,
                        feed_dict={input_image: valid_unseen_x[begin_inds:valid_unseen_x.shape[0], :, :, :], is_training:False})
        hashcode_unseen_val = np.sign(hashcode_unseen_val)

        St1 = np.dot(test_unseen_y, np.transpose(valid_unseen_y))
        Wt1 = np.float32(St1 > 0)
        [p1, r1, map1, wap1, acg1, ndcg1] = evaluation(hashcode_unseen_test, hashcode_unseen_val, Wt1, St1, top_nums)

        print_to_file(args.log, 'Retrieval evaluation on unseen dataset:')
        for i in range(len(top_nums)):
            tmp_str = ('top_%d, precission: %f, recall: %f, map: %f, wap: %f, acg: %f, ndcg: %f' %
                       (top_nums[i], p1[i], r1[i], map1[i], wap1[i], acg1[i], ndcg1[i]))
            print_to_file(args.log, tmp_str)
        print_to_file(args.log, '\n')
        maxMAP= map1[-1]
        steps = int(math.floor(nb_train_seen / args.batch_size))

        train_fuse_x = np.vstack((train_seen_x, train_unseen_x))
        steps3 = int(math.floor(train_fuse_x.shape[0] / batch_size))

        for i_epoch in range(args.num_epochs):
            t1=time.clock()

            rand_idx = list(range(nb_train_seen))
            random.shuffle(rand_idx)

            rand_idx2 = list(range(nb_train_seen))
            random.shuffle(rand_idx2)

            for i in range(steps):
                batch_x = train_seen_x[rand_idx2[i * batch_size:(i + 1) * batch_size], :, :, :]
                batch_y = train_seen_y[rand_idx2[i * batch_size:(i + 1) * batch_size], :]
                _, v_loss, m1, g_step1 = sess.run([visual_train_op, visual_rank_loss, c_matrix1, global_step1],
                                              feed_dict={input_image: batch_x,
                                                         decay_step: steps*5,
                                                         semantic_embedding: semantic_embedding_updated,
                                                         input_labels: batch_y, is_training:True})
                if g_step1 % 10 == 0:
                    print('[%d/%d], visual ranking loss: %f' % (g_step1, steps, v_loss))


            visual_embedding_unseen_train = np.zeros([train_unseen_x.shape[0], args.delen], np.float32)
            begin_inds = 0
            end_inds = args.batch_size
            while end_inds < train_unseen_x.shape[0]:
                tmp_hashcode = sess.run(visual_fc1,
                                        feed_dict={input_image: train_unseen_x[begin_inds:end_inds, :, :, :], is_training:False})
                visual_embedding_unseen_train[begin_inds:end_inds, :] = tmp_hashcode
                begin_inds = begin_inds + args.batch_size
                end_inds = end_inds + args.batch_size
            visual_embedding_unseen_train[begin_inds:train_unseen_x.shape[0], :] = sess.run(visual_fc1,
                                feed_dict={input_image: train_unseen_x[begin_inds:train_unseen_x.shape[0], :, :, :], is_training:False})

            confidence_train= np.dot(visual_embedding_unseen_train, np.transpose(semantic_mat))

            k = args.topk
            sort_indices = np.argsort(confidence_train)
            sort_indices = sort_indices[:, ::-1]
            top_k_indices = sort_indices[:, 0:k]
            pred_train_unseen_y = np.zeros(shape=[train_unseen_y.shape[0], args.nb_labels_seen+args.nb_labels_unseen], dtype=np.float32)
            for i in range(train_unseen_y.shape[0]):
                pred_train_unseen_y[i,top_k_indices[i]] = 1

            # pred_train_unseen_y[:, 0:nb_labels_seen] = 0
            train_fuse_y = np.vstack((train_seen_y_all, pred_train_unseen_y))
            # train_fuse_y = np.vstack((train_seen_y_all, train_unseen_y_all))

            rand_idx3 = list(range(train_fuse_x.shape[0]))
            random.shuffle(rand_idx3)

            for i in range(steps3):
                batch_x = train_fuse_x[rand_idx3[i * batch_size:(i + 1) * batch_size], :, :, :]
                batch_y = train_fuse_y[rand_idx3[i * batch_size:(i + 1) * batch_size], :]
                _, h_loss, g_step2 = sess.run([hash_train_op, hash_loss, global_step2],
                                              feed_dict={input_image: batch_x,
                                                         input_labels: batch_y,
                                                         decay_step: steps3 * 5,
                                                         is_training:True})
                if i % 10 == 0:
                    print('[%d/%d], hash loss: %f' % (g_step2, steps3, h_loss))

            t2=time.clock()
            t=t2-t1
            tstr=('hashbits_%d, iter: %f, time: %f,' %
                        (args.hash_bits,i_epoch,t))
            # print_to_file('time.txt',  args.hash_bits)
            print_to_file('time.txt',tstr)
            hashcode_unseen_test = np.zeros([test_unseen_x.shape[0], args.hash_bits], np.float32)
            begin_inds = 0
            end_inds = args.batch_size
            while end_inds < test_unseen_x.shape[0]:
                tmp_hashcode = sess.run(hash_layer, feed_dict={input_image: test_unseen_x[begin_inds:end_inds, :, :, :],
                                                               is_training: False})
                hashcode_unseen_test[begin_inds:end_inds, :] = tmp_hashcode
                begin_inds = begin_inds + args.batch_size
                end_inds = end_inds + args.batch_size
            hashcode_unseen_test[begin_inds:test_unseen_x.shape[0], :] = sess.run(hash_layer,
                                                                                  feed_dict={input_image: test_unseen_x[
                                                                                                          begin_inds:
                                                                                                          test_unseen_x.shape[
                                                                                                              0], :, :,
                                                                                                          :],
                                                                                             is_training: False})
            hashcode_unseen_test = np.sign(hashcode_unseen_test)

           
            hashcode_unseen_val = np.zeros([valid_unseen_x.shape[0], args.hash_bits], np.float32)
            begin_inds = 0
            end_inds = args.batch_size
            while end_inds < valid_unseen_x.shape[0]:
                tmp_hashcode = sess.run(hash_layer,
                                        feed_dict={input_image: valid_unseen_x[begin_inds:end_inds, :, :, :], is_training:False})
                hashcode_unseen_val[begin_inds:end_inds, :] = tmp_hashcode
                begin_inds = begin_inds + args.batch_size
                end_inds = end_inds + args.batch_size
            hashcode_unseen_val[begin_inds:valid_unseen_x.shape[0], :] = sess.run(hash_layer,
                                feed_dict={input_image: valid_unseen_x[begin_inds:valid_unseen_x.shape[0], :, :, :], is_training:False})
            hashcode_unseen_val = np.sign(hashcode_unseen_val)

            St1 = np.dot(test_unseen_y, np.transpose(valid_unseen_y))
            Wt1 = np.float32(St1 > 0)
            [p1, r1, map1, wap1, acg1, ndcg1] = evaluation(hashcode_unseen_test, hashcode_unseen_val, Wt1, St1,
                                                           top_nums)
            print_to_file(args.log, 'Retrieval evaluation on unseen dataset:')
            for i in range(len(top_nums)):
                tmp_str = ('top_%d, precission: %f, recall: %f, map: %f, wap: %f, acg: %f, ndcg: %f' %
                      (top_nums[i], p1[i], r1[i], map1[i], wap1[i], acg1[i], ndcg1[i]))
                print_to_file(args.log, tmp_str)

            hashcode_seen_test = np.zeros([test_seen_x.shape[0], args.hash_bits], np.float32)
            begin_inds = 0
            end_inds = args.batch_size
            while end_inds < test_seen_x.shape[0]:
                tmp_hashcode = sess.run(hash_layer,
                                        feed_dict={input_image: test_seen_x[begin_inds:end_inds, :, :, :], is_training:False})
                hashcode_seen_test[begin_inds:end_inds, :] = tmp_hashcode
                begin_inds = begin_inds + args.batch_size
                end_inds = end_inds + args.batch_size
            hashcode_seen_test[begin_inds:test_seen_x.shape[0], :] = sess.run(hash_layer,
                            feed_dict={input_image: test_seen_x[begin_inds:test_seen_x.shape[0], :, :, :], is_training:False})
            hashcode_seen_test = np.sign(hashcode_seen_test)

            hashcode_seen_val = np.zeros([valid_seen_x.shape[0], args.hash_bits], np.float32)
            begin_inds = 0
            end_inds = args.batch_size
            while end_inds < valid_seen_x.shape[0]:
                tmp_hashcode = sess.run(hash_layer,
                                        feed_dict={input_image: valid_seen_x[begin_inds:end_inds, :, :, :], is_training:False})
                hashcode_seen_val[begin_inds:end_inds, :] = tmp_hashcode
                begin_inds = begin_inds + args.batch_size
                end_inds = end_inds + args.batch_size
            hashcode_seen_val[begin_inds:valid_seen_x.shape[0], :] = sess.run(hash_layer,
                                feed_dict={input_image: valid_seen_x[begin_inds:valid_seen_x.shape[0], :, :, :],is_training:False})
            hashcode_seen_val = np.sign(hashcode_seen_val)

            St2 = np.dot(test_seen_y, np.transpose(valid_seen_y))
            Wt2 = np.float32(St2 > 0)
            [p2, r2, map2, wap2, acg2, ndcg2] = evaluation(hashcode_seen_test, hashcode_seen_val, Wt2, St2, top_nums)

            print_to_file(args.log,'Retrieval evaluation on seen dataset:')
            for i in range(len(top_nums)):
                tmp_str = ('top_%d, precission: %f, recall: %f, map: %f, wap: %f, acg: %f, ndcg: %f' %
                      (top_nums[i], p2[i], r2[i], map2[i], wap2[i], acg2[i], ndcg2[i]))
                print_to_file(args.log, tmp_str)
            print_to_file(args.log, '\n')

            if(map1[-1]>maxMAP):
                maxMAP=map1[-1]
                saver.save(sess, args.output_dir + '/mlzsh_top'+str(args.topk) + '_epoch' + str(i_epoch+1))

       

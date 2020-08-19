import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
import time
import GLAM
import h5py
from tqdm import tqdm

batch_size = 64
lr = 0.0001
lr_decay = 0.1
lr_decay_epoch = 5
epoch = 10
class_num = 188
display_step = 3000
content_clip_num = 6
motion_clip_num = 6
hidden_size = 1024
weight_decay=0.0
lstm_layer = 1
keep_prob_ratio = 1.0
lstm_keep_prob_ratio = 0.5
mean = False
batch_num = 250
r = 4
fusion_size = 1024
num = 3
# seed = 1

ckpt_path = 'ha_tsfn'
model_path = 'ha_tsfn_attention_num'.format(num)
model_name = 'final_model'

print('\033[0;31;40m')
print('batch_size:{}'.format(batch_size))
print('initial lr:{}'.format(lr))
print('lr_decay:{} lr_decay_epoch:{}'.format(lr_decay, lr_decay_epoch))
print('epoch:{}'.format(epoch))
print('weight_decay:{}'.format(weight_decay))
print('hidden_size:{} lstm_layer:{}'.format(hidden_size, lstm_layer))
print('keep_prob:{} lstm_keep_prob:{}'.format(keep_prob_ratio, lstm_keep_prob_ratio))
print('mean:{}'.format(mean))
print('fusion_size:{}'.format(fusion_size))
print('num:{}'.format(num))
print('\033[0m')

vine_content_feature_path = '/home/ycz/data_feature/content_feature_map_h5'
vine_motion_feature_path = '/home/ycz/data_feature/motion_feature_h5'

if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

if not os.path.exists(os.path.join(ckpt_path, model_path)):
    os.mkdir(os.path.join(ckpt_path, model_path))

print(os.path.join(ckpt_path, model_path))


def get_file_list(feature_dir, feature_str):
    file_list = []
    for file in os.listdir(os.path.join(feature_dir, feature_str)):
        file_list.append(os.path.join(feature_dir, '{}/{}'.format(feature_str, file)))
    file_list = sorted(file_list, key=lambda d: int(d.split('_')[-1].split('.')[0]))
    return file_list


def GA(x, num):
    alpha = tf.get_variable(name='alpha_{}'.format(num), shape=[1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0))
    reuse = False
    global_feature_map = []
    for i in range(content_clip_num):
        clip_x = x[:, i, :, :, :]
        global_feature_map_output = GLAM.global_attention(clip_x, name='global_attention_map_{}'.format(num),
                                                              squash=tf.nn.sigmoid, reuse=reuse, alpha=alpha)
        reuse = True
        global_feature_map.append(global_feature_map_output)
    global_feature_map = tf.convert_to_tensor(global_feature_map)
    global_feature_map = tf.transpose(global_feature_map, [1, 0, 2, 3, 4])

    return global_feature_map


def LA(x, num):
    beta = tf.get_variable(name='beta_{}'.format(num), shape=[1], dtype=tf.float32,
                           initializer=tf.constant_initializer(0))
    reuse = False
    local_feature_map = []
    for i in range(content_clip_num):
        clip_x = x[:, i, :, :, :]
        local_feature_map_output = GLAM.local_attention(clip_x, name='local_attention_map_{}'.format(num),
                                                               squash=tf.nn.softmax, reuse=reuse,
                                                        intermediate_kernel=kernel_size, r=r, beta=beta)
        reuse = True
        local_feature_map.append(local_feature_map_output)
    local_feature_map = tf.convert_to_tensor(local_feature_map)
    local_feature_map = tf.transpose(local_feature_map, [1, 0, 2, 3, 4])

    return local_feature_map


def train():
    c_train_file_list = get_file_list(vine_content_feature_path, 'train')
    c_val_file_list = get_file_list(vine_content_feature_path, 'valid')
    c_test_file_list = get_file_list(vine_content_feature_path, 'test')

    m_train_file_list = get_file_list(vine_motion_feature_path, 'train')
    m_val_file_list = get_file_list(vine_motion_feature_path, 'valid')
    m_test_file_list = get_file_list(vine_motion_feature_path, 'test')

    train_step = 3378
    val_step = 422
    test_step = 422

    print('train_step:{} val_step:{} test_step:{}'.format(train_step, val_step, test_step))

    # tf.set_random_seed(seed)
    content_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, content_clip_num, 2048, 7, 7], name='input')
    t_x = tf.transpose(content_x, (0, 1, 3, 4, 2))
    motion_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, motion_clip_num, 2048], name='motion_x')

    print(t_x.shape)
    y = tf.placeholder(dtype=tf.int64, shape=[None], name='y_')

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    lstm_keep_prob = tf.placeholder(dtype=tf.float32, name='lstm_keep_prob')

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.Variable(lr, trainable=False)

    def use_lstm(lstm_name):
        with tf.variable_scope(lstm_name):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=lstm_keep_prob)
        return lstm_cell

    def use_convlstm(convlstm_name):
        with tf.variable_scope(convlstm_name):
            convlstm_cell = tf.contrib.rnn.ConvLSTMCell(2, [7, 7, 2048], hidden_size, [kernel_size, kernel_size])
            convlstm_cell = tf.nn.rnn_cell.DropoutWrapper(convlstm_cell, input_keep_prob=lstm_keep_prob)
        return convlstm_cell

    global_feature_map = t_x
    for j in range(num):
        global_feature_map = GA(global_feature_map, j)

    local_feature_map = t_x
    for j in range(num):
        local_feature_map = LA(local_feature_map, j)

    concat_x = tf.concat([global_feature_map, local_feature_map], axis=-1)

    fusion_x_list = []
    reuse = False
    for i in range(content_clip_num):
        fusion_x = tf.layers.conv2d(inputs=concat_x[:, i, :, :, :], filters=2048, kernel_size=1, strides=1,
                                    padding='SAME', use_bias=True, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    reuse=reuse)
        fusion_x_list.append(fusion_x)
        reuse = True
    fusion_x_list = tf.convert_to_tensor(fusion_x_list)
    fusion_x_list = tf.transpose(fusion_x_list, [1, 0, 2, 3, 4])

    if lstm_layer > 0:
        c_convlstm_cell = tf.nn.rnn_cell.MultiRNNCell([use_convlstm('c_lstm_{}'.format(i)) for i in range(lstm_layer)])
        m_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([use_lstm('m_lstm_{}'.format(i)) for i in range(lstm_layer)])

    c_initial_state = c_convlstm_cell.zero_state(batch_size, dtype=tf.float32)
    c_output, c_state = tf.nn.dynamic_rnn(c_convlstm_cell, fusion_x_list, initial_state=c_initial_state)

    m_initial_state = m_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    m_output, m_state = tf.nn.dynamic_rnn(m_lstm_cell, motion_x, initial_state=m_initial_state)

    if not mean:
        c_output = c_output[:, -1, :, :, :]
        m_output = m_output[:, -1, :]
    else:
        c_output = tf.reduce_mean(c_output, axis=1)
        m_output = tf.reduce_mean(m_output, axis=1)

    # spatial average pooling
    c_output = tf.reduce_mean(c_output, reduction_indices=[1, 2])

    concat_x = tf.concat([c_output, m_output], axis=1)

    output = tf.layers.dense(inputs=concat_x, units=fusion_size, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.1))

    # classify
    logits = tf.layers.dense(inputs=output, units=class_num, activation=None, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.1))

    prob = tf.nn.softmax(logits)

    for var in tf.trainable_variables():
        if 'kernel' in var.name or 'weights' in var.name:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)

    print(tf.get_collection(tf.GraphKeys.WEIGHTS))

    l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
    reg_loss = tf.contrib.layers.apply_regularization(l2_reg)

    task_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    total_loss = task_loss + reg_loss

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob, axis=1), y), tf.float32))

    update = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=15)

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        print(tf.trainable_variables())

        train_res = np.zeros(shape=[epoch, 3])
        val_res = np.zeros(shape=[epoch, 3])
        test_res = np.zeros(shape=[epoch, 3])

        for e in range(epoch):
            print('\033[0;31;40m')
            print('batch_size:{}'.format(batch_size))
            print('initial lr:{}'.format(lr))
            print('lr_decay:{} lr_decay_epoch:{}'.format(lr_decay, lr_decay_epoch))
            print('epoch:{}'.format(epoch))
            print('weight_decay:{}'.format(weight_decay))
            print('hidden_size:{} lstm_layer:{}'.format(hidden_size, lstm_layer))
            print('keep_prob:{} lstm_keep_prob:{}'.format(keep_prob_ratio, lstm_keep_prob_ratio))
            print('mean:{}'.format(mean))
            print('fusion_size:{}'.format(fusion_size))
            print('num:{}'.format(num))
            print('\033[0m')

            print("learning_rate:{:.8f}".format(sess.run(learning_rate)))

            train_video_pred_label = []
            train_video_true_label = []
            train_acc = []
            train_cost = []
            for s in range(len(c_train_file_list)):
                c_train_file = c_train_file_list[s]
                c_train_f = h5py.File(c_train_file, 'r')
                c_train_x = c_train_f['feature']
                train_y = c_train_f['label']

                m_train_file = m_train_file_list[s]
                m_train_f = h5py.File(m_train_file, 'r')
                m_train_x = m_train_f['feature']

                train_x_len = len(c_train_x)
                for k in tqdm(range(0, train_x_len, batch_size)):
                    c_train_batch_x = c_train_x[k:k + batch_size]
                    m_train_batch_x = m_train_x[k:k + batch_size]
                    train_batch_y = train_y[k:k + batch_size]

                    _, cost, acc, p = sess.run(
                        [update, total_loss, accuracy, prob],
                        feed_dict={content_x: c_train_batch_x, motion_x: m_train_batch_x, y: train_batch_y,
                                   keep_prob: keep_prob_ratio, lstm_keep_prob: lstm_keep_prob_ratio})

                    train_video_pred_label += list(np.argmax(p, axis=1))
                    train_video_true_label += list(train_batch_y)

                    train_acc.append(acc)
                    train_cost.append(cost)

                c_train_f.close()
                m_train_f.close()

                if ((s + 1) * batch_num) % display_step == 0:
                    print('time:{} epoch:{} train_step:{} train_cost:{:.4f} train_acc:{:.4f}'.format(
                        time.asctime(time.localtime(time.time())), e + 1, (s + 1) * batch_num, np.mean(train_cost),
                        np.mean(train_acc)))

                    train_macro_f = metrics.f1_score(train_video_true_label, train_video_pred_label,
                                                     labels=np.array(range(class_num)),
                                                     average='macro')
                    train_micro_f = metrics.f1_score(train_video_true_label, train_video_pred_label,
                                                     labels=np.array(range(class_num)),
                                                     average='micro')
                    train_res[e, 0] = train_macro_f
                    train_res[e, 1] = train_micro_f
                    train_res[e, 2] = np.mean(train_cost)

                    print('\t\t\t\t\tvideo_macro_f:{:.4f} video_micro_f:{:.4f}'.format(
                        train_macro_f, train_micro_f))
                    train_cost = []
                    train_acc = []

            print('-' * 20 + 'start valid and test' + '-' * 20)

            val_video_pred_label = []
            val_video_true_label = []
            val_cost = []
            val_acc = []

            for s in range(len(c_val_file_list)):
                c_val_file = c_val_file_list[s]
                c_val_f = h5py.File(c_val_file, 'r')
                c_val_x = c_val_f['feature']
                val_y = c_val_f['label']

                m_val_file = m_val_file_list[s]
                m_val_f = h5py.File(m_val_file, 'r')
                m_val_x = m_val_f['feature']

                val_x_len = len(c_val_x)
                for k in tqdm(range(0, val_x_len, batch_size)):
                    c_val_batch_x = c_val_x[k:k + batch_size]
                    m_val_batch_x = m_val_x[k:k + batch_size]
                    val_batch_y = val_y[k:k + batch_size]

                    cost, acc, p = sess.run(
                        [total_loss, accuracy, prob],
                        feed_dict={content_x: c_val_batch_x, motion_x: m_val_batch_x, y: val_batch_y, keep_prob: 1.0,
                                   lstm_keep_prob: 1.0})

                    val_video_pred_label += list(np.argmax(p, axis=1))
                    val_video_true_label += list(val_batch_y)

                    val_acc.append(acc)
                    val_cost.append(cost)

                c_val_f.close()
                m_val_f.close()

            print(
                'time:{} epoch:{}   val_cost:{:.4f}   val_acc:{:.4f}'.format(
                    time.asctime(time.localtime(time.time())), e + 1, np.mean(val_cost),
                    np.mean(val_acc)))

            val_macro_f1 = metrics.f1_score(val_video_true_label, val_video_pred_label,
                                            labels=np.array(range(class_num)),
                                            average='macro')
            val_micro_f1 = metrics.f1_score(val_video_true_label, val_video_pred_label,
                                            labels=np.array(range(class_num)),
                                            average='micro')
            print(
                '\t\t\t\t\tvideo_macro_f:\033[0;34;40m{:.4f}\033[0m video_micro_f:\033[0;34;40m{:.4f}\033[0m'.format(
                    val_macro_f1, val_micro_f1))

            val_res[e, 0] = val_macro_f1
            val_res[e, 1] = val_micro_f1
            val_res[e, 2] = np.mean(val_cost)

            test_video_pred_label = []
            test_video_true_label = []
            test_cost = []
            test_acc = []

            for s in range(len(c_test_file_list)):
                c_test_file = c_test_file_list[s]
                c_test_f = h5py.File(c_test_file, 'r')
                c_test_x = c_test_f['feature']
                test_y = c_test_f['label']

                m_test_file = m_test_file_list[s]
                m_test_f = h5py.File(m_test_file, 'r')
                m_test_x = m_test_f['feature']

                test_x_len = len(c_test_x)
                for k in tqdm(range(0, test_x_len, batch_size)):
                    c_test_batch_x = c_test_x[k:k + batch_size]
                    m_test_batch_x = m_test_x[k:k + batch_size]
                    test_batch_y = test_y[k:k + batch_size]

                    cost, acc, p = sess.run(
                        [total_loss, accuracy, prob],
                        feed_dict={content_x: c_test_batch_x, motion_x: m_test_batch_x, y: test_batch_y, keep_prob: 1.0,
                                   lstm_keep_prob: 1.0})

                    test_video_pred_label += list(np.argmax(p, axis=1))
                    test_video_true_label += list(test_batch_y)

                    test_acc.append(acc)
                    test_cost.append(cost)
                c_test_f.close()
                m_test_f.close()

            print(
                'time:{} epoch:{}   test_cost:{:.4f}   test_acc:{:.4f}'.format(
                    time.asctime(time.localtime(time.time())), e + 1, np.mean(test_cost),
                    np.mean(test_acc)))

            test_macro_f1 = metrics.f1_score(test_video_true_label, test_video_pred_label,
                                             labels=np.array(range(class_num)),
                                             average='macro')
            test_micro_f1 = metrics.f1_score(test_video_true_label, test_video_pred_label,
                                             labels=np.array(range(class_num)),
                                             average='micro')
            print(
                '\t\t\t\t\tvideo_macro_f:\033[0;34;40m{:.4f}\033[0m video_micro_f:\033[0;34;40m{:.4f}\033[0m'.format(
                    test_macro_f1, test_micro_f1))
            test_res[e, 0] = test_macro_f1
            test_res[e, 1] = test_micro_f1
            test_res[e, 2] = np.mean(test_cost)

            index = np.argmax(val_res[:, 1])
            print(
                'val_res:\033[0;34;40m{:.4f}\033[0m\t\t\033[0;34;40m{:.4f}\033[0m'.format(val_res[index, 0],
                                                                                          val_res[index, 1]))
            print('test_res:\033[0;34;40m{:.4f}\033[0m\t\t\033[0;34;40m{:.4f}\033[0m'.format(test_res[index, 0],
                                                                                             test_res[index, 1]))
            np.save(os.path.join(ckpt_path, model_path) + '/val_res.npy', val_res)
            np.save(os.path.join(ckpt_path, model_path) + '/test_res.npy', test_res)
            np.save(os.path.join(ckpt_path, model_path) + '/train_res.npy', train_res)

            saver.save(sess, ckpt_path + '/' + model_path + '/' + model_name, global_step=global_step,
                       write_meta_graph=False)

            if (e + 1) % lr_decay_epoch == 0:
                sess.run(tf.assign(learning_rate, learning_rate * 0.1))


if __name__ == '__main__':
    train()
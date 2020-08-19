from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def global_attention(
        bottom,
        global_pooling=tf.reduce_mean,
        intermediate_nl=tf.nn.relu,
        squash=tf.nn.sigmoid,
        name=None,
        r=4,
        reuse=False,
        alpha=None):
    # squeeze
    mu = global_pooling(bottom, reduction_indices=[1, 2], keep_dims=True)
    c = int(mu.get_shape()[-1])
    intermediate_size = int(c / r)

    intermediate_activities = intermediate_nl(
        fc_layer(
            bottom=tf.contrib.layers.flatten(mu),
            out_size=intermediate_size,
            name='%s_ATTENTION_intermediate' % name,
            reuse=reuse))

    # expand
    out_size = c
    output_activities = fc_layer(
        bottom=intermediate_activities,
        out_size=out_size,
        name='%s_ATTENTION_output' % name,
        reuse=reuse)

    output_activities = squash(output_activities)
    exp_activities = tf.reshape(output_activities,[-1,1,1,c])

    scaled_bottom = alpha * (bottom * exp_activities) + bottom
    return scaled_bottom


def local_attention(
        bottom,
        intermediate_nl=tf.nn.relu,
        squash=tf.nn.sigmoid,
        name=None,
        intermediate_kernel=1,
        r=4,
        reuse=False,
        beta=None):
    # squeeze
    c = int(bottom.get_shape()[-1])
    intermediate_channels = int(c / r)
    intermediate_activities = tf.layers.conv2d(
        inputs=bottom,
        filters=intermediate_channels,
        kernel_size=intermediate_kernel,
        activation=intermediate_nl,
        padding='SAME',
        use_bias=True,
        kernel_initializer=tf.variance_scaling_initializer(),
        name='%s_ATTENTION_intermediate' % name,
        reuse=reuse)


    # expand
    output_activities = tf.layers.conv2d(
        inputs=intermediate_activities,
        filters=1,
        kernel_size=intermediate_kernel,
        padding='SAME',
        use_bias=True,
        activation=None,
        kernel_initializer=tf.variance_scaling_initializer(),
        name='%s_ATTENTION_output' % name,
        reuse=reuse)


    output_activities = squash(output_activities)

    scaled_bottom =  beta * (bottom * output_activities) + bottom
    return scaled_bottom


def fc_layer(
        bottom,
        out_size=None,
        name=None,
        reuse=False):

    in_size = int(bottom.get_shape()[-1])
    x = tf.reshape(bottom, [-1, in_size]) if len(bottom.get_shape()) > 2 else bottom
    out = tf.contrib.layers.fully_connected(x, out_size, scope=name,reuse=reuse, activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return out

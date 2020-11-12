import scipy
import numpy as np
from numpy import array
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print(tf.test.is_gpu_available())
print(tf.__version__)

import matplotlib.pyplot as plt




def generator(z,size_g_w1, size_g_b1, size_g_w2, size_g_b2):

    w1_std = 1.0/tf.sqrt(size_g_w1/2.0)
    G_W1 = tf.Variable(tf.random_normal(shape=[size_g_w1, size_g_b1], stddev=w1_std))
    G_b1 = tf.Variable(tf.zeros(shape=[size_g_b1]))

    w2_std = 1.0/tf.sqrt(size_g_w2/2.0)
    G_W2 = tf.Variable(tf.random_normal(shape=[size_g_w2, size_g_b2], stddev=w2_std))
    G_b2 = tf.Variable(tf.zeros(shape=[size_g_b2]))

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_logit = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_logit)

    theta_G = [G_W1, G_W2, G_b1, G_b2]


    return G_prob, G_logit, theta_G




def discriminator(x, size_d_w1,size_d_b1, size_d_w2, size_d_b2):

    w1_std = 1.0/tf.sqrt(size_d_w1/2.0)
    D_W1 = tf.Variable(tf.random_normal(shape=[size_d_w1,size_d_b1], stddev=w1_std))
    D_b1 = tf.Variable(tf.zeros(shape=[size_d_b1]))

    w2_std = 1.0/tf.sqrt(size_d_w2/2.0)
    D_W2 = tf.Variable(tf.random_normal(shape=[size_d_w2,size_d_b2], stddev=w2_std))
    D_b2 = tf.Variable(tf.zeros(shape=[size_d_b2]))

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    return D_prob, D_logit, theta_D

def sample_z(m, n):
    # randomly generate samples for generator
    return np.random.uniform(-1.0, 1.0, size = [m, n])
    

def get_G_loss(D_logit_fake, theta_G,learning_rate):

  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
  G_solver = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(G_loss, var_list=theta_G)
  return G_loss, G_solver

def get_D_loss(D_logit_real,D_logit_fake,theta_D,learning_rate):  
  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
  D_loss = D_loss_real + D_loss_fake
  D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
  return D_loss, D_solver
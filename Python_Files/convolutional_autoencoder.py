
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from numpy import array



def conv_layer(x, weight_shape, bias_shape):
    """
    Computes a 2 dimentional convolution given the 4d input and filter
    input:
        x: [batch, in_height, in_width, in_channels]
        weight: [filter_height, filter_width, in_channels, out_channels]
        bias: [out_channels]
    output:
        The relu activation of convolution
    """

    print([weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]])
    sizeIn = weight_shape[0] * weight_shape[1] * weight_shape[2]
    
    # initialize weights with data generated from a normal distribution.
    weight_init = tf.random_normal_initializer(stddev=(2.0/sizeIn)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    # initialize bias with zeros
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b))




def upconvolution (input, output_channel_size, filter_size_h, filter_size_w, stride_h, stride_w, dtype=tf.float32):
    """
    Computes a 2 dimentional upconvolution given the 4d input and filter
    input:
        input: [batch, in_height, in_width, in_channels]
        output_channel_size
        filter_size_h, filter_size_w: vertical, horizontal extent of the filters
        stride_h, stride_w: vertical, horizontal stride
        layer_name: scope for the layer
    output:
        The relu activation of upconvolution
    """
  
    input_channel_size = input.get_shape().as_list()[3]
    input_size_h = input.get_shape().as_list()[1]
    input_size_w = input.get_shape().as_list()[2]
    stride_shape = [1, stride_h, stride_w, 1]


    output_size_h = (input_size_h - 1)*stride_h + 2
    output_size_w = (input_size_w - 1)*stride_w + 2
  
    output_shape = tf.stack([tf.shape(input)[0], output_size_h, output_size_w, output_channel_size])


    #creating weights:
    shape_W = [filter_size_h, filter_size_w, output_channel_size, input_channel_size]
    W_upconv = tf.get_variable("w", shape=shape_W, dtype=dtype,initializer=tf.random_normal_initializer(stddev=(2.0/1000)**0.5))
    
    shape_b=[output_channel_size]
    b_upconv = tf.get_variable("b", shape=shape_b, dtype=dtype, initializer=tf.constant_initializer(value=0))
    
    upconv = tf.nn.conv2d_transpose(input, W_upconv, output_shape, stride_shape,padding='SAME')
    output = tf.nn.bias_add(upconv, b_upconv)
    
    output = tf.reshape(output, output_shape)
    
    return output




def pooling(x, k):
    """
    Extracts the main information of the conv layer by performs the max pooling on the input x.
    input:
        x: A 4-D Tensor. [batch, in_height, in_width, in_channels]
        k: The length of window
    """
    
    #value: A 4-D Tensor of the format specified by data_format. That is x in this case.
    #ksize: A 1-D int Tensor of 4 elements. The size of the window for each dimension of input
    #strides: A 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension of input
    #padding: A string, either 'VALID' or 'SAME'. Difference of 'VALID' and 'SAME' in tf.nn.max_pool:
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')




def conv_autoencoder(x):
    """
    define the structure of the whole network
    input:
        - x: a batch of pictures , input shape = [batch_size, image_size, image_size, 1]
    output:
        - a batch vector corresponding to the logits predicted by the network(output shape = input shape) 
    """

    x = tf.reshape(x, shape=[-1, 512, 512, 1])
    
    with tf.variable_scope("convolutional_layer_1"):

        # convolutional layer with 32 filters and spatial extent e = 5
        # this causes in taking an input of volume with depth of 1 and producing an output tensor with 32 channels.
        convolutional_1 = conv_layer(x, [5, 5, 1, 32], [32])
        
        # output in passed to max-pooling to be compressed (k=2 non-overlapping).
        pooling_1 = pooling(convolutional_1, 2)

        # shape : batch_size*256*256*32

    with tf.variable_scope("convolutional_layer_2"):
        
        # convolutional layer with 64 filters with spatial extent e = 5
        # taking an input tensor with depth of 32 and producing an output tensor with depth 64
        convolutional_2 = conv_layer(pooling_1, [5, 5, 32, 64], [64])
        
        # output in passed to max-pooling to be compressed (k=2 non-overlapping).
        pooling_2 = pooling(convolutional_2, 2)

        # shape : batch_size*128*128*64


    with tf.variable_scope("convolutional_layer_3"):
        
        # convolutional layer with 128 filters with spatial extent e = 5
        # taking an input tensor with depth of 64 and producing an output tensor with depth 128
        convolutional_3 = conv_layer(pooling_2, [5, 5, 64, 128], [128])
        
        # output in passed to max-pooling to be compressed (k=2 non-overlapping).
        pooling_3 = pooling(convolutional_3, 2)

        # shape : batch_size*64*64*128


    with tf.variable_scope("convolutional_layer_4"):
        
        # convolutional layer with 256 filters with spatial extent e = 5
        # taking an input tensor with depth of 128 and producing an output tensor with depth 256
        convolutional_4 = conv_layer(pooling_3, [5, 5, 128, 256], [256])
        
        # output in passed to max-pooling to be compressed (k=2 non-overlapping).
        pooling_4 = pooling(convolutional_4, 2)

        # shape : batch_size*32*32*256


    # downsampling is over, let's upsample
    with tf.variable_scope('upconv_1'):
        upsample_1 = upconvolution(pooling_4, 128, 5, 5, 2, 2, dtype=tf.float32)
        # shape : batch_size*64*64*128
    with tf.variable_scope('upconv_2'):
        upsample_2 = upconvolution(upsample_1, 64, 5, 5, 2, 2, dtype=tf.float32)
        # shape : batch_size*128*128*64
    with tf.variable_scope('upconv_3'):
        upsample_3 = upconvolution(upsample_2, 32, 5, 5, 2, 2, dtype=tf.float32)
        # shape : batch_size*256*256*32
    with tf.variable_scope('upconv_4'):
        upsample_4 = upconvolution(upsample_3, 1, 5, 5, 2, 2, dtype=tf.float32)
        # shape : batch_size*512*512*1

    return upsample_4




def loss(output, x): #L2 loss
    """
    Compute the loss of the auto-encoder
    intput:
        - output: the output of the decoder (shape: (batch_size * num_of_classes))
        - x: true value of the sample batch - this is the input of the encoder - (shape: (batch_size * num_of_classes))
    output:
        - loss: loss of the corresponding batch (scalar tensor)
    
    """
    with tf.variable_scope("training"):
        
        l2_measure = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), 1))
        train_loss = tf.reduce_mean(l2_measure)
        return train_loss




def training(cost, learning_rate = 0.001, global_step): # Adam optimizer   
    """
    defines the necessary elements to train the network
    input:
        - cost: the cost is the loss of the corresponding batch
        - global_step: number of batch seen so far, it is incremented by one 
     The .minimize() function is called at each iteration
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def next_batch(data, num):
    
    idx = np.arange(0 , int(data.shape[0]))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = array([data[i] for i in idx])

    return data_shuffle


def evaluate(output, x):
  
    """
    evaluates the accuracy on the validation set 
    input:
        -output: prediction vector of the network for the validation set
        -x: true value for the validation set
    output:
        - val_loss: loss of the autoencoder
        - in_image_op: input image 
        - out_image_op:reconstructed image 
        - val_summary_op: summary of the loss
    """
    
    with tf.variable_scope("validation"):
    
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x, name="val_diff")), 1))
        val_loss = tf.reduce_mean(l2_norm)

        return val_loss



def convolutional_autoencoder(data, batch_size,n_epochs, learning_rate): 

  #with strategy.scope():

  with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

              x = tf.placeholder("float", [None, 512,512,1])

              # define the autoencoder
              output = conv_autoencoder(x)

              # compute the loss
              cost = loss(output, x)

              # initialize the value of the global_step variable
              global_step = tf.Variable(0, name='global_step', trainable=False)

              train_op = training(cost,learning_rate, global_step)

              # evaluate the accuracy of the network (done on a validation set)
              eval_op = evaluate(output, x)

              # save and restore variables to and from checkpoints.
              saver = tf.train.Saver()

              # defines a session
              sess = tf.Session()

              # initialization of the variables
              init_op = tf.global_variables_initializer()
              sess.run(init_op)

              # Training cycle
              for epoch in range(n_epochs):

                  avg_cost = 0.
                  total_batch = int(data.shape[0]/batch_size)

                  # Loop over all batches
                  for i in range(total_batch):

                      minibatch_x = next_batch(data,batch_size)

                      # Fit training using batch data
                      _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x})

                      # Compute average loss
                      avg_cost += new_cost/total_batch



                  # Display logs per epoch step
                  if epoch % display_step == 0:

                      print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                  if epoch % 20 ==0: 
                      r = sess.run([output], feed_dict={x: next_batch(data,data.shape[0])})
                      f1 = plot_sample2(r[0][50:59], 3, 3)


              print("Optimization Done")   
              result = sess.run([output], feed_dict={x: next_batch(data,data.shape[0])})
              r_ = result[0]
              for p in range(222): 
                fname = 'GAN'+ '_' + str(p) + '_' + '.png'
                path = '/ResultGAN' + '/' + fname
                imageio.imwrite(f'{fname}',r_[p])

              print("Results Saved")

  return result[0]

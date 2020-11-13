

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from numpy import array


def layer(x, weight_shape, bias_shape, phase_train):
    
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector of the layer after the matrix multiplication and non linear transformation
    """
    
    #initialize weights
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    logits = tf.matmul(x, W) + b
    
    #apply the non-linear function after the batch normalization
    return tf.nn.sigmoid(layer_batch_normalization(logits, weight_shape[1], phase_train))



def layer_batch_normalization(x, n_out, phase_train):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - n_out: integer, depth of input maps - number of sample in the batch 
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - batch-normalized maps   
    """

    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    #calculate mean and variance of x
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

    #Maintains moving averages of variables by employing an exponential decay.
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
       
    #Return true_fn() if the predicate pred is true else false_fn()
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var, beta, gamma, 1e-3, True)
    
    return tf.reshape(normed, [-1, n_out])



def encoder(x, input_size, n_code, phase_train, neurons_per_layer):
    """
    Defines the network encoder part
    input:
        - x: input vector of the encoder
        - n_code: number of neurons in the code layer (output of the encoder - input of the decoder) 
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector: reduced dimension
    """
    n_encoder_h_1 = neurons_per_layer[0]
    n_encoder_h_2 = neurons_per_layer[1]
    n_encoder_h_3 = neurons_per_layer[2]
    n_encoder_h_4 = neurons_per_layer[3]


    with tf.variable_scope("encoder"):
        
        with tf.variable_scope("h_1"):
            h_1 = layer(x, [input_size, n_encoder_h_1], [n_encoder_h_1], phase_train)

        with tf.variable_scope("h_2"):
            h_2 = layer(h_1, [n_encoder_h_1, n_encoder_h_2], [n_encoder_h_2], phase_train)

        with tf.variable_scope("h_3"):
            h_3 = layer(h_2, [n_encoder_h_2, n_encoder_h_3], [n_encoder_h_3], phase_train)

        with tf.variable_scope("h_4"):
            h_4 = layer(h_3, [n_encoder_h_3, n_encoder_h_4], [n_encoder_h_4], phase_train)

        with tf.variable_scope("code"):
            output = layer(h_4, [n_encoder_h_4, n_code], [n_code], phase_train)

    return output





def decoder(x, output_size, n_code, phase_train ,neurons_per_layer):
    """
    Defines the network encoder part
    input:
        - x: input vector of the decoder - reduced dimension vector
        - n_code: number of neurons in the code layer (output of the encoder - input of the decoder)
        - phase_train: boolean tf.Varialbe, true indicates training phase
    output:
        - output vector: reconstructed dimension of the initial vector
    """
    
    n_decoder_h_1 = neurons_per_layer[0]
    n_decoder_h_2 = neurons_per_layer[1]
    n_decoder_h_3 = neurons_per_layer[2]
    n_decoder_h_4 = neurons_per_layer[3]
    
    with tf.variable_scope("decoder"):
        
        with tf.variable_scope("h_1"):
            h_1 = layer(x, [n_code, n_decoder_h_1], [n_decoder_h_1], phase_train)

        with tf.variable_scope("h_2"):
            h_2 = layer(h_1, [n_decoder_h_1, n_decoder_h_2], [n_decoder_h_2], phase_train)

        with tf.variable_scope("h_3"):
            h_3 = layer(h_2, [n_decoder_h_2, n_decoder_h_3], [n_decoder_h_3], phase_train)

        with tf.variable_scope("h_4"):
            h_4 = layer(h_3, [n_decoder_h_3, n_decoder_h_4], [n_decoder_h_4], phase_train)

        with tf.variable_scope("output"):
            output = layer(h_4, [n_decoder_h_4, output_size], [output_size], phase_train)

    return output



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




def training(cost,learning_rate=0.001, global_step): # Adam optimizer	
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
    
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = array([data[ i] for i in idx])

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







def ff_autoencoder(data, model_path,input_size,output_size, n_code, encoder_neurons, decoder_neurons,learning_rate,training_epochs, batch_size, display_step): 

  with strategy.scope():

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            # the input variables are first define as placeholder (a variable/data which will be assigned later)
            x = tf.placeholder("float", [None, 512*512])

            phase_train = tf.placeholder(tf.bool)

            # define the encoder/decoder model
            code = encoder(x, int(input_size), int(n_code), phase_train, encoder_neurons)
            output = decoder(code, int(output_size), int(n_code), phase_train, decoder_neurons)

            # compute the loss
            cost = loss(output, x)

            # initialize the value of the global_step variable
            # it is incremented by one each time the .minimize() is called, not trainable
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # define training operation
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
            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(len(data)/batch_size)

                # Loop over all batches
                for i in range(total_batch):

                    minibatch_x = next_batch(data,batch_size).reshape(-1,512*512)

                    # Fit training using batch data
                    _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x, phase_train: True})

                    # Compute average loss
                    avg_cost += new_cost/total_batch


                if epoch % display_step == 0:

                    print("Epoch:", '%04d' % (epoch+1),"cost =", "{:.9f}".format(avg_cost))
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)


            results = sess.run([output], feed_dict={x: data.reshape(-1,512*512), phase_train: False})
            print("Optimization Done")
            return results[0]



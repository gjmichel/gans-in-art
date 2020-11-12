
import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import imageio



def read_tensor_from_image_file(path, input_height, input_width, input_mean=0, input_std=255):
    
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(path, input_name)
    image_reader = tf.image.decode_png(file_reader, channels = 1)
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result 


def load_all_images(numInSeries,nSeries, image_size ): 

  nImages = numInSeries*nSeries
  img  = np.zeros((nImages, image_size,image_size))
  counter = 0

  for j in range(0,nSeries):
    for i in range(1,numInSeries+1):
          fname = str(i) + '_' + str(j) + '.png'
          path = '../data' + '/' + fname
          orig_img = read_tensor_from_image_file(path, image_size, image_size).reshape(1,512,512)
          img[counter] = orig_img.reshape(image_size,image_size)
          counter = counter+1
          if i%50 == 0:
            print(i)
  
  print('final_shape', img.shape)

  return img


def plot_sample(samples, size1, size2):
    
    fig1 = plt.figure(figsize=(size1, size2))
    gs = gridspec.GridSpec(size1, size2)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(512, 512), cmap='gray')

    return fig1


def save_images(results,numInSeries, path_to_folder): 
  nSeries = results.shape[0]
  for j in range(0,numInSeries):
    for i in range(nSeries):
        fname = str(i) + '_' + str(j) + '.png'
        path = path_to_folder + '/' + fname
        imageio.imwrite(f'{fname}',results[i])


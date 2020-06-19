import random
import matplotlib.pyplot as plt
import os, time,itertools, imageio, pickle
import tensorflow as tf
import numpy as np
from PIL import Image

tf.reset_default_graph()


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
b_init = tf.constant_initializer(1.0)

def generator(x, isTrain=True, reuse=False):
    
    with tf.variable_scope('generator', reuse=reuse):

        # encoder
        conv1 = lrelu(tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init))
        conv2 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv1, 128, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv3 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv2, 256, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv4 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv3, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv5 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv4, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv6 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv5, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv7 = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv6, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))
        conv8 = tf.nn.relu(tf.layers.conv2d(conv7, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init))

        # decoder and skip connections
        deconv1 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(conv8, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv1 = tf.nn.relu(tf.concat([deconv1, conv7], 3))
        
        deconv2 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv1, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv2 = tf.nn.relu(tf.concat([deconv2, conv6], 3))
        
        deconv3 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv2, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv3 = tf.nn.relu(tf.concat([deconv3, conv5], 3))
        
        deconv4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv3, 512, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv4 = tf.nn.relu(tf.concat([deconv4, conv4], 3))
        
        deconv5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv4, 256, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv5 = tf.nn.relu(tf.concat([deconv5, conv3], 3))
        
        deconv6 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv5, 128, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv6 = tf.nn.relu(tf.concat([deconv6, conv2], 3))
        
        deconv7 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(deconv6, 64, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv7 = tf.nn.relu(tf.concat([deconv7, conv1], 3))
        
        deconv8 = tf.nn.tanh(tf.layers.conv2d_transpose(deconv7, 3, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init))

        return deconv8

''' The following function actually builds the discriminator network for a single patch of the image'''
      
def discriminator_patch(x_patch, isTrain=True, reuse=False):
  
    with tf.variable_scope('discriminator', reuse=reuse):
        """
        Create the PatchGAN discriminator using the hyperparameter values defined below
        """
        kernel_size = 4
        strides = 2
        leakyrelu_alpha = 0.2
        padding = 'same'
        num_filters_start = 64  # Number of filters to start with
        num_kernels = 100
        kernel_dim = 5
        patchgan_output_dim = (256, 256, 1)
        patchgan_patch_dim = (256, 256, 1)
        number_patches = int(
            (patchgan_output_dim[0] / patchgan_patch_dim[0]) * (patchgan_output_dim[1] / patchgan_patch_dim[1]))

        conv = lrelu(tf.layers.conv2d(x_patch, 64, [kernel_size, kernel_size], strides=(strides, strides), padding=padding, kernel_initializer=w_init, bias_initializer=b_init))

        # Calculate the number of convolutional layers
        total_conv_layers = int(np.floor(np.log(patchgan_output_dim[1]) / np.log(2)))
        list_filters = [num_filters_start * min(total_conv_layers, (2 ** i)) for i in range(total_conv_layers)]

        # Next 7 Convolutional blocks
        for filters in list_filters[1:]:
            conv = lrelu(tf.layers.batch_normalization(tf.layers.conv2d(conv, filters, [kernel_size, kernel_size], strides=(strides, strides), padding=padding, kernel_initializer=w_init, bias_initializer=b_init), training=isTrain))

        flatten_layer = tf.layers.flatten(conv)
        out = tf.layers.dense(flatten_layer, 1, use_bias=False, activation=None)
        out = tf.nn.sigmoid(out)
        return out, flatten_layer

# This is the final part of the discriminator network that after having the prediction on each patch of the image produces the final prediction on the whole image
# by averagin the patches output and computing some sort of perceptual loss using the flattened layer output build in the discriminator
    
def discriminator_patch_final(out, flatten_layer, isTrain = True, reuse = False):
  with tf.variable_scope("discriminator_final", reuse = reuse):
    num_kernels = 100
    kernel_dim = 5
    
    output2 = tf.layers.dense(flatten_layer, num_kernels * kernel_dim, use_bias=False, activation=None)
    # Reshape the output2 tensor
    output2 = tf.reshape(output2, [num_kernels, kernel_dim])

    # Pass the output2 tensor through the custom_loss_layer
    # This should be the compution of the perceptual loss through the use of the output of the previous flatten layer
    output2 = tf.reduce_sum(
        tf.exp(-tf.reduce_sum(tf.abs(tf.expand_dims(out, 3) - tf.expand_dims(tf.transpose(out, perm=[1,2,0]), 0)), 2)), 2)
    output2 = tf.expand_dims(output2, 2)
    out = tf.concat([out, output2], axis = -1)
    logits = tf.layers.dense(out, 1, use_bias=False, activation=None)
    final_out = tf.nn.sigmoid(logits)
    return final_out, logits
      
def build_model(x,y,x_patch,x_final,x_flatten,g_x_final,g_x_flatten):
  
  G_x = generator(x)
  # The following 2 lines compute the prediction and the output of the flatten layer on the patches of the real images and generated images
  D_real_outputs_patch, D_real_flats = discriminator_patch(x_patch)
  D_fake_outputs_patch, D_fake_flats = discriminator_patch(x_patch, reuse=True)
  # The following 2 lines, instead, actually compute the discriminator output of the complete real or fake image after having the discriminator classification
  # of each image patch
  D_real_outputs, D_real_logits = discriminator_patch_final(x_final,x_flatten)
  D_fake_outputs, D_fake_logits = discriminator_patch_final(g_x_final, g_x_flatten, reuse=True)

  # Here I am defining the losses for the discriminator and generator using the outputs of the discriminator on the whole image
  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, 
                                                                       labels=tf.ones_like(D_real_logits)))
  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                       labels=tf.zeros_like(D_fake_logits)))
  D_loss = (D_loss_real + D_loss_fake)

  G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, 
                                                                  labels=tf.ones_like(D_fake_logits)))
  G_l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(G_x - y), 3))

  G_loss = G_loss_gan + G_l1_loss * l1_weight


  # For the discriminator I have to group togheter the weights of the discriminator patch network and final discriminator because they actually are part of the same
  # network and so during training their weights need to be updated at the same iteration
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator_final')
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

  D_solver = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(D_loss, var_list=D_vars)
  G_solver = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(G_loss, var_list=G_vars)
  return D_solver, G_solver, D_loss, G_loss, G_x, D_real_outputs_patch, D_real_flats, D_fake_outputs_patch, D_fake_flats

      
# to show the results

def norm(img):
    return (img - 127.5) / 127.5

def show_result(num_epoch):
    
    test_images = sess.run(G_x, {x: test_x})

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(4, 4))
    
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        #a = np.reshape(test_images[k], (256, 256, 3))*255
        a = np.reshape(test_images[k], (256, 256, 3)) * 127 + 127
        ax[i, j].imshow(a.astype(np.uint8))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    plt.show()
    
class data_loader:
    def __init__(self, root, batch_size=1, shuffle=False):
        self.root = root
        self.batch_size = batch_size
        self.file_list = os.listdir(self.root)
        if shuffle:
            self.file_list = list(np.array(self.file_list)[random.sample(range(0, len(self.file_list)), len(self.file_list))])
        img = plt.imread(self.root + '/' + self.file_list[0])
        self.shape = (len(self.file_list), img.shape[0], img.shape[1], img.shape[2])
        self.flag = 0

    def next_batch(self):
        if self.flag + self.batch_size > self.shape[0]:
            self.file_list = list(np.array(self.file_list)[random.sample(range(0, len(self.file_list)), len(self.file_list))])
            self.flag = 0

        output = np.zeros((self.batch_size, self.shape[1], self.shape[2], self.shape[3]))
        temp = 0
        for i in range(self.flag, self.flag + self.batch_size):
            output[temp] = plt.imread(self.root + '/' + self.file_list[i])
            temp = temp + 1

        self.flag += self.batch_size

        return output
   
# This is the function to extract the patches given the batch data of real and generated images   
def generate_and_extract_patches(real, generated, patch_dim, output_images):

    patches_real_all = []
    patches_generated_all = []
    for real_img,generated_img in zip(real,generated):
      patches_real = []
      patches_generated = []
      for y in range(0, output_images[0], patch_dim[0]):
          for x in range(0, output_images[1], patch_dim[1]):
              image_patches = real_img[y: y + patch_dim[0], x: x + patch_dim[1], :]
              patches_real.append(np.asarray(image_patches, dtype=np.float32))
              image_patches = generated_img[y: y + patch_dim[0], x: x + patch_dim[1], :]
              patches_generated.append(np.asarray(image_patches, dtype=np.float32))
      patches_real_all.append(patches_real)
      patches_generated_all.append(patches_generated)

    return patches_real_all, patches_generated_all

def training():    
  DATA_PATH = "/content/drive/My Drive/VisionAndPerception/pix2pix-tensorflow-master/datasets/facades/"
  batch_size = 1
  batch_size_test = 16

  train = data_loader(DATA_PATH+"train", batch_size, shuffle=True)
  test = data_loader(DATA_PATH + "test", batch_size_test, shuffle=True)

  test_img = test.next_batch()
  img_size = test_img.shape[1]
  channels = 3
  l1_weight = 100.0
  patchgan_output_dim = (256, 256, 1)
  patchgan_patch_dim = (256, 256, 1)
  image_output_dim = (256, 256, 1)
  number_patches = int(
          (patchgan_output_dim[0] / patchgan_patch_dim[0]) * (patchgan_output_dim[1] / patchgan_patch_dim[1]))
  test_img.shape

  x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, channels))
  y = tf.placeholder(tf.float32, shape=(None, img_size, img_size, channels))
  x_patch = tf.placeholder(tf.float32, shape=(None, patchgan_patch_dim[0], patchgan_patch_dim[1], channels))
  x_final = tf.placeholder(tf.float32, shape=(None, number_patches, 1))
  x_flatten = tf.placeholder(tf.float32, shape=(None, None, 512))
  g_x_final = tf.placeholder(tf.float32, shape=(None, number_patches, 1))
  g_x_flatten = tf.placeholder(tf.float32, shape=(None, None, 512))

  D_solver, G_solver, D_loss, G_loss, G_x, D_real_outputs_patch, D_real_flats, D_fake_outputs_patch, D_fake_flats = build_model(x,y,x_patch,x_final,x_flatten,g_x_final,g_x_flatten)

  test_x = test_img[:, :, img_size:, :]
  test_y_ = test_img[:, :, 0:img_size, :]
  test_x = norm(test_x)

  plt.imshow(test_x[7].astype(np.uint8))
  plt.show()

  saver = tf.train.Saver()

  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for it in range(150):

          epoch_start_time = time.time()

          for iter in range(train.shape[0] // batch_size):

              train_data = train.next_batch()

              train_x = norm(train_data[:, :, img_size:, :])
              train_y = norm(train_data[:, :, 0:img_size, :])
              # I need to concatenate back the label and input image because we need to extractthe patches from the couple
              real_concatenate = np.concatenate((train_x,train_y),axis = 2)
              real_concatenate = norm(real_concatenate)
              # I need to generate the fake images and concatenate with input images in order to extract the patches to train discriminator
              generated_images = sess.run(G_x, {x: train_x})
              generated_concatenate = np.concatenate((train_x,generated_images),axis = 2)
              generated_concatenate = norm(generated_concatenate)
              # Now I extract the patches from the real and fake images
              patches_real, patches_generated = generate_and_extract_patches(real_concatenate, generated_concatenate, patchgan_patch_dim, image_output_dim)
              # The following lines run the patch discriminator on each patch of the images in the batch and retrieve the discriminator classification and output
              # of the flatten layer. Meanwhile it also build the correct input to pass to the final part of the discriminator for its training
              outs_real_all = []
              outs_gen_all = []
              flats_real_all = []
              flats_gen_all = []
              for real,gen in zip(patches_real,patches_generated):
                outs_real = []
                outs_gen = []
                flats_real = []
                flats_gen = []
                for patch_real,patch_gen in zip(real,gen):
                  # For each patch both in the case of real and fake images run the discriminator and get the discriminator output and the flatten layer output
                  out_real, flat_real = sess.run([D_real_outputs_patch, D_real_flats], feed_dict={x_patch: [patch_real]})
                  out_gen, flat_gen = sess.run([D_fake_outputs_patch, D_fake_flats], feed_dict={x_patch: [patch_gen]})
                  # For each image in the batch accumulate the patch classification and flatten layer output
                  outs_real.append(out_real[0])
                  flats_real.append(flat_real[0])
                  outs_gen.append(out_gen[0])
                  flats_gen.append(flat_gen[0])
                # Now we are building the correct batch input to the final part of the discriminator
                outs_real_all.append(outs_real)
                outs_gen_all.append(outs_gen)
                flats_real_all.append(flats_real)
                flats_gen_all.append(flats_gen)
                
              # After having run the discriminator in each image patch I can run the updatin operation for the discriminator
              _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x_final: outs_real_all,x_flatten: flats_real_all, g_x_final: outs_gen_all,g_x_flatten: flats_gen_all})
              # Now we can update the generator
              _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: train_x, y: train_y,g_x_final: outs_gen_all,g_x_flatten: flats_gen_all})
              

          if it % 1 == 0:

              show_result((it + 1))

              print('Iter: {}'.format(it))
              print('D loss: {:.4}'. format(D_loss_curr))
              print('G_loss: {:.4}'.format(G_loss_curr))
              print()
              saver.save(sess, "/content/drive/My Drive/VisionAndPerception/model_NLP.ckpt")

          epoch_end_time = time.time()
          per_epoch_ptime = epoch_end_time - epoch_start_time
          print("The total time for epoch{0} is{1} ".format(it,per_epoch_ptime))
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from glob import glob
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    vgg_tag = 'vgg16'
    model = tf.saved_model.loader.load(sess, [vgg_tag], export_dir=vgg_path)

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    graph = tf.get_default_graph()
    w0   = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3   = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4   = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7   = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w0, keep, w3, w4, w7
# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Use a 1x1 convolution before using layers 3, 4, and 7
    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Upsample layer 7
    conv_2x_7 = tf.layers.conv2d_transpose(conv_1x1_7, num_classes, 4, strides=(2, 2), padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Combine layers 4 and 7
    comb_4_7 = tf.add(conv_2x_7, conv_1x1_4)
    # Upsample the combined layers
    conv_2x_4_7 = tf.layers.conv2d_transpose(comb_4_7, num_classes, 4, strides=(2, 2), padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Combine layers 3, 4, and 7
    comb_3_4_7 = tf.add(conv_2x_4_7, conv_1x1_3)
    # Upsample the combined layers
    conv_8x_3_4_7 = tf.layers.conv2d_transpose(comb_3_4_7, num_classes, 16, strides=(8, 8), padding="same", \
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    
    return conv_8x_3_4_7
# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # print(logits)
    # tf.Print(logits,[logits])
    correct_label1 =  tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label1))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    #Evaluate how well the loss and accuracy of the model for a given dataset.
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label1, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #Evaluate the accuracy within each class
    per_class_accuracy = None # tf.metrics.mean_per_class_accuracy(labels=tf.argmax(correct_label1, 1), predictions=tf.argmax(logits, 1), num_classes=num_classes)
    predictions = tf.argmax(logits, 1)
    ground_truths = tf.argmax(correct_label1, 1)
    return logits, accuracy_operation, train_op, per_class_accuracy, predictions, ground_truths
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, accuracy_operation, image_input, \
             correct_label, keep_prob, learning_rate, images_count, image_shape, data_dir, per_class_accuracy, predictions, ground_truths):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Training...")
    print()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(epochs):
        total_accuracy = 0.0
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape, batch_size)
        #create a dictionary that will hold accuracy information
        acc_dict = {}
        for j in range(0,images_count,batch_size):
            # print(j)
            images_input, images_labels = next(get_batches_fn)
            #obtain the weights for each type of label to be inputted into the training
            weights = [0]*len(set(images_labels*1))
            for lab in list(set(images_labels*1)):
                weights[lab] = len(np.where(np.multiply(images_labels,1)==lab)[0])*1.0/len(images_labels)
            sess.run(train_op, feed_dict={image_input:images_input, correct_label:images_labels, keep_prob:0.5})
            accuracy = sess.run(accuracy_operation, feed_dict={image_input:images_input, correct_label:images_labels, keep_prob:1.0})
            preds = sess.run(predictions, feed_dict={image_input:images_input, correct_label:images_labels, keep_prob:1.0})
            gts = sess.run(ground_truths, feed_dict={image_input:images_input, correct_label:images_labels, keep_prob:1.0})
            # Use the preds and gts to find the accuracy within each class
            #for each label in the gts
            for k in list(set(gts)):
                #if k is not in acc_dict.keys, add it in
                if not k in acc_dict.keys():
                    acc_dict[k] = [0,0]
                #two items must be kept: the number of correct predictions[0] and the number of ground truths equal to k[1]
                acc_dict[k][0] = acc_dict[k][0] + np.sum(np.isin(np.where(gts==k)[0],np.where(preds==k)[0])) #np.sum([np.where(gts==k)[0][l] in np.where(preds==k)[0] for l in range(len(np.where(gts==k)[0]))])
                acc_dict[k][1] = acc_dict[k][1] + len(np.where(gts==k)[0])
            # print(class_accuracy)
            # print(accuracy)
            # print(images_input.shape)
            # print(images_input.shape[0])
            total_accuracy += (accuracy*images_input.shape[0])
        total_accuracy /= images_count
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(total_accuracy))
        # print the accuracy for each class
        for k in acc_dict.keys():
            print(k, " , ", acc_dict[k][0]*1.0/acc_dict[k][1])
        print()

        
    pass
# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = '/runs'
    #obtain the number of images in the training set
    images_count = len(glob(os.path.join(data_dir, 'data_road/training/image_2/*.png')))
    # print(images_count)
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        #initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Path to vgg model
        vgg_path = 'vgg_model/vgg'
        w0, keep, w3, w4, w7 = load_vgg(sess, vgg_path)
        # print(w0)
        #initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Create function to get batches
        BATCH_SIZE = 15
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape, BATCH_SIZE)
        a = next(get_batches_fn)
        # print(type(a))
        # print(len(a))
        # print(type(a[0]))
        # print(len(a[0]))
        imgs, labels = a
        # print(imgs.shape)
        # print(labels.shape)
        # print(np.unique(labels))
        # print(labels[0,80,200:300,0:2])

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        output = layers(w3, w4, w7, num_classes)
        correct_label = tf.placeholder(tf.float32, [None,160,576,2], name="y_restore")
        # print(correct_label)
        # print(tf.trainable_variables())
        learning_rate = 1e-3
        logits, accuracy_operation, train_op, per_class_accuracy, predictions, ground_truths = optimize(output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        EPOCHS = 50

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, accuracy_operation, w0, \
            correct_label, keep, learning_rate, images_count, image_shape, data_dir, per_class_accuracy, predictions, ground_truths)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

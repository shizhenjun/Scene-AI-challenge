"""Training a scene classification with TensorFlow using softmax cross entropy loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from scipy import misc
from util import load_image
import models
from models import nets_factory 
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import scenenet
import h5py
import json
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def main(args):
  
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    accuracyfile = os.path.join(model_dir, 'accuracy.json')

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    print ('Writing arguments in experiment to: %s' % os.path.join(log_dir, 'arguments.txt'))
    scenenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
       
    # Store some git revision info in a text file in the log directory
    #src_path,_ = os.path.split(os.path.realpath(__file__))
    #scenenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    train_dir = os.path.expanduser(args.train_data_dir)
    print ('train dataset: %s' %  train_dir)
    train_set = scenenet.get_dataset(args.train_data_dir)
    nrof_classes = len(train_set)

    validation_dir=os.path.expanduser(args.val_dir)
    print ('validation dataset: %s' % validation_dir)
    val_set = scenenet.get_dataset(args.val_dir)
    
    pretrained_model = None
    if args.pretrained_model_path:
        if os.path.exists(args.pretrained_model_path):
            pretrained_model = os.path.expanduser(args.pretrained_model_path)
            print('Pre-trained model: %s' % pretrained_model)
    
    imagenetmodel = None
    if args.fine_tune and args.imagenetmodel_path:
        if os.path.exists(args.imagenetmodel_path):
            imagenetmodel = os.path.expanduser(args.imagenetmodel_path)
            print('ImageNet Model: %s' % imagenetmodel)
    
    def resize_misc(img):
        return misc.imresize(img,(args.image_size, args.image_size))

    def resize_misc_short(img):
        return scenenet.resize_short(img, 256)

    def resize_misc_random(img):
        shorter_size=random.randint(240,272)
        return scenenet.resize_short(img, shorter_size)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)      
        
        #train set
        image_list, label_list = scenenet.get_image_paths_and_labels(train_set)
        #val set
        eval_image_list,eval_label_list=scenenet.get_image_paths_and_labels(val_set)
        num_of_samples = len(label_list)
        assert num_of_samples > 0, 'The dataset should not be empty'

        #cal the num of batch in one epoch
        batch_num_perepoch = num_of_samples // args.batch_size
        #batch_num_perepoch = 100#ce shi shi yong
        
        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size * batch_num_perepoch, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int32],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
        
        #nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(args.nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                #bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
                #bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(image_size =  tf.shape(image), bounding_boxes=bbox)
                #image = tf.slice(image, bbox_begin, bbox_size)
                #image = tf.image.resize_images(image, [args.image_size, args.image_size])
                #image = tf.py_func(resize_misc, [image], tf.uint8)
                if args.random_crop:                
                    if args.scalejittering:
                        image = tf.py_func(resize_misc_random, [image], tf.uint8)
                        image = tf.random_crop(image,[args.image_size, args.image_size, 3])
                    else:
                        image = tf.py_func(resize_misc_short, [image], tf.uint8)
                        image = tf.random_crop(image,[args.image_size, args.image_size, 3])
                else:
                    tf.py_func(resize_misc, [image], tf.uint8)

                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
                
                if args.random_coloradjustment:
                    image = random_distort_color(image)
    
                #if args.pcajittering:
                #    image = tf.py_func(scenenet.PcaJittering, [image], tf.float32)
                
                if args.random_rotate:
                    image = tf.py_func(scenenet.random_rotate_image, [image], tf.uint8)

                #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                
                #image = tf.image.resize_images(image, [args.image_size, args.image_size])
                #image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                image.set_shape((args.image_size, args.image_size, 3))
                #image = tf.py_func(scenenet.prewhiten, [image], tf.float32)
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * args.nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')        
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        
        print('Building training graph')
        #train net
        network = nets_factory.get_res_network_fn(name=args.modelname, num_classes=80,
            weight_decay=args.weight_decay, is_training=phase_train_placeholder, reuse=None)        
        logits,_ = network(image_batch)

        #logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
        #        weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
        #        weights_regularizer=slim.l2_regularizer(args.weight_decay),
        #        scope='Logits', reuse=False)        
        init_fn = None
        if imagenetmodel:
            print ('Load imagenetmodel from: %s' % imagenetmodel)
            exclude_map ={
                #'resnet_v2_50': ['resnet_v2_50/logits','Logits'],
                #'resnet_v2_101': ['resnet_v2_101/logits','Logits'],
                #'resnet_v2_152': ['resnet_v2_152/logits','Logits'],
                'resnet_v2_50': ['resnet_v2_50/logits'],
                'resnet_v2_101': ['resnet_v2_101/logits'],
                'resnet_v2_152': ['resnet_v2_152/logits'],
                #'resnet_v1_50': ['resnet_v1_50/logits','Logits'],
                #'resnet_v1_101': ['resnet_v1_101/logits','Logits'],
                #'resnet_v1_152': ['resnet_v1_152/logits','Logits']
                'resnet_v1_50': ['resnet_v1_50/logits'],
                'resnet_v1_101': ['resnet_v1_101/logits'],
                'resnet_v1_152': ['resnet_v1_152/logits']                                                
                }
            exclude = exclude_map[args.modelname]
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)
            print ('the number of parameters from ImageNetModel: {}'.format(len(variables_to_restore)))
            print (variables_to_restore)   
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(imagenetmodel, variables_to_restore) 


        predictions = tf.nn.softmax(logits, name='predictions')
        top1 = tf.argmax(predictions, axis=1, name='top1')
        _,top3 = tf.nn.top_k(predictions, k=3, name='top3')

        #evaluate net
        eval_x_placeholder=tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 3])
        eval_network = nets_factory.get_res_network_fn(name=args.modelname, num_classes=80,
            weight_decay=0.0, is_training=False, reuse=True)
        eval_logits,_=eval_network(eval_x_placeholder)
        #eval_logits=slim.fully_connected(eval_prelogits,len(train_set),activation_fn=None,
        #    weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
        #    weights_regularizer=slim.l2_regularizer(0.0),
        #    scope='Logits', reuse=True)
        eval_predictions=tf.nn.softmax(eval_logits,name='eval_predictions')
        eval_top1=tf.argmax(eval_predictions, axis=1, name='eval_top1')
        _,eval_top3=tf.nn.top_k(eval_predictions, k=3, name='eval_top3')
        
        
        global_step = tf.Variable(0, trainable=False)
        # Add center loss
        #if args.center_loss_factor>0.0:
        #    prelogits_center_loss, _ = scenenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        #    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
        #setup learning rate
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs * batch_num_perepoch, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = scenenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver = tf.train.Saver( max_to_keep=3)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()


        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #print ('Initialize uninitialized variables')
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        
        #There are some problems probably,be careful!!


        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            print ('Initialize uninitialized variables')
            if init_fn is not None:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())           
                init_fn(sess)
            else:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
           
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                ckpt = tf.train.get_checkpoint_state(pretrained_model)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    if os.path.exists(os.path.join(pretrained_model,'accuracy.json')):
                        acc_json = json.load(open(os.path.join(pretrained_model,'accuracy.json'), 'r'))
                        last_top3_acc = acc_json['accuracy']
                        last_epoch = acc_json['epoch']
                    print ('Model restored from {}. Last accuracy {}. Last epoch {}.'\
                        .format(ckpt.model_checkpoint_path, last_top3_acc, last_epoch))
                    #top3_acc,top1_acc=evaluate(args, sess, eval_image_list, eval_label_list, eval_top1, eval_top3, eval_x_placeholder, 1, log_dir)
            #tf.get_default_graph().finalize()
          
            # Training and validation loop
            print('Running training')
            epoch = 0
            last_top3_acc = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // batch_num_perepoch
                # Train for one epoch
                train(args, sess, epoch, batch_num_perepoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)

                
                top3_acc,top1_acc=evaluate(args, sess, eval_image_list, eval_label_list, eval_top1, eval_top3, eval_x_placeholder, epoch, log_dir)
                if top3_acc > last_top3_acc:
                    last_top3_acc = top3_acc
                    #Save variables and the metagraph if it doesn't exist already
                    save_variables_and_metagraph(sess, saver, summary_writer, model_dir, args.modelname, epoch)
                    acc_json = {'accuracy':last_top3_acc, 'epoch':epoch}
                    with open(accuracyfile, 'w') as f:
                        json.dump(acc_json, f, indent = 4)
                    print ('Better Accuracy: {}.Save model at {}.Save accuracy at {}'\
                        .format(last_top3_acc, model_dir, accuracyfile))


    return model_dir

 
def train(args, sess, epoch, batch_num_perepoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
        learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
        loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):     
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = scenenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    epoch_start_time = time.time()
    for batch_number in tqdm(xrange(batch_num_perepoch)):
    #batch_number = 0
    #while batch_number < batch_num_perepoch:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)

            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
                (epoch, batch_number+1, batch_num_perepoch, duration, err, np.sum(reg_loss)))
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        #batch_number += 1
    
    epoch_end_time = time.time()
    epoch_total_train_time = epoch_end_time - epoch_start_time
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=epoch_total_train_time)
    summary_writer.add_summary(summary, step)
    return step

def random_distort_color(image):
    color_order = random.randint(0, 5)
    if color_order == 0:
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)
    elif color_order == 1:
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)
    elif color_order == 2:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)
    elif color_order == 3:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)
    elif color_order == 4:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)
    else:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.0)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.0)

    return tf.clip_by_value(image, 0.0, 1.0)


def evaluate(args, sess, eval_image_list, eval_label_list, eval_top1, eval_top3, eval_x_placeholder, epoch, log_dir):
    start_time = time.time()
    top1_corrent = 0
    top3_corrent = 0
    n_data = 0
    num = len(eval_image_list)
    batch_size=args.batch_size
    for start, end in zip(
        xrange(0, num+batch_size, batch_size),
        xrange(batch_size, num+batch_size, batch_size)):
        current_image_paths = eval_image_list[start:end]
        current_image_labels = eval_label_list[start:end]
        current_images = scenenet.load_data_with_resize(current_image_paths, False, False, args.image_size)
        #good_index = np.array(map(lambda x: x is not None, current_images))
        #good_index = []
        #for i in range(len(current_images)):
        #    if current_images[i] is not None:
        #        good_index.append(i)
        #good_image_paths = []
        #good_image_labels = []
        #good_images = []
        #for i in good_index:
        #    good_image_paths.append(current_image_paths[i])
        #    good_image_labels.append(current_image_labels[i])
        #    good_images.append(current_images[i])
        #good_images = np.stack(good_images)
        
        val_top1, val_top3 = sess.run([eval_top1,eval_top3], 
                                    feed_dict = {eval_x_placeholder:current_images})
        for j, row in enumerate(val_top3):
            if current_image_labels[j] in row:
                top3_corrent += 1

        for j, cla in enumerate(val_top1):
            if current_image_labels[j] == val_top1[j]:
                top1_corrent += 1
        n_data += len(current_image_labels)

    end_time = time.time()
    durationtime = end_time - start_time
    top1_accuracy = top1_corrent / float(n_data)
    top3_accuracy = top3_corrent / float(n_data)
    print ('\nvalidation top3_accuracy: {:.5f}, top1_accuracy: {:.5f},time: {:.2f}'\
        .format(top3_accuracy, top1_accuracy, durationtime))
    with open(os.path.join(log_dir, 'accuracylog.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (epoch, top3_accuracy, top1_accuracy))
    return top3_accuracy,top1_accuracy


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='../logs/')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='../models/')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model_path', type=str,
        help='Load a pretrained model before training starts.A dir for the pretrained model')
    parser.add_argument('--fine_tune',
        help='Fine-tune model from ImageNet Model', action='store_true')
    parser.add_argument('--imagenetmodel_path', type=str,
        help='path to a ImageNet Model')
    parser.add_argument('--train_data_dir', type=str,
        help='Path to the train data directory. Multiple directories are separated with colon.',
        default='../dataset/train/')
    parser.add_argument('--modelname', type=str,
        help='Model Name.', default='resnet_v2_152')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=50)
    parser.add_argument('--dropout_keep',type=float,
        help='dropout_keep_prob', default=0.8)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=16)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    #parser.add_argument('--epoch_size', type=int,
    #    help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=512)

    parser.add_argument('--scalejittering',
        help='Performs scalejittering on train image', action='store_true')
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images.', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')

    parser.add_argument('--random_coloradjustment', 
        help='Peforms random color augmentation', action='store_true')
    #parser.add_argument('--pcajittering',
    #    help='if or not do pcajittering on image', action='store_true')    
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0001)
    #parser.add_argument('--center_loss_factor', type=float,
    #    help='Center loss factor.', default=0.0)
    #parser.add_argument('--center_loss_alfa', type=float,
    #    help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0005)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='../data/learning_rate_schedule.txt')
    parser.add_argument('--val_dir', type=str,
        help='path to the dir of validation dataset', default='../dataset/validation/')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

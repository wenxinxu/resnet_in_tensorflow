from cifar10_input import *
from resnet import *

def test(test_image_array):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance

    :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
    img_depth]
    :return: the softmax probability with shape [num_test_images, num_labels]
    '''
    num_test_images = len(test_image_array)
    num_batches = num_test_images // FLAGS.test_batch_size
    remain_images = num_test_images % FLAGS.test_batch_size
    print '%i test batches in total...' %num_batches

    # Create the test image and labels placeholders
    test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    # Build the test graph
    logits = inference(test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
    predictions = tf.nn.softmax(logits)

    # Initialize a new session and restore a checkpoint
    saver = tf.train.Saver(tf.all_variables())
    
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver.restore(sess, FLAGS.test_ckpt_path)
    print 'Model restored from ', FLAGS.test_ckpt_path

    prediction_array = np.array([]).reshape(-1, NUM_CLASS)
    # Test by batches
    for step in range(num_batches):
        if step % 10 == 0:
            print '%i batches finished!' %step
        offset = step * FLAGS.test_batch_size
        test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

        batch_prediction_array = sess.run(predictions,
                                    feed_dict={test_image_placeholder: test_image_batch})

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))

    # If test_batch_size is not a divisor of num_test_images
    if remain_images != 0:
        test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        # Build the test graph
        logits = inference(test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
        predictions = tf.nn.softmax(logits)

        test_image_batch = test_image_array[-remain_images:, ...]

        batch_prediction_array = sess.run(predictions, feed_dict={
            test_image_placeholder: test_image_batch})

        prediction_array = np.concatenate((prediction_array, batch_prediction_array))

    return prediction_array



write_file = open("predict_ret.txt", "w+")
test_image_array,test_labels = read_validation_data() # Better to be whitened in advance. Shape = [-1, img_height, img_width, img_depth]
predictions = test(test_image_array)

accuracy,cnt = 0,0

for i in range(len(predictions)):
    top1_predicted_label = np.argmax(predictions[i])

    true_label = int(test_labels[i])

    write_file.write('{}, {}, {}, {}\n'.format(
        true_label,
        predictions[i][true_label],
        top1_predicted_label,
        predictions[i][top1_predicted_label]))
    cnt += 1
    if true_label==top1_predicted_label:
        accuracy += 1
print("done!")
print("Test Accuracy={}".format(float(accuracy)/float(cnt)))
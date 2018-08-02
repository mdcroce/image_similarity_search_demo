import numpy as np
import time
import utils
from datetime import datetime


def feature_extraction_queue(feature_extractor, image_path, layer_names,
                             batch_size, num_images=300000):
    '''
    Given a directory containing images, this function extracts features
    for all images. The layers to extract features from are specified
    as a list of strings. First, we seek for all images in the directory,
    sort the list and feed them to the filename queue. Then, batches are
    processed and features are stored in a large object `features`.

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param layer_names: list of str, list of layer names
    :param batch_size: int, batch size
    :param num_images: int, number of images to process (default=100000)
    :return:
    '''

    # Add a list of images to process, note that the list is ordered.
    image_files = utils.find_files(image_path, ("jpg", "png"))
    num_images = min(len(image_files), num_images)
    image_files = image_files[0:num_images]

    num_examples = len(image_files)
    num_batches = int(np.ceil(num_examples/batch_size))

    # Fill-up last batch so it is full (otherwise queue hangs)
    utils.fill_last_batch(image_files, batch_size)

    print("#"*80)
    print("Batch Size: {}".format(batch_size))
    print("Number of Examples: {}".format(num_examples))
    print("Number of Batches: {}".format(num_batches))

    # Add all the images to the filename queue
    feature_extractor.enqueue_image_files(image_files)

    # Initialize containers for storing processed filenames and features
    feature_dataset = {'filenames': []}
    for i, layer_name in enumerate(layer_names):
        layer_shape = feature_extractor.layer_size(layer_name)
        layer_shape[0] = len(image_files)  # replace ? by number of examples
        feature_dataset[layer_name] = np.zeros(layer_shape, np.float32)
        print("Extracting features for layer '{}' with shape {}".format(layer_name, layer_shape))

    # Perform feed-forward through the batches
    for batch_index in range(num_batches):

        t1 = time.time()

        # Feed-forward one batch through the network
        outputs = feature_extractor.feed_forward_batch(layer_names)

        for layer_name in layer_names:
            start = batch_index*batch_size
            end   = start+batch_size
            feature_dataset[layer_name][start:end] = outputs[layer_name]

        # Save the filenames of the images in the batch
        feature_dataset['filenames'].extend(outputs['filenames'])

        t2 = time.time()
        examples_in_queue = outputs['examples_in_queue']
        examples_per_second = batch_size/float(t2-t1)

        print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Examples in Queue = {}, Examples/Sec = {:.2f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), batch_index+1,
            num_batches, batch_size, examples_in_queue, examples_per_second
        ))

    # If the number of pre-processing threads >1 then the output order is
    # non-deterministic. Therefore, we order the outputs again by filenames so
    # the images and corresponding features are sorted in alphabetical order.
    if feature_extractor.num_preproc_threads > 1:
        utils.sort_feature_dataset(feature_dataset)

    # We cut-off the last part of the final batch since this was filled-up
    feature_dataset['filenames'] = feature_dataset['filenames'][0:num_examples]
    for layer_name in layer_names:
        feature_dataset[layer_name] = feature_dataset[layer_name][0:num_examples]

    return feature_dataset
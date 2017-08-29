import numpy as np
import os
import sys
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

# Necessary to explicitly add the models/ folders
sys.path.append('/home/edlectrico/anaconda3/lib/python3.6/site-packages/tensorflow/models')
sys.path.append('/home/edlectrico/anaconda3/lib/python3.6/site-packages/tensorflow/models/slim')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def test_display_image():
    print('testing image show from plt')
    image = Image.open('images/teste-7.jpg')
    image_np = load_image_into_numpy_array(image)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()


if __name__ == '__main__':

    print('Running __main__')
    # Importing object_detection modules from TensorFlow. The compiler might say that
    # there is an error in these lines, but the truth is that when sys.path.append lines
    # are executed, then the compiler is able to import these two modules
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util

    print('Model preparation...')
    '''
    Any model exported using the export_inference_graph.py tool can be loaded here simply by 
    changing PATH_TO_CKPT to point to a new .pb file. By default we use an "SSD with Mobilenet" 
    model here. See the detection model zoo for a list of other models that can be run 
    out-of-the-box with varying speeds and accuracies.
    '''

    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    print('Loading a (frozen) Tensorflow model into memory...')
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('Model loaded')

    print('Loading label map...')
    '''
    Label maps map indices to category names, so that when our convolution network 
    predicts 5, we know that this corresponds to airplane. Here we use internal utility 
    functions, but anything that returns a dictionary mapping integers to appropriate 
    string labels would be fine
    '''
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('Label map loaded')

    print('Detection task started...')
    PATH_TO_TEST_IMAGES_DIR = 'images'
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'teste-{}.jpg'.format(i)) for i in range(1, 7) ]
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'teste-{}.jpg'.format(i)) for i in range(6, 8) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (24, 16)

    # test_display_image()


    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                plt.show()

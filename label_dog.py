import tensorflow as tf
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def label_dog(image):
    #^ Path to the image from command line

    image_file = tf.io.gfile.GFile(image, 'rb')
    #^ Image being read

    data = image_file.read()
    #^ Data from image file


    # Loads label file, strips off carriage return
    #classes = [line.rstrip() for line in tf.io.gfile.GFile("hot_dog_labels.txt")]
    classes = ["hotdog", "nothotdog"]

    # Unpersists graph from file
    with tf.io.gfile.GFile("hotdog-or-not-hotdog/hot_dog_graph.pb", 'rb') as inception_graph:
        definition = tf.compat.v1.GraphDef()
        definition.ParseFromString(inception_graph.read())
        _ = tf.import_graph_def(definition, name='')

    top_results = None

    with tf.compat.v1.Session() as session:
        tensor = session.graph.get_tensor_by_name('final_result:0')
        #^ Feeding data as input and find the first prediction
        result = session.run(tensor, {'DecodeJpeg/contents:0': data})

        top_results = result[0].argsort()[-len(result[0]):][::-1]
        for type in top_results:
            hot_dog_or_not = classes[type]
            score = result[0][type]
            print('%-20s : %.5f' % (hot_dog_or_not, score))

    return top_results

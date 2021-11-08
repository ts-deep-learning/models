'''
works on branch r1.13.0
This script needs to be placed in <path_to_tensorflow/models>/research/object_detection.
With paths to label file, frozen model and test images it will save the detections in the results directory.
'''

import numpy as np
import os
import sys
from tqdm import tqdm
import tensorflow as tf
import json
import argparse
import time
import cv2


parser = argparse.ArgumentParser(
    description="Script to get test_report.json from object det api models")
parser.add_argument('-m', default="", help="PATH_TO_FROZEN_GRAPH")
parser.add_argument('-th', default=0.5, type=float, help="SCORE_THRESHOLD")
parser.add_argument('-dr', default="", type=str, help="Data root directory")
parser.add_argument('-j', default="", type=str, help="Dataset JSON file path")
args = parser.parse_args()

PATH_TO_FROZEN_GRAPH = args.m
SCORE_THRESHOLD = args.th
DATA_ROOT = args.dr
JSON_PATH = args.j

PATH_TO_RESULTS = os.path.dirname(args.m)
OUTPUT_IMAGES_DIR = os.path.join(PATH_TO_RESULTS, 'images')

if not os.path.exists(PATH_TO_RESULTS):
    os.mkdir(PATH_TO_RESULTS)

if not os.path.exists(OUTPUT_IMAGES_DIR):
    os.mkdir(OUTPUT_IMAGES_DIR)

conf = tf.ConfigProto()
# conf.gpu_options.per_process_gpu_memory_fraction=0.4


def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def run_inference(graph):
    det_boxes = []
    det_scores = []
    det_labels = []
    true_boxes = []

    # Open the JSON and get annotations
    with open(JSON_PATH, 'r') as fp:
        gt_data = json.load(fp)
    annotations = gt_data['dataset'][0]['annotations']

    print('Total number of images : ', len(annotations))

    with graph.as_default():
        with tf.Session(config=conf) as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph(
                    ).get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # for file_name in tqdm(file_names):
            for annotation in tqdm(annotations):
                image_path = annotation['image_path']
                #image_path = os.path.join(DATA_ROOT, file_name)
                image_np = load_image_into_numpy_array(image_path)

                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                # adding det boxes, labels and scores
                index_to_keep = output_dict['detection_scores'] > SCORE_THRESHOLD
                mod_det_scores = output_dict['detection_scores'][index_to_keep].tolist(
                )
                mod_det_boxes = []
                height, width = image_np.shape[:2]
                for bbox in (output_dict['detection_boxes'][index_to_keep]):
                    ymin, xmin, ymax, xmax = bbox
                    mod_det_boxes.append(
                        [int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)])
                det_boxes.append(mod_det_boxes)
                det_scores.append(mod_det_scores)
                det_labels.append(
                    output_dict['detection_classes'][index_to_keep].tolist())
                gt_boxes = [bbox['box_coordinates'] for bbox in annotation['bbox_info']]
                true_boxes.append(gt_boxes)

    json_output = {
        'det_boxes': det_boxes,
        'det_labels': det_labels,
        'det_scores': det_scores,
        'true_boxes': true_boxes
    }
    filename = os.path.basename(
        args.m) + '_' + str(time.time()) + '_test_results.json'
    results_path = os.path.join(PATH_TO_RESULTS, filename)
    with open(results_path, 'w') as fp:
        json.dump(json_output, fp)


def main():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    run_inference(detection_graph)


if __name__ == '__main__':
    main()

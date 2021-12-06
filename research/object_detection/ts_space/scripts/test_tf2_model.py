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
from pprint import pprint


parser = argparse.ArgumentParser(
    description="Script to get test_report.json from object det api models")
parser.add_argument('-m', default="", help="PATH_TO_FROZEN_GRAPH")
parser.add_argument('-th', default=0.1, type=float, help="SCORE_THRESHOLD")
parser.add_argument('-dr', default="", type=str, help="Data root directory")
parser.add_argument('-j', default="", type=str, help="Dataset JSON file path")
args = parser.parse_args()

PATH_TO_SAVED_MODEL = args.m
SCORE_THRESHOLD = args.th
DATA_ROOT = args.dr
JSON_PATH = args.j

PATH_TO_RESULTS = os.path.dirname(args.m)
OUTPUT_IMAGES_DIR = os.path.join(PATH_TO_RESULTS, 'images')

if not os.path.exists(PATH_TO_RESULTS):
    os.mkdir(PATH_TO_RESULTS)

if not os.path.exists(OUTPUT_IMAGES_DIR):
    os.mkdir(OUTPUT_IMAGES_DIR)


def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def run_inference(detect_fn):
    det_boxes = []
    det_scores = []
    det_labels = []
    true_boxes = []

    # Open the JSON and get annotations
    with open(JSON_PATH, 'r') as fp:
        gt_data = json.load(fp)
    annotations = gt_data['dataset'][0]['annotations']

    print('Total number of images : ', len(annotations))

    # for file_name in tqdm(file_names):
    for annotation in tqdm(annotations):
        image_path = annotation['image_path']
        image_path = os.path.join(DATA_ROOT, image_path)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        output_dict = detect_fn(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

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
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    run_inference(detect_fn)


if __name__ == '__main__':
    main()

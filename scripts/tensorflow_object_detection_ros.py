#!/usr/bin/env python2
# -*-coding:utf-8 -*-

import numpy as np
import os
import sys
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
from collections import defaultdict
from io import StringIO
import matplotlib
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream

# for ros
import rospy
from std_msgs.msg import String 
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from copy import deepcopy
from amsl_recog_msgs.msg import ObjectInfoWithROI
from amsl_recog_msgs.msg import ObjectInfoArray


CONFIG_PATH = os.path.dirname(sys.path[0]) + '/config'+'/config.yml'
print("config path : %r" % (CONFIG_PATH))

with open(CONFIG_PATH, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
max_frames          = cfg['max_frames']
width               = cfg['width']
height              = cfg['height']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_interval        = cfg['det_interval']
det_th              = cfg['det_th']
model_name          = cfg['model_name']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
split_model         = cfg['split_model']
log_device          = cfg['log_device']

MODEL_PATH = os.path.dirname(sys.path[0]) + "/" + model_path
print("model path : %r" % (MODEL_PATH))

LABEL_PATH = os.path.dirname(sys.path[0]) + "/scripts" + "/" + label_path
print("label path : %r" % (LABEL_PATH))

# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'   
    if not os.path.isfile(MODEL_PATH):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')
 
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

def load_frozenmodel():
    print('Loading frozen model into memory')
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None
    
    else:
        # load a frozen Model and split it into GPU and CPU graphs
        # need to add gpu_options 
        input_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_memory_growth
        with tf.Session(graph=input_graph, config=config):
            score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node
                    
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
        
            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
              n = _node_name(node.name)
              name_to_node_map[n] = node
              edges[n] = [_node_name(x) for x in node.input]
              node_seq[n] = seq
              seq += 1
        
            for d in dest_nodes:
              assert d in name_to_node_map, "%s is not in graph" % d
        
            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
              n = next_to_visit[0]
              del next_to_visit[0]
              if n in nodes_to_keep:
                continue
              nodes_to_keep.add(n)
              next_to_visit += edges[n]
        
            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
        
            nodes_to_remove = set()
            for n in node_seq:
              if n in nodes_to_keep_list: continue
              nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])
        
            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
              keep.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
              remove.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            with tf.device('/gpu:0'):
              tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
              tf.import_graph_def(remove, name='')
              
        return detection_graph, score, expand

def load_labelmap():
    print('Loading label map')
    label_map = label_map_util.load_labelmap(LABEL_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


class ObjectDetection(object):
    def __init__(self, detection_graph, category_index, score, expand):
        self.image_sub = rospy.Subscriber("/image", Image, self.imageCallback, queue_size=10)
        self.image_pub = rospy.Publisher("/object_image", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("/object_info", ObjectInfoArray, queue_size=10)
        self.detection_graph = detection_graph
        self.category_index = category_index
        self.score = score
        self.expand = expand
        self.flag = False

    def imageCallback(self, image_msg):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
            self.frame_id = image_msg.header.frame_id
            self.flag = True
        except CvBridgeerror as e:
            print (e)

    def publish_boxes_and_labels(
                     self,
                     image,
                     boxes,
                     classes,
                     scores,
                     category_index,
                     max_boxes_to_draw=20,
                     min_score_thresh=.30):
        bbox_array = ObjectInfoArray()
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                box = tuple(boxes[i].tolist())
                label = classes[i]
                score = scores[i]
                ymin, xmin, ymax, xmax = box
                im_height = image.shape[0]
                im_width  = image.shape[1]
                (left, right, top, dowm) = (xmin * im_width, xmax * im_width,
                        ymin * im_height, ymax * im_height)
                bbox = ObjectInfoWithROI()
                bbox.Class = class_name
                bbox.probability = score
                bbox.xmin = left
                bbox.xmax = right
                bbox.ymin = top
                bbox.ymax = dowm
                bbox_array.object_array.append(bbox)
                print('%s:%.2f left:%d right:%d top:%d dowm:%d' % (class_name, score, left, right, top, dowm))
        bbox_array.header.stamp = rospy.Time.now()
        bbox_array.header.frame_id = self.frame_id
        self.bbox_pub.publish(bbox_array)

    def object_detection(self, sess):
        image_np = self.cv_image
        image_np = cv2.resize(image_np, (height, width))
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # actual Detection
        if not split_model:
            (boxes, scores, classes, num) = sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        else:
            # Split Detection in two sessions.
            (score, expand) = sess.run([self.score_out, self.expand_out], feed_dict={self.image_tensor: image_np_expanded})
            (boxes, scores, classes, num) = sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.score_in:score, self.expand_in: expand})

        # publish object info
        self.publish_boxes_and_labels(
            self.cv_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index)
        
        # Visualization of the results of a detection.
        if visualize:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
        
        image_height = self.cv_image.shape[0]
        image_width  = self.cv_image.shape[1]
        resize_image = cv2.resize(image_np, (image_width, image_height))
        pub_image = CvBridge().cv2_to_imgmsg(resize_image, "bgr8")
        self.image_pub.publish(pub_image)

    def main(self):
        rospy.init_node('detection_node')
        rate = rospy.Rate(30)
        per = rospy.get_param('~per', 0.8)
        dev = rospy.get_param('~dev', "0")

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
        config.gpu_options.allow_growth=allow_memory_growth

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph,config=config) as sess:
                # Define Input and Ouput tensors
                self.image_tensor        = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes     = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores    = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes   = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections      = self.detection_graph.get_tensor_by_name('num_detections:0')
                if split_model:
                    self.score_out       = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                    self.expand_out      = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                    self.score_in        = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                    self.expand_in       = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')

                    while not rospy.is_shutdown():
                        if self.flag:
                            self.object_detection(sess)
                            self.flag = False
                        rate.sleep()
                
def main():
    download_model()
    graph, score, expand = load_frozenmodel()
    category = load_labelmap()
    detection = ObjectDetection(graph, category, score, expand)
    detection.main()

if __name__ == '__main__':
    main()

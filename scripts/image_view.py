#!/usr/bin/env python
# -*-coding:utf-8

import rospy
from std_msgs.msg import String 
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from copy import deepcopy
from amsl_recog_msgs.msg import ObjectInfoWithROI
from amsl_recog_msgs.msg import ObjectInfoArray
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class view(object):
    def __init__(self):
        self.image_sub   = rospy.Subscriber("/image",  Image, self.imageCallback, queue_size=10)
        self.cluster_sub = rospy.Subscriber("/object_info", ObjectInfoArray, self.clusterCallback, queue_size=10)  
        self.image_pub = rospy.Publisher("/object_image", Image, queue_size=10)
        self.flag = False

        self.list = ['person', 'car']

    def imageCallback(self, image_msg):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
            self.frame_id = image_msg.header.frame_id
            self.flag = True
        except CvBridgeerror as e:
            print (e)

    def clusterCallback(self, cluster_msg):
        if self.flag:
            print("ALL GREEN")
            image = self.cv_image.copy()
            for i in range(len(cluster_msg.object_array)):
                bbox = cluster_msg.object_array[i]
                if bbox.Class in self.list:
                    print("class:%s score:%5.2f xmin:%d ymin:%d xmax:%d ymax:%d" % (bbox.Class, bbox.probability, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))
                    if bbox.Class == self.list[0]:
                        cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 200, 0), 5)
                    elif bbox.Class == self.list[1]:
                        cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 0, 200), 5)
           
            pub_image = CvBridge().cv2_to_imgmsg(image, "bgr8")
            self.image_pub.publish(pub_image)

    def main(self):
        rospy.init_node("image_view")
        rospy.spin()

def main():
    viewer = view()
    viewer.main()

if __name__ == '__main__':
    main()

# tensorflow_object_detection_tx2
tensoflow_object_detectio for tx2
These codes correspond to ros and are optimized to speed up processing


## Requirements
1. tensorflow v1.8
2. ros
3. [amsl_recog_msgs](https://github.com/Sadaku1993/amsl_recog_msgs)

## How to Set Up
1. Download tensorflow-gpu(v1.8)
2. Download ros(kinetic)
3. Dounload [amsl_recog_msgs](https://github.com/Sadaku1993/amsl_recog_msgs)
```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/Sadaku1993/amsl_recog_msgs
$ cd ..
$ catkin_make
```
4. Download [tensorflow_object_detection_tx2](https://github.com/Sadaku1993/tensorflow_object_detection_tx2)
```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/Sadaku1993/tensorflow_object_detection_tx2
$ cd ..
$ catkin_make
```

## How to Use(usb_cam)
```
$roscore
$roslaunch usb_cam usb_cam-test.launch
$rosrun tensorflow object_detection_tx2 tensorflow_object_detection_ros.py
```

prease check topic name
tensorflow_object_detection_tx2/scripts/tensorflow_object_detection_ros.py
```python
self.image_sub = rospy.Subscriber("/image", Image, self.imageCallback, queue_size=10)
self.image_pub = rospy.Publisher("/object_image", Image, queue_size=10)
self.bbox_pub = rospy.Publisher("/object_info", ObjectInfoArray, queue_size=10)
```

# subscribe pointcloud and odometry, images, publish the synchronized pointcloud and odometry, images

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from collections import deque
from fruit_counting.msg import SyncedPcOdomImg
import numpy as np
import sys
import os 
import utils

class pc_odom_img_sync:
    def __init__(self, pc_topic, odom_topic, img_topic):
        self.pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)
        self.odom_sub = message_filters.Subscriber(odom_topic, Odometry)
        self.img_sub = message_filters.Subscriber(img_topic, Image)
        self.rgb_pose_sync = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.img_sub], queue_size=50, slop=0.02)
        self.rgb_pose_sync.registerCallback(self.img_pose_callback)
        self.pc_odom_img_pub = rospy.Publisher('/pc_odom_img_sync', SyncedPcOdomImg, queue_size=10)
        self.pc_queue = deque(maxlen=10)

    def pc_callback(self, pc_msg):
        self.pc_queue.append(pc_msg)

    def img_pose_callback(self, odom_msg, rgb_msg):
        cur_accumulated_pc_msg_list = list(self.pc_queue)
        if cur_accumulated_pc_msg_list is None or len(cur_accumulated_pc_msg_list) == 0:
            print("no point cloud in the queue, skip")
            return
        print("lastest pc time stamp and rgb time stamp difference: ", cur_accumulated_pc_msg_list[-1].header.stamp.to_sec() - rgb_msg.header.stamp.to_sec())
        print("earliest pc time stamp and rgb time stamp difference: ", cur_accumulated_pc_msg_list[0].header.stamp.to_sec() - rgb_msg.header.stamp.to_sec())
        print("sync difference in sec: ", rgb_msg.header.stamp.to_sec() - odom_msg.header.stamp.to_sec())

        stamp = rgb_msg.header.stamp
        frame_id = "camera_init"
        # publish the synchronized pointcloud, odometry and image
        sync_msg = SyncedPcOdomImg()
        sync_msg.header.stamp = stamp
        sync_msg.header.frame_id = frame_id
        
        pc_xyz = []
        for pc_msg in cur_accumulated_pc_msg_list:
            field_corrected_pc_msg = utils.modifyPcMsgFields(pc_msg) # in world frame
            pc_xyz_numpy = utils.pcMsg2NumpyArray(field_corrected_pc_msg)
            pc_xyz.append(pc_xyz_numpy)
        pc_xyz = np.vstack(pc_xyz)
        sync_msg.pc = utils.numpyXYZ2PcMsg(pc_xyz, stamp, frame_id)
        sync_msg.odom = odom_msg
        sync_msg.img = rgb_msg
        self.pc_odom_img_pub.publish(sync_msg)

if __name__ == "__main__":
    rospy.init_node('pc_odom_img_sync_node')
    print("pc_odom_img_sync_node started")
    rgb_topic = "/spinnaker/image_raw"
    pc_topic = "/cloud_registered"
    odom_topic = "/Odometry"
    pc_odom_img_sync(pc_topic=pc_topic, odom_topic=odom_topic, img_topic=rgb_topic)
    rospy.spin()



        

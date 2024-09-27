#!/usr/bin/env python

import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
import ros_numpy
import cv2
import yaml
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import h5py

CLOUD_REGISTERED_FIELD = [
    {'name': 'x', 'offset': 0, 'datatype': 7, 'count': 1},
    {'name': 'y', 'offset': 4, 'datatype': 7, 'count': 1},
    {'name': 'z', 'offset': 8, 'datatype': 7, 'count': 1},
    {'name': 'normal_x', 'offset': 16, 'datatype': 7, 'count': 1},
    {'name': 'normal_y', 'offset': 20, 'datatype': 7, 'count': 1},
    {'name': 'normal_z', 'offset': 24, 'datatype': 7, 'count': 1},
    {'name': 'intensity', 'offset': 32, 'datatype': 7, 'count': 1},
    {'name': 'curvature', 'offset': 36, 'datatype': 7, 'count': 1}
]

### hdf5 relevant files
def get_data_from_hdf5(hdf5_file, grp_name, idx = None):
    '''
    hdf5_file: h5py.File
    idx: a list of indices, int
    '''
    data = []
    prefix = 'tracklet_'

    f = h5py.File(hdf5_file, 'r')
    if idx is None:
        num_tracklets = len(f.keys())
        i_array = np.arange(num_tracklets)
    else:
        i_array = np.array(idx)
    f.close()


    with h5py.File(hdf5_file, 'r') as f:
        if grp_name == 'mask':
            for i in i_array:
                tracklet = f[prefix + str(i)]
                mask = []
                num_masks = len(tracklet['mask'].keys())
                for k in range(num_masks):
                    mask.append(tracklet['mask'][str(k)][:])
                mask = np.array(mask, dtype=object)
                data.append(mask)
        else:
            for i in i_array:
                tracklet = f[prefix + str(i)]
                data.append(tracklet[grp_name][:])
    f.close()
    return data


def load_params(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content



def visSemanticMask(semantic_mask, label_list):
    '''
    semantic_mask: np.array(height, width), each element is the class label
    '''
    # visualize the semantic mask
    colorized_mask = np.full((semantic_mask.shape[0], semantic_mask.shape[1]), 255,dtype=np.uint8) # white background
    colorized_mask[semantic_mask != 0] = 0 # black mask

    return colorized_mask

def modifyPcMsgFields(pc_msg):
    '''
    pc_msg: PointCloud2 message
    return: PointCloud2 message with correct PointField
    '''
    original_fields_list = pc_msg.fields
    assert len(original_fields_list) == len(CLOUD_REGISTERED_FIELD)
    corrected_fields_list = []
    for i in range(len(original_fields_list)):
        corrected_fields_list.append(PointField(name=CLOUD_REGISTERED_FIELD[i]['name'], offset=CLOUD_REGISTERED_FIELD[i]['offset'], datatype=CLOUD_REGISTERED_FIELD[i]['datatype'], count=CLOUD_REGISTERED_FIELD[i]['count']))
    pc_msg.fields = corrected_fields_list
    return pc_msg

def numpyXYZ2PcMsg(pc_xyz, stamp, frame_id):
    '''
    pc_xyz: np.array(num_points, 3)
    stamp: rospy.Time
    frame_id: str
    return: PointCloud2 message
    '''
    pc_array = np.zeros(len(pc_xyz), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    pc_array['x'] = pc_xyz[:, 0]
    pc_array['y'] = pc_xyz[:, 1]
    pc_array['z'] = pc_xyz[:, 2]
    pc_array['intensity'] = np.zeros(len(pc_xyz), dtype=np.float32)

    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp, frame_id)
    return pc_msg

def numpyXYZL2PcMsg(pc_xyzl, stamp, frame_id):
    '''
    pc_xyzl: np.array(num_points, 4)
    stamp: rospy.Time
    frame_id: str
    return: PointCloud2 message
    '''
    pc_array = np.zeros(len(pc_xyzl), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    pc_array['x'] = pc_xyzl[:, 0]
    pc_array['y'] = pc_xyzl[:, 1]
    pc_array['z'] = pc_xyzl[:, 2]
    pc_array['intensity'] = pc_xyzl[:, 3]

    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp, frame_id)
    return pc_msg


def numpyXYZLBGR2PcMsg(pc_xyzlbgr, stamp, frame_id):
    '''
    pc_xyzlbgr: np.array(num_points, 7)
    stamp: rospy.Time
    frame_id: str
    return: PointCloud2 message
    
    '''
    pc_array = np.zeros(len(pc_xyzlbgr), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('b', np.uint8),
        ('g', np.uint8),
        ('r', np.uint8),
        ('rgb', np.uint32),
    ])
    pc_array['x'] = pc_xyzlbgr[:, 0]
    pc_array['y'] = pc_xyzlbgr[:, 1]
    pc_array['z'] = pc_xyzlbgr[:, 2]
    pc_array['intensity'] = pc_xyzlbgr[:, 3]
    pc_array['b'] = pc_xyzlbgr[:, 4]
    pc_array['g'] = pc_xyzlbgr[:, 5]
    pc_array['r'] = pc_xyzlbgr[:, 6]
    pc_array['rgb'] = (pc_array['r'] << 16) | (pc_array['g'] << 8) | pc_array['b']
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp, frame_id)
    return pc_msg

def pcMsg2NumpyXYZL(pc_msg):
    '''
    pc_msg: PointCloud2 message
    return: np.array(num_points, 4) xyzl
    '''
    recordArray = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg,squeeze=True)
    pc_xyz = ros_numpy.point_cloud2.get_xyz_points(recordArray, dtype=float) # num_points by 3
    pc_xyzl = np.hstack((pc_xyz, recordArray['intensity'].reshape(-1, 1))) # num_points by 4
    return pc_xyzl

def pcMsg2NumpyXYZLBGR(pc_msg):
    '''
    pc_msg: PointCloud2 message
    return: np.array(num_points, 7) xyzlbgr
    '''
    recordArray = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg,squeeze=True)
    x = recordArray['x']
    y = recordArray['y']
    z = recordArray['z']
    intensity = recordArray['intensity']
    b = recordArray['b']
    g = recordArray['g']
    r = recordArray['r']
    pc_xyzlbgr = np.vstack((x, y, z, intensity, b, g, r)).T
    return pc_xyzlbgr

def PoseMsg2NumpyArray(pose_msg):
    '''
    pose_msg: geometry_msgs/Pose Message
    '''
    return ros_numpy.geometry.pose_to_numpy(pose_msg)

def NumpyArray2PoseMsg(pose_array):
    '''
    pose_array: np.array(4, 4)
    '''
    return ros_numpy.geometry.numpy_to_pose(pose_array)

def transformPcMsg(tf, pc_msg):
    '''
    tf: 4x4 transformation matrix, usually from the pose message in world frame
    pc_msg: PointCloud2 message in world frame
    return: PointCloud_xyz in local frame in the form of np.array(num_points, 3)
    '''
    tf_from_world_to_local = np.linalg.inv(tf)
    # read the pointcloud from the pc_msg
    recordArray = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg,squeeze=True)
    pc_xyz = ros_numpy.point_cloud2.get_xyz_points(recordArray, dtype=float) # num_points by 3

    # transform the pointcloud using tf
    pc_xyz1 = np.hstack((pc_xyz, np.ones((pc_xyz.shape[0], 1)))) # num_points by 4
    local_pc_xyz1 = tf_from_world_to_local @ pc_xyz1.T # 4xnum_points
    return (local_pc_xyz1.T)[:, :3]

def transformPcNumpyArray(tf, pc_xyz):
    '''
    pc_xyz: np.array(num_points, 3)
    tf: 4x4 transformation matrix. In our use case, the transformation matrix is from lidar frame to camera frame
    '''
    pc_xyz1 = np.hstack((pc_xyz, np.ones((pc_xyz.shape[0], 1)))) # num_points by 4
    local_pc_xyz1 = tf @ pc_xyz1.T # 4xnum_points
    return (local_pc_xyz1.T)[:, :3]

def transformPcNumpyArray2(tf, pc_xyzl):
    '''
    pc_xyzl: np.array(num_points, 4), keep the intensity value not changed
    tf: 4x4 transformation matrix. In our use case, the transformation matrix is from lidar frame to camera frame
    '''
    pc_xyz1 = np.hstack((pc_xyzl[:, :3], np.ones((pc_xyzl.shape[0], 1)))) # num_points by 4
    local_pc_xyz1 = tf @ pc_xyz1.T # 4xnum_points
    local_pc_xyzl = np.hstack(((local_pc_xyz1.T)[:, :3], pc_xyzl[:, 3].reshape(-1, 1)))
    return local_pc_xyzl

def pcMsg2NumpyArray(pc_msg):
    '''
    pc_msg: PointCloud2 message
    return: np.array(num_points, 3)
    '''
    recordArray = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg,squeeze=True)
    pc_xyz = ros_numpy.point_cloud2.get_xyz_points(recordArray) # num_points by 3
    return pc_xyz

def debugPc2depthImg(pc_xyz, calibration_matrix, distorsion_coefficients, image_size, original_rgb_img=None, frame_id="camera", stamp=None):
    '''
    pc_xyz: np.array(num_points, 3) in camera frame
    projection_matrix: 3x3
    image_size: (width, height)
    original_rgb_img: np.array(height, width, 3), the original rgb image
    stamp: image msg stamp
    return: depth image
    '''
    xy = cv2.projectPoints(pc_xyz, np.zeros((3, 1)), np.zeros((3, 1)), calibration_matrix, distorsion_coefficients)[0] # image points n by 1 by 2
    xy = np.squeeze(xy) # n by 2 (x is col index, y is row index)
    xy = np.round(xy).astype(int) # n by 2
    depth_img = np.full(image_size, -1, dtype=np.float32)
    effective_xy_mask = (xy[:, 0] >= 0) & (xy[:, 0] < image_size[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < image_size[0])
    effective_xy = xy[effective_xy_mask]
    distance = np.linalg.norm(pc_xyz, axis=1)
    effective_range = distance[effective_xy_mask]
    depth_img[effective_xy[:, 1], effective_xy[:, 0]] = effective_range

    effective_pc_xyz = pc_xyz[effective_xy_mask]
    effective_bgr = original_rgb_img[effective_xy[:, 1], effective_xy[:, 0]] # m by 3
    effective_rgb = effective_bgr[:, [2, 1, 0]] # m by 3
    pc_rgb_pair = np.hstack((effective_pc_xyz, effective_rgb))
    r = pc_rgb_pair[:, 3].astype(np.uint32)
    g = pc_rgb_pair[:, 4].astype(np.uint32)
    b = pc_rgb_pair[:, 5].astype(np.uint32)
    rgb = (r << 16) | (g << 8) | b
    pc_rgb_pair = np.hstack((pc_rgb_pair, rgb.reshape(-1, 1)))
    colorized_pc_msg = None
    if original_rgb_img is not None:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id  # Provide the frame ID for the point cloud
        # create an array that each column is of different data type
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('rgb', np.uint32)]
        points = np.zeros(pc_rgb_pair.shape[0], dtype=dtype)
        for i, field in enumerate(dtype):
            points[field[0]] = pc_rgb_pair[:, i]
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('r', 12, PointField.UINT8, 1),
                  PointField('g', 13, PointField.UINT8, 1),
                  PointField('b', 14, PointField.UINT8, 1),
                  PointField('rgb', 16, PointField.UINT32, 1)]
        colorized_pc_msg = pc2.create_cloud(header, fields, points)

    return depth_img, colorized_pc_msg

def draw_squares(image_array, coordinates, square_color=(255, 0, 0), square_thickness=2, square_size=3):
    
    # Calculate the corner coordinates of all squares
    x1 = coordinates[:, 0] - square_size // 2
    y1 = coordinates[:, 1] - square_size // 2
    x2 = coordinates[:, 0] + square_size // 2
    y2 = coordinates[:, 1] + square_size // 2
    # limit the coordinates within the image
    x1 = np.clip(x1, 0, image_array.shape[1]-1)
    y1 = np.clip(y1, 0, image_array.shape[0]-1)
    x2 = np.clip(x2, 0, image_array.shape[1]-1)
    y2 = np.clip(y2, 0, image_array.shape[0]-1)
    # make sure the coordinates are integers
    x1 = x1.astype(int)
    y1 = y1.astype(int)
    x2 = x2.astype(int)
    y2 = y2.astype(int)
    
    # Create a mask for all the squares but don't use for loop
    mask = np.zeros_like(image_array)
    for i in range(square_size):
        for j in range(square_size):
            # limit the coordinates within the image
            new_y1 = np.clip(y1 + i, 0, image_array.shape[0]-1)
            new_x1 = np.clip(x1 + j, 0, image_array.shape[1]-1)
            image_array[new_y1, new_x1] = square_color

    return image_array
    
    

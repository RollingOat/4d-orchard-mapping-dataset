# 4d-orchard-mapping-dataset

This repository holds scripts and tutorials on using the dataset presented in the paper **4D Metric-Semantic Mapping for Persistent Orchard Monitoring: Method and Dataset**

Project website: [https://4d-metric-semantic-mapping.org/overview/](https://4d-metric-semantic-mapping.org/overview/)

## Scripts
* manual_counting.py: helper script to get the ground truth of fruit count
* pc_odom_rgb_sync_node.py: sychronize pointcloud, rgb images, and odometry in the bag
* PrepareImagesForTraining.py: Extract images and labels from a folder, convert labels in json format to yolo required format, and automatically divide them into training and validation set
* spinnaker_exposure.py: adjust the exposure level of rosbag images as some of them looks dark
* utils.py: helper functions
* extract_image.py: get the images from rosbags 

## How to use [Faster-lio](https://github.com/gaoxiang12/faster-lio) to get point cloud from lidar packets

It's recommended to use ROS as our dataset is recorded as ros bags.

Dependency: 
* ROS1: Faster-lio is implemented to use with ROS. If you don't have experience in ROS, refer to [ROS](https://wiki.ros.org/Installation) for more information on installation and tutorial. 
* [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html): build ros workspace
* dependencies of [faster-lio](https://github.com/gaoxiang12/faster-lio)
* dependencies of [ouster-ros](https://github.com/ouster-lidar/ouster-ros)

After installing ROS1, following the steps below:

1. create ros workspace

    ```
    cd ~
    mkdir ws
    cd ws
    mkdir src
    ```

2. faster-lio

    ```
    cd ~/ws/src
    git clone https://github.com/gaoxiang12/faster-lio.git
    ```

3. ouster-ros

    ```
    cd ~/ws/src
    git clone https://github.com/ouster-lidar/ouster-ros.git
    ```


4. build the workspace
   ```
    catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
   ```

5. config files
   1. Put the all the files in the launch folder into faster-lio's launch folder. 
   2. Put ouster128.yaml inside faster-lio's config
   3. Change the value of *metadata* to be the path to the metadata provided.

6. launch the ouster driver and faster-lio
   ```
    roslaunch faster_lio mapping_ouster128_with_driver.launch
   ```


## How to do calibration using calibration bags

1. use the same workspace, compile kalibr
    ```
    cd ~/ws/src
    git clone https://github.com/ethz-asl/kalibr.git
    catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```

2. run kalibr
    ```
    rosrun kalibr kalibr_calibrate_cameras --target path_to_apriltag.yaml --models pinhole-radtan --topics image_topic --bag path_to_your_bag --bag-freq 4.0
    ```

## Questions
Please feel free to open an issue if there are any questions or bugs.

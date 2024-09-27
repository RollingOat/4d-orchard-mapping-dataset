
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
from spinnaker_exposure import gamma_correction
from spinnaker_exposure import perform_adaptive_histeq
home = os.path.expanduser("~")

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-04-10-15-00_image_processed940_to_990.bag" 
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-04-10-15-00_image_processed1090_to_1140.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-19-11-22-12_image_processed110_to_170.bag"
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-19-11-22-12_image_processed1380_to_1430.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apples_both_rows_2024-05-24-10-45-08_image_processed1090_to_1140.bag"
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apples_both_rows_2024-05-24-10-45-08_image_processed1200_to_1260.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_row_2024-08-13-12-08-14_image_processed1250_to_1300.bag"
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_row_2024-08-13-12-08-14_image_processed160_to_220.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_both_row_2024-06-12-12-08-14_image_processed1260_to_1320.bag"
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_both_row_2024-06-12-12-08-14_image_processed1410_to_1520.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_2024-04-16-12-47-42_image_processed130_to_170.bag"
# bag_path_2 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_2024-04-16-12-47-42_image_processed950_to_990.bag"

# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_2024-08-13-12-08-14_image_processed800_to_1300.bag"
# bag_path_1 = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_2024-08-13-12-08-14_image_processed2000_to_2500.bag"
# bag_path = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-19-11-22-12_image_processed1290_to_1400.bag"
# bag_path = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard-apples-both-row_2024-07-04-10-15-00_image_processed_image_processed1090_to_1290.bag"
# bag_path = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apple_both_row_2024-06-12-12-08-14_image_processed1420_to_1790.bag"
bag_path = home+"/bags/fruit_counting/synced_pc_odom_img_hands-on-earth-orchard_apples_both_rows_2024-05-24-10-45-08_image_processed1200_to_1400.bag"

Bayered = False

# image_folder_name = bag_path.split("/")[-1].split(".")[0].split("_")[-5]
# image_folder_path = home+"/bags/fruit_counting/image_labelling_data/"+image_folder_name+"_apple/any_labeling"
# image_folder_path = home+"/bags/aug-13th-row1-sun-side"
# image_folder_path = home+"/bags/july-19th-row1-sun-side"
# image_folder_path = home+"/bags/july-4th-row1-sun-side"
# image_folder_path = home+"/bags/june-12th-row1-sun-side"
image_folder_path = home+"/bags/may-24th-row1-sun-side"
print("image_folder_path: ", image_folder_path)
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)
i = 0
bridge = CvBridge()
with rosbag.Bag(bag_path, 'r') as bag:
    message_count = bag.get_message_count()
    for topic, msg, t in bag.read_messages():
        i += 1
        print("\n")
        print("Processing message: ", i, "/", message_count)
        image_msg = msg.img
        if Bayered:
            image_msg = bridge.imgmsg_to_cv2(image_msg)
            image = cv2.cvtColor(image_msg, cv2.COLOR_BayerBG2BGR)
        else:   
        # save the image to a folder
            image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # rotate the image counter-clockwise 90 degree
        image = np.rot90(image)
        # save the image
        cv2.imwrite(image_folder_path+"/apples_"+str(i)+".png", image)

# with rosbag.Bag(bag_path_2, 'r') as bag:
#     message_count = bag.get_message_count() + message_count
#     for topic, msg, t in bag.read_messages():
#         i += 1
#         print("\n")
#         print("Processing message: ", i, "/", message_count)
#         image_msg = msg.img
#         if Bayered:
#             image_msg = bridge.imgmsg_to_cv2(image_msg)
#             image = cv2.cvtColor(image_msg, cv2.COLOR_BayerBG2BGR)
#         # save the image to a folder
#         else:
#             image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
#         # rotate the image counter-clockwise 90 degree
#         image = np.rot90(image)
#         # save the image
#         cv2.imwrite(image_folder_path+"/apples_"+str(i)+".png", image)

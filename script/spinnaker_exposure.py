import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path
from skimage import exposure
import os
from tqdm import tqdm


def enhance_contrast(image, pmin=0, pmax=85):
    """
    Enhance image contrast
    :param image:
    :param pmin:
    :param pmax:
    :return:
    """
    vmin, vmax = np.percentile(image, (pmin, pmax))
    return exposure.rescale_intensity(image, in_range=(vmin, vmax))

def perform_adaptive_histeq(image, clip_limit=0.01, kernel_size=8):
    """
    Perform adaptive histogram equalization on image
    :param image:
    :param clip_limit:
    :param tile_grid_size:
    :return:
    """
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)

def perform_rgb_with_LAB_CLAHE(image, clip_limit=3.5, tile_grid_size=(8, 8)):
    """
    Perform CLAHE on RGB image
    :param image:
    :param clip_limit:
    :param tile_grid_size:
    :return:
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def perform_rgb_with_RGB_CLAHE(image, clip_limit=3.5, tile_grid_size=(8, 8)):
    """
    Perform CLAHE on RGB image
    :param image:
    :param clip_limit:
    :param tile_grid_size:
    :return:
    """
    r, g, b = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)

    return cv2.merge((r, g, b))

def log_correction(image):
    """
    Apply log correction to the image
    :param image:
    :return:
    """
    return exposure.adjust_log(image, gain=1, inv=False)

def gamma_correction(image, gamma=0.8):
    """
    Apply gamma correction to the image
    :param image:
    :param gamma:
    :return:
    """
    return exposure.adjust_gamma(image, gamma=gamma)

def sigmoid_correction(image, cutoff=0., gain=1, inv=False):
    """
    Apply sigmoid correction to the image
    :param image:
    :param cutoff:
    :param gain:
    :param inv:
    :return:
    """
    return exposure.adjust_sigmoid(image, cutoff=cutoff, gain=gain, inv=inv)

bridge = CvBridge()
exposure_corrected_pub = rospy.Publisher('/spinnaker/image_raw_corrected', Image, queue_size=10)

def adjust_exposure(image, gamma=1.0):
    # Apply gamma correction
    adjusted_image = np.uint8(cv2.pow(image / 255.0, gamma) * 255)
    return adjusted_image

def image_callback(msg):
    # Convert ROS Image message to OpenCV image
    image = bridge.imgmsg_to_cv2(msg)
    cv_image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
    image_np = np.array(cv_image, dtype=np.uint8)
    # enhace contrast
    exposed_image = gamma_correction(image_np, gamma=0.3)
    enhanced_image = enhance_contrast(exposed_image)
    # convert pencv image to ROS Image message
    enhanced_image_msg = bridge.cv2_to_imgmsg(enhanced_image, encoding='bgr8')
    # Publish the enhanced image
    exposure_corrected_pub.publish(enhanced_image_msg)

def ros_main():
    rospy.init_node('adjust_exposure')
    print("node launched")
    rospy.Subscriber('/spinnaker/image_raw', Image, image_callback)
    rospy.spin()

def folder_main():
    print(cv2.__version__)
    to_save = True
    
    image_path = "" # replace the image_path to be your own
    
    save_path = "" # replace the path to be your own
    if not os.path.exists(save_path):
        # create the save path if it does not exist
        os.makedirs(save_path)
    image_names = []
    for image in sorted(os.listdir(image_path)):
        if image.endswith(".png"):
            image_names.append(image)
    image_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    i = 0
    for image in tqdm(image_names):    
        image = Path(image_path) / image
        image = cv2.imread(str(image))
        adjusted_image = enhance_contrast(image)
        adjusted_image = gamma_correction(adjusted_image, gamma=0.6)
        # uncomment or comment out the following code at your own choice to try different processing methods
        # adjusted_image = log_correction(image)
        # adjusted_image = sigmoid_correction(image)
        # adjusted_image2 = perform_rgb_with_RGB_CLAHE(image)
        # adjusted_image2 = perform_adaptive_histeq(image, clip_limit=0.05, kernel_size=image.shape[0]//8)
        adjusted_image2 = gamma_correction(adjusted_image, gamma=0.8)

        if to_save:
            save_name = Path(save_path) / image_names[i]
            # adjusted_image2_255 = np.uint8(adjusted_image2 * 255)
            # cv2.imwrite(str(save_name), adjusted_image2_255)
            cv2.imwrite(str(save_name), adjusted_image)

        # adjust the image window size to fit the screen
        if not to_save:
            cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Adjusted Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Adjusted Image 2", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Original Image", 800, 800)
            cv2.resizeWindow("Adjusted Image", 800, 800)
            cv2.resizeWindow("Adjusted Image 2", 800, 800)
            # put the image window in the top left corner and the adjusted image window in the top right corner
            cv2.moveWindow("Original Image", 0, 0)
            cv2.moveWindow("Adjusted Image", 800, 0)
            cv2.moveWindow("Adjusted Image 2", 1600, 0)
            cv2.imshow("Original Image", image)
            cv2.imshow("Adjusted Image", adjusted_image)
            cv2.imshow("Adjusted Image 2", adjusted_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        i += 1
        print(f"Processed image {i}")

if __name__ == '__main__':
    use_folder_image = False
    if use_folder_image:
        folder_main()
    else:
        ros_main()

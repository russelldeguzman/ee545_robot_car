#!/usr/bin/env python

import cv2
import time
import sys
import rospy
import rosbag
import numpy as np
import scipy.signal
import utils as Utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
import math
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError


IMAGE_TOPIC = '/camera/color/image_raw'
CMD_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/nav_0' # The topic to publish controls to
IMGPUB_TOPIC = '/cv_module/image_op'

once = True
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):

    assert img.shape[2] == 3

    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for i, channel in enumerate(channels):
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        if i == 0:
            low_val  = flat[int(math.floor(n_cols * half_percent))]
            high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        # print "Lowval: ", low_val
        # print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)




class RBFilter:

    def __init__(self, min_angle, max_angle, angle_incr):

        # Storing Params if needed
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_incr = angle_incr
        self.angles = np.arange(min_angle, max_angle, angle_incr)

        self.bridge = CvBridge()
        #Publisher, Subscribers
        self.img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_cb)
        self.img_pub = rospy.Publisher(IMGPUB_TOPIC, Image, queue_size= 10)

        self.hsv_img = None
        self.mask_red = None
        self.mask_blue = None

    def is_object_present(self, mask, threshold):
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours ) == 0:
            return False, 0, 0

        # Find the largest contour
        largest_contour = max(contours, key=lambda x:cv2.contourArea(x))
        if cv2.contourArea(largest_contour) > threshold:
            M = cv2.moments(largest_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return True, cX, cY

        return False, 0, 0

    def hsv_thresh(self, hsv_samp, tol=[5, 25, 25]):
        # Define HSV Parameter bounds for CV2
        CV_MAX_HSV = [180, 255, 255]
        CV_MIN_HSV = [0, 0, 0]

        HUE_SHIFT = tol[0] + 1
        # HUE_SHIFT = 0
        # Convert PS HSV form [0-360, 0-100, 0-100] to CV2 form [0-180, 0-255, 0-255]
        hsv_cv = np.zeros_like(hsv_samp)
        hsv_cv[0] = int(round(hsv_samp[0]/2)) - HUE_SHIFT
        hsv_cv[1] = int(round((hsv_samp[1]/100.0)*255))
        hsv_cv[2] = int(round((hsv_samp[2]/100.0)*255))

        lower_hsv = np.array(np.maximum(CV_MIN_HSV, hsv_cv - tol), dtype=int)
        # lower_hsv = np.array([0, int(round(0.2*255)), int(round(0.95*255))], dtype=int)
        upper_hsv = np.array(np.minimum(CV_MAX_HSV, hsv_cv + tol), dtype=int)
        # upper_hsv = np.array([5, int(round(0.8*255)), 255], dtype=int)

        # This is a hacky way of getting around the max red hue value wrapping around 0
        hue_shift_img = self.hsv_img.copy()
        hue_shift_img[:,:,0] -= HUE_SHIFT
        hue_shift_img = np.array(np.where(hue_shift_img < 0, 180 + hue_shift_img, hue_shift_img), dtype=int)
        hsv_mask = cv2.inRange(hue_shift_img, lower_hsv, upper_hsv)

        return hsv_mask

    def area_check(self, rgb_img):
        blue_samp = np.mean([[208, 90, 90], [208, 70, 88], [207,100,96], [204,73,100]], axis=0)  # NOTE: these are in Photoshop HSV and must be converted to CV2's weird ranges
        red_samp = np.mean([[355, 40, 100], [355, 35, 100], [338, 25, 100], [356,58,100]], axis=0) # NOTE: these are in Photoshop HSV and must be converted to CV2's weird ranges
        # red_rollover =[2, 28,100]

        # BLUE_TOL = [10, 50, 20]   # Blue hue tolerance
        # RED_TOL = [10, 50, 20]  # Red hue tolerance

        self.hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        self.mask_red = self.hsv_thresh(red_samp, RED_TOL)
        self.mask_blue = self.hsv_thresh(blue_samp, BLUE_TOL)

        cv2.imshow("blue_mask", self.mask_blue)
        cv2.imshow("red_mask", self.mask_red)

        # Adjust threshold based on output
        square_area_threshold = 50
        # TODO - it might be better if we adjust this threshold dynamically based on either depth info from the RealSense or our best guess as to distance-to-target based on ParticleFilter
        # Dead simple version - do it as a function of the centroid height?

        is_red_square_present, x, y = self.is_object_present(self.mask_red, square_area_threshold)

        cv2.circle(rgb_img, (x, y), 7, (255, 255, 255), -1)

        is_blue_square_present, x, y = self.is_object_present(self.mask_blue, square_area_threshold)

        cv2.circle(rgb_img, (x, y), 7, (255, 255, 255), -1)

        #Now order of precedence would be for red over blue

        # if is_blue_square_present and is_red_square_present:
        #     print "Both Blue and red present"

        # if is_blue_square_present:
        #     print "Blue present"

        # if is_red_square_present:
        #     print "Red present"

        # cv2.imshow("HSV Image", hsv)
        # cv2.imshow("BGR8 Image", rgb_img)

    def calc_turn_angle ( self, x, y, img_width, turn_towards ):
        turn_angle = 0
        center = int(img_width / 2)
        pixels_per_bin = img_width / len(self.angles)
        if turn_towards: # we're trying to head towards a blue square
            turn_angle = self.angles[int( x / pixels_per_bin )]
        else: # We need to get away from a red square
            if x < center: # red to the left
                turn_angle = self.angles[len(self.angles) - 1] # we want to turn to the right ASAP
            else: # red to the right
                turn_angle = self.angles[0] # we want to turn to the left ASAP
        return turn_angle

    def image_cb(self,data):
        # print('callback')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = simplest_cb(cv_image, 7)

        self.area_check(cv_image)

        cv2.imshow("BGR8 Image", cv_image)

        cv2.waitKey(3)

        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('cv_module', anonymous=True)

    min_angle = rospy.get_param('~min_angle')# Default val: -0.34
    max_angle = rospy.get_param('~max_angle')# Default val: 0.341
    angle_incr = rospy.get_param('~angle_incr')# Starting val: 0.34/3 (consider changing the denominator)
    angle_incr /= 3
    im_filter = RBFilter(min_angle, max_angle, angle_incr)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down')

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

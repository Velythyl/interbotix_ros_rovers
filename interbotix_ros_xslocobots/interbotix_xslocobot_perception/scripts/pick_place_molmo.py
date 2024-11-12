import math
import time
from interbotix_xs_modules.locobot import InterbotixLocobotXS

# This script uses the perception pipeline to pick up objects and place them in some virtual basket on the left side of the robot
# It also uses the AR tag on the arm to get a better idea of where the arm is relative to the camera (though the URDF is pretty accurate already).
#
# To get started, open a terminal and type...
# 'roslaunch interbotix_xslocobot_control xslocobot_python.launch robot_model:=locobot_wx200 use_perception:=true'
# Then change to this directory and type 'python pick_place_no_armtag.py'

import rospy

import numpy as np

IMAGE_DIR = "/tmp/rgbd"

import os
import glob

# Define paths
base_dir = "/tmp/rgbd"
folders = ["depth", "intrinsics", "pose", "rgb"]


# Function to get the latest file from a folder
def get_latest_file(folder_path, extension):
    files = glob.glob(os.path.join(folder_path, f"*.{extension}"))
    if not files:
        return None
    # Sort files by modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_latest_DIPR():
    # Get the latest files from each folder
    latest_depth = get_latest_file(os.path.join(base_dir, "depth"), "png")
    latest_intrinsics = get_latest_file(os.path.join(base_dir, "intrinsics"), "json")
    latest_pose = get_latest_file(os.path.join(base_dir, "pose"), "*")  # Assuming any file type
    latest_rgb = get_latest_file(os.path.join(base_dir, "rgb"), "jpg")

    # Create a tuple of the latest items
    latest_tuple = (latest_depth, latest_intrinsics, latest_pose, latest_rgb)
    print(latest_tuple)
    return latest_tuple


from barebonesllmchat.terminal.interface import ChatbotClient
from barebonesllmchat.common.chat_history import CHAT_ROLE, ChatHistory, ChatHistoryWithImages

client = None


def vlm_point_at(image_path, queries):
    global client

    if client is None:
        client = ChatbotClient("http://127.0.0.1:5000", "your_api_key")

    chat_history_with_images = ChatHistoryWithImages(ChatHistory(), {})

    answers = []
    for query in queries:
        # if this ever goes to production, this can easily be batch on the llm side, but the client doesn't support it yet
        chat_history_with_images = chat_history_with_images.add(CHAT_ROLE.USER, query, image_path)
        client.send_history("new chat!", chat_history_with_images)
        answers.append(client.get_chat_messages("new chat!"))

    print(answers)

    points = {}
    for query, answer in zip(queries, answers):
        pointed = answer.history[-1]["content"]

        x = float(pointed.split(' x="')[-1].split('" ')[0])
        y = float(pointed.split(' y="')[-1].split('" ')[0])

        points[query] = (x, y)

    return points


import tf2_ros
import cv2
import numpy as np
from tf.transformations import euler_from_quaternion
from interbotix_common_modules import angle_manipulation as ang


def pixel_relxy_to_pixel_absxyd(relative_xy, rgb_path, depth_path, intrinsic_path):
    # Load the RGB image to get its dimensions
    rgb_image = cv2.imread(rgb_path)
    height, width, _ = rgb_image.shape

    # Convert relative coordinates (0-100%) to absolute pixel values
    u = int(relative_xy[0] / 100 * width)
    v = int(relative_xy[1] / 100 * height)

    # Load the depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("Failed to load the depth image.")

    # Get the depth value at the pixel (u, v)
    z = depth_image[v, u] / 1000.0  # Assuming depth is in millimeters, convert to meters

    # Check for invalid depth
    if z == 0:
        raise ValueError("Depth value is zero at the specified pixel.")
    return u, v, z


import pyrealsense2


def pixel_absxyd_to_cameraframe(absxyd):
    cameraInfo = rospy.wait_for_message("/locobot/camera/color/camera_info", CameraInfo)

    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]

    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model = pyrealsense2.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [absxyd[0], absxyd[1]], absxyd[
        2])  # result[0]: right, result[1]: down, result[2]: forward
    return result  # result[2], -result[0], -result[1]


def cameraframe_xyz_to_otherframe(xyz, ref_frame="locobot/arm_base_link"):
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    cluster_frame = "locobot/camera_color_optical_frame"
    try:
        trans = tfBuffer.lookup_transform(ref_frame, cluster_frame, rospy.Time(0), rospy.Duration(4.0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr("Failed to look up the transform from '%s' to '%s'." % (ref_frame, cluster_frame))
        raise e
    tranx = trans.transform.translation.x
    trany = trans.transform.translation.y
    tranz = trans.transform.translation.z
    quat = trans.transform.rotation
    q = [quat.x, quat.y, quat.z, quat.w]
    rpy = euler_from_quaternion(q)
    T_rc = ang.poseToTransformationMatrix([tranx, trany, tranz, rpy[0], rpy[1], rpy[2]])

    x, y, z = xyz
    p_co = [x, y, z, 1]
    p_ro = np.dot(T_rc, p_co)

    return tuple(list(p_ro)[:-1])


def magic_final_adjustments(xyz, distance=0.025):
    # Convert point to a numpy array
    point = np.array(xyz)

    # Compute the direction vector from the origin
    direction = point

    # Normalize the direction vector
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("The point is at the origin, cannot determine direction.")
    direction_normalized = direction / norm

    # Scale the direction vector by the specified distance
    scaled_vector = direction_normalized * distance

    # Move the point by adding the scaled vector
    new_point = point + scaled_vector
    return tuple(list(new_point))


from sensor_msgs.msg import Image, CameraInfo, CompressedImage


def dictmap(dico, func):
    ret = {}
    for key, val in dico.items():
        ret[key] = func(val)
    return ret


import functools


def main():
    bot = InterbotixLocobotXS("locobot_wx250s", arm_model="mobile_wx250s")

    from ovmm_ros.srv import SaveRGBD
    print("connecting to rgbd")
    rospy.wait_for_service("save_rgbd")
    service = rospy.ServiceProxy("save_rgbd", SaveRGBD)
    print("connected")
    bot.camera.pan_tilt_move(0, 0.75)
    time.sleep(1)

    result = service()
    rospy.loginfo(result)
    print("service called")

    bot.arm.set_ee_pose_components(x=0.3, z=0.2, moving_time=1.5)
    bot.gripper.open()

    depth, intrinsics, pose, rgb = get_latest_DIPR()

    import sys
    queries = []
    for arg in sys.argv[1:]:
        queries.append(f"Point at the center of the {arg} please")

    # queries = ["Point at the center of the white duckie please"]#, "Point at the green duckie please", "Point at the white duckie please"]
    points_xy = vlm_point_at(rgb, queries)
    print("pixel rel xy")
    print(points_xy)

    points_xyd = dictmap(points_xy, functools.partial(pixel_relxy_to_pixel_absxyd, rgb_path=rgb, depth_path=depth,
                                                      intrinsic_path=intrinsics))
    print("pixel abs xyd")
    print(points_xyd)
    points_xyz = dictmap(points_xyd, functools.partial(pixel_absxyd_to_cameraframe))
    print("camera frame xyz")
    print(points_xyz)
    points_xyz = dictmap(points_xyz, functools.partial(cameraframe_xyz_to_otherframe))
    print("arm base frame xyz")
    print(points_xyz)

    points_xyz = dictmap(points_xyz, functools.partial(magic_final_adjustments))
    print("arm base frame xyz")
    print(points_xyz)

    # exit()
    for query, cluster in points_xyz.items():
        print(cluster)
        x, y, z = cluster
        bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.05, pitch=0.5)
        bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=0.5)
        bot.gripper.close()
        bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.05, pitch=0.5)
        bot.arm.set_ee_pose_components(y=0.3, z=0.2)
        bot.gripper.open()
    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.arm.go_to_sleep_pose()

    exit()


if __name__ == '__main__':
    main()

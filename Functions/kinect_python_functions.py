"""
Almost everything in here is based off https://github.com/KonstantinosAng
A true master of the pykinect2
God bless his soul
"""
def _depth_to_color_rec(kinect, depth_space_point, depth_frame_data):
    import numpy as np
    import ctypes

    # init point class, from python kinect
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    # Map the color frame to the depth points
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)

    # Imported from https://github.com/Kinect/PyKinect2/issues/80#issuecomment-626590524
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(1080*1920,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(1080, 1920, 2).astype(np.int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, 512 - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, 424 - 1)
    depth_frame = kinect.get_last_depth_frame()
    depth_img = depth_frame.reshape((424, 512, 1)).astype(np.uint16)
    align_depth_img = np.zeros((1080, 1920, 4), dtype=np.uint16)
    align_depth_img[:, :] = depth_img[depthYs, depthXs, :]

    return align_depth_img

def _depth_to_color_vid(kinect, depth_space_point, depth_frame):
    import numpy as np
    import ctypes

    # Convert the depth_frame back into an LP_c_ushort array
    depth_frame_data = depth_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    # init point class, from python kinect
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    # Map the color frame to the depth points
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)

    # Imported from https://github.com/Kinect/PyKinect2/issues/80#issuecomment-626590524
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(1080*1920,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(1080, 1920, 2).astype(np.int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, 512 - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, 424 - 1)

    depth_img = depth_frame.reshape((424, 512, 1)).astype(np.uint16)
    align_depth_img = np.zeros((1080, 1920, 4), dtype=np.uint16)
    align_depth_img[:, :] = depth_img[depthYs, depthXs, :]
    return align_depth_img

def _rgbd_converter(kinect, color_space_point, depth_frame, color_frame):
    import numpy as np
    import ctypes

    # Convert the depth_frame back into an LP_c_ushort array
    depth_frame_data = depth_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    # Map Depth to Color Space
    depth2color_points_type = color_space_point * np.int(512 * 424)
    depth2color_points = ctypes.cast(depth2color_points_type(), ctypes.POINTER(color_space_point))
    kinect._mapper.MapDepthFrameToColorSpace(ctypes.c_uint(512 * 424), depth_frame_data, kinect._depth_frame_data_capacity, depth2color_points)

    colorXYs = np.copy(np.ctypeslib.as_array(depth2color_points, shape=(424 * 512,)))  # Convert ctype pointer to array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(424, 512, 2).astype(np.int)
    colorXs = np.clip(colorXYs[:, :, 0], 0, 1920 - 1)
    colorYs = np.clip(colorXYs[:, :, 1], 0, 1080 - 1)

    color_img = color_frame.reshape((1080, 1920, 3)).astype(np.uint8)
    align_color_img = np.zeros((424, 512, 3), dtype=np.uint8)
    align_color_img[:, :] = color_img[colorYs, colorXs, :]
    return align_color_img


def record_rgbd(kinect_obj):
    from pykinect2 import PyKinectV2

    rgbd_frame = _depth_to_color_rec(kinect_obj, PyKinectV2._DepthSpacePoint, kinect_obj._depth_frame_data)
    return rgbd_frame


def convert_to_rgbd(kinect_obj, depth_frame, color_frame):
    from pykinect2 import PyKinectV2

    # Unfortunatly we cannot use the ._mapper functions without connection to the kinect
    rgbd_frame = _rgbd_converter(kinect_obj, PyKinectV2._ColorSpacePoint, depth_frame, color_frame)
    return rgbd_frame

def convert_to_depthCamera(kinect_obj,depth_frame):
    from pykinect2 import PyKinectV2

    # Unfortunatly we cannot use the ._mapper functions without connection to the kinect
    rgbd_frame = _depth_to_color_vid(kinect_obj, PyKinectV2._DepthSpacePoint, depth_frame)
    return rgbd_frame
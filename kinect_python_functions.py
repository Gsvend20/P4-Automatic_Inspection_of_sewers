
def depth_to_color(kinect, depth_space_point, depth_frame_data):
    import numpy as np
    import ctypes

    # init point class, from python kinect
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    # Map the color frame to the depth points
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2depth_points)

    # Imported from https://github.com/Kinect/PyKinect2/issues/80#issuecomment-626590524
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(kinect.color_frame_desc.Height*kinect.color_frame_desc.Width,)))  # Convert ctype pointer to array
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 2).astype(np.int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, kinect.depth_frame_desc.Width - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, kinect.depth_frame_desc.Height - 1)
    depth_frame = kinect.get_last_depth_frame()
    depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width, 1)).astype(np.uint16)
    align_depth_img = np.zeros((1080, 1920, 4), dtype=np.uint16)
    align_depth_img[:, :] = depth_img[depthYs, depthXs, :]

    return align_depth_img


def record_rgbd(kinect_obj):
    from pykinect2 import PyKinectV2

    rgbd_frame = depth_to_color(kinect_obj, PyKinectV2._DepthSpacePoint, kinect_obj._depth_frame_data)
    return rgbd_frame


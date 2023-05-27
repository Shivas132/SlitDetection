from image_process_utils import *


def blocks_objects_filtering(video, non_slit_frame, cov_ratio=0.3, cube_size=10):
    """
        Filters out blocks/objects from the video frames based on a coverage ratio threshold.

        Args:
            video (ndarray): 3D array representing the video frames.
            non_slit_frame (int): Index of the non-slit frame to use for filtering.
            cov_ratio (float, optional): Coverage ratio threshold. Blocks/objects with a coverage ratio
                higher than this threshold will be filtered out. Defaults to 0.3.
            cube_size (int, optional): Size of the cubes used for analysis. Defaults to 10.

        Returns:
            ndarray: Filtered video with blocks/objects removed based on the coverage ratio threshold.
        """
    frames, rows, cols = video.shape
    mask = np.ones((rows, cols))
    frame = video[non_slit_frame]
    ratios = {}
    for i in range(0, rows, cube_size // 5):
        for j in range(0, cols, cube_size // 5):
            cube = frame[i: i + cube_size, j: j + cube_size]
            ratio = np.sum(cube) / np.square(cube_size)
            ratios[(i, j)] = ratio
            if ratio > cov_ratio:
                mask[i: i + cube_size, j: j + cube_size] = 0
    masked_video = np.logical_and(video, mask)
    return masked_video


video_path = f"{DELTAS}exp_1_deltas_thresh_0.05_size_4_eccen_0.7_filter.dat"
video = frames_as_matrix_from_binary_file(video_path, offset=False)
masked_video = blocks_objects_filtering(video, 64)
masked_video = normalize_to_int(masked_video)
save_video(masked_video, f'exp1_object_remove_ratio_0.3_overlapping_8pxls', DELTAS)


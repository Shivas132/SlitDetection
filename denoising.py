"""Module to search and test denoising methods."""

from image_process_utils import *
import cv2 as cv


# --- creating denoised video: ---
def denoise_video(data: np.array, h: int = 3, template_window_size: int = 7, search_window_size: int = 21) -> np.array:
    """
        Applies denoising to a video data using the fast non-local means denoising algorithm.

        Args:
            data (numpy.ndarray): The input video data.
            h (int, optional): The parameter controlling filter strength.
                               Higher h removes more noise but can also remove details. Defaults to 3.
            template_window_size (int, optional): The size of the pixel neighborhood used for denoising.
                                                  Larger values can remove more noise but can also remove details.
                                                  Defaults to 7.
            search_window_size (int, optional): The size of the pixel neighborhood used for the search step.
                                                Larger values increase computational complexity. Defaults to 21.

        Returns:
            numpy.ndarray: The denoised video data.
    """
    new_video = np.empty(data.shape, dtype=np.uint8)
    for i in range(data.shape[0]):
        new_video[i] = cv.fastNlMeansDenoising(data[i], h=h, templateWindowSize=template_window_size,
                                               searchWindowSize=search_window_size)
    return new_video

import cv2 as cv
from skimage.restoration import denoise_tv_chambolle ,denoise_nl_means, estimate_sigma
from skimage import io
import numpy as np


def nl_means(input_, h=3, template_window_size=7,search_window_size=21, video=False):
        return cv.fastNlMeansDenoising(input_, h=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)
"""Module to find slits properties."""

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure
import cv2 as cv
from image_process_utils import *
from difference_finding import deltas_video


def region_props_hover(img: np.array, props: list[str], title: str = "") -> None:
    """Show region properties in an interactive image.

    Open an HTML page with the image where the regions are marked.
    Properties of region are displayed by hovering the region with the mouse.

    Args:
        img: image to find regions in.
        props: list of properties to display.
        title: title to the image. by default, it's an empty string.

    """

    labels = measure.label(img)
    fig = px.imshow(img, binary_string=True, title=title)
    fig.update_traces(hoverinfo='skip')
    properties = measure.regionprops(labels, img)
    for index in range(1, labels.max()):
        label_i = properties[index].label
        contour = measure.find_contours(labels == label_i)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop in props:
            hoverinfo += f'<b>{prop}: {getattr(properties[index], prop):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)


def imfill(img: np.array) -> np.array:
    """Fill holes in objects in an image using flood fill algorithm.

    Args:
        img: must be binary image, and type of values is int.

    Returns:
        np.array: the image with the holes filled.
    """
    rows, cols = img.shape
    mask = np.zeros((rows + 2, cols + 2), np.uint8)
    im_ff = img.copy()
    cv.floodFill(im_ff, mask, (0, 0), MAX_GRAY_VAL)
    im_ff = cv.bitwise_not(im_ff)
    return img | im_ff


def deltas_with_noise_removal(video: np.array, thresh: float = 0.02) -> np.array:
    """Create deltas video from data, with removal of non-slit objects.

    The method gets a video and create a for every frame a frame that contains the differences from the previous one.
    In each new frame, a white pixel is a pixel that has changed, and a black one is a pixel that hasn't.

    Additionally, objects that had been identified as no slits are being blacken.
    Identification is Based on a minimal area of the object, and range of angles at which a slit may appear.

    Args:
        video: input video.
        thresh: threshold for the thresholding process for each frame.

    Returns:
        np.array: a new video with only the filtered differences.
    """
    data = np.copy(video)
    data = normalize_to_int(data)
    frames_num, rows, cols = data.shape
    deltas = np.empty((frames_num, rows, cols), dtype=np.uint8)
    sum_of_deltas = np.zeros((rows, cols), np.float64)
    for i in range(1, frames_num):
        delta_img = data[i] - data[i - 1]
        delta_img = thresholding(delta_img, thresh)
        sum_of_deltas = cv.bitwise_or(sum_of_deltas, delta_img)
        deltas[i] = sum_of_deltas
        # remove noise by area, orientation, eccentricity:
        deltas[i] = imfill(deltas[i])
        labels = measure.label(deltas[i])
        regs = measure.regionprops(labels, deltas[i])
        # properties to recognize a slit:
        min_area = 4
        orientation = 0.01
        min_eccentricity = 0.7
        for reg in regs:
            if reg.area < min_area or np.abs(reg.orientation) < orientation or reg.eccentricity < min_eccentricity:
                min_r, min_c, max_r, max_c = reg.bbox
                deltas[i][min_r:max_r, min_c:max_c] = 0
    return deltas


def noise_remove_by_props(video, min_area=4, min_eccentricity=0.7):
    """
        Removes noise in the data based on specified properties.

        Args:
            video (array-like): The input data.
            min_area (int, optional): The minimum area threshold for noise removal. Defaults to 4.
            min_eccentricity (float, optional): The minimum eccentricity threshold for noise removal. Defaults to 0.7.

        Returns:
            numpy.ndarray: The data with noise removed.
    """
    data = np.copy(video)
    data = normalize_to_int(data)
    frames_num, rows, cols = data.shape
    for i in range(frames_num):
        # remove noise by area, orientation, eccentricity:
        data[i] = imfill(data[i])
        labels = measure.label(data[i])
        regs = measure.regionprops(labels, data[i])
        # properties to recognize a slit:
        for reg in regs:
            if reg.area < min_area or reg.eccentricity < min_eccentricity:
                min_r, min_c, max_r, max_c = reg.bbox
                data[i][min_r:max_r, min_c:max_c] = 0
    return data


# @numba.jit(nopython=True)
def extract_area(video, area):
    """
        Extracts a specific area from a video.

        Args:
            video (array-like): The input video.
            area (tuple): The coordinates of the area to be extracted (x1, y1, x2, y2).

        Returns:
            numpy.ndarray: The extracted area as a video.
    """
    video = normalize_to_float(video)
    x1, y1, x2, y2 = area  # Extract the rectangle coordinates
    result = np.zeros_like(video)  # Create a black video of the same shape as the input video
    result[:, y1:y2, x1:x2] = video[:, y1:y2, x1:x2]  # Set the rectangle region to the corresponding region in the input video
    return result


def clean_area(video, area):
    """
        Extracts a specific area from a video.

        Args:
            video (array-like): The input video.
            area (tuple): The coordinates of the area to be extracted (x1, y1, x2, y2).

        Returns:
            numpy.ndarray: The extracted area as a video.
    """
    video = normalize_to_float(video)
    x1, y1, x2, y2 = area  # Extract the rectangle coordinates
    result = video  # Create a black video of the same shape as the input video
    result[:, y1:y2, x1:x2] = 0  # Set the rectangle region to the corresponding region in the input video
    return normalize_to_int(result)


def create_deltas_videos(video, area):
    """
        Chooses thresholds for a video based on a specific area.

        Args:
            video (array-like): The input video.
            area (tuple): The coordinates of the area to be considered (x1, y1, x2, y2).

        Returns:
            numpy.ndarray: An array of thresholded videos.
    """
    video = normalize_to_float(video)
    video = extract_area(video, area)
    videos = np.zeros((20,) + video.shape, dtype=video.dtype)
    threshes = [i / 1000 for i in range(0, 100, 5)]
    for i in range(20):
        deltas = deltas_video(video, thresh=threshes[i])
        videos[i] = normalize_to_int(deltas)
        videos[i] = normalize_to_int(videos[i])
    return videos

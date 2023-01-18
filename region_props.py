"""Module to find slits properties."""

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, data
import numpy as np
import pandas as pd
from skimage.morphology import closing, square
import cv2 as cv
from image_process_utils import *

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\outputs\deltas_video_thresh_at_end.dat"
video = frames_as_matrix_from_binary_file(video_path, offset=False)

# --- create interactive image of region props for last frame: ---

# frame_num = 127
# frame = video[frame_num]
#
# morph_size = 3
# closed = closing(frame, square(morph_size))


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


properties = ['area', 'eccentricity', 'perimeter', 'axis_major_length', 'orientation']


# --- comparing props of 2 frames: ---

def imfill(img: np.array) -> np.array:
    """Fill holes in objects in an image using flood fill algorithm.

    Args:
        img: must be binary image, and type of values is int.

    Returns:
        np.array: the image with the holes filled.
    """
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    im_ff = img.copy()
    cv.floodFill(im_ff, mask, (0, 0), MAX_GRAY_VAL)
    im_ff = cv.bitwise_not(im_ff)
    return img | im_ff


frames = video[-2:]
frames = normalize_to_int(frames, MAX_GRAY_VAL)

filled = np.array([imfill(frame) for frame in frames])
labeled_frames = [measure.label(frame) for frame in filled]
tables = [measure.regionprops_table(frame, properties=properties) for frame in labeled_frames]
tables = [pd.DataFrame(table) for table in tables]

print(tables[0], '\n\n', tables[1])
region_props_hover(filled[1], properties)


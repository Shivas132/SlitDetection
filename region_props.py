"""Module to find slits properties."""

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure
from skimage.morphology import closing, square
from image_process_utils import *

video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\dat_files\outputs\deltas_video_thresh_at_end.dat"

with open(video_path, 'rb') as f:
    video = np.fromfile(f, dtype='<i2', count=FRAMES_NUM * FRAME_HEIGHT * FRAME_WIDTH)
video = video.reshape((FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH))

frame_num = 127
frame = video[frame_num]

morph_size = 3
closed = closing(frame, square(morph_size))

labels = measure.label(closed)

fig = px.imshow(frame, binary_string=True, title=f'region props after closing with square({morph_size})')
fig.update_traces(hoverinfo='skip')

props = measure.regionprops(labels, frame)
properties = ['area', 'eccentricity', 'perimeter', 'axis_major_length', 'orientation']

for index in range(1, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))

plotly.io.show(fig)


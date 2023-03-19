"""Module to utilize the additional videos of every experiment."""

from image_process_utils import *


video_path = r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_0\static.dat"
data = frames_as_matrix_from_binary_file(video_path)

mean_frame = np.mean(data, axis=0)
# show_frame(mean_frame)

data = data - mean_frame
# print(np.min(data))

# save_video(data, 'exp_0_static_substructed_mean')

# exp_0 = frames_as_matrix_from_binary_file(r"C:\Users\obaryosef\PycharmProjects\slitDetectionProject\SlitDetection\inputs\exp_0\exp.dat")
# denoised_1 = exp_0 - data
#
data -= np.min(data)
save_video(data, 'exp_0_static_substructed_mean')

# denoised_2 = exp_0 - data
#
# show_frame(denoised_1[64], 'exp - mean')
# show_frame(denoised_2[64], 'exp - mean + min(mean)')

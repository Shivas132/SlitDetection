

from skimage.measure import regionprops


# min_val, max_val = np.min(data[50]), np.max(data[50])
# print(min_val, max_val)

# TODO: create histogram to original data, data_norm ==> look at distribution of values.
# first, move values to [0, 1], type=float









# show_frame(background_frame, 'background frame')




show_frame(background_frame, 'background')
show_frame(denoise_data, 'denoise_data')

# gaus_blur_64 = cv.GaussianBlur(img_to_denoise, (5, 5), 0, 0)
# show_frame(gaus_blur_64, "gaus blur 64")

diff_64 = -1 * (denoise_data - background_frame)
# show_frame(diff_64, "diff 64, -1")







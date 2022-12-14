from matplotlib import pyplot as plt
import numpy as np
import cv2

# Open the .dat file in binary mode
with open("CAM1_22_41_46.dat", 'rb') as f:
  # Move the file pointer to the specified offset
  f.seek(6336)

  # Read the data into a NumPy array
  data = np.fromfile(f, dtype='<i2', count=128*250*400)

# Reshape the array into the desired dimensions
data = data.reshape((128, 250, 400))

dataNorm = (data*(255/np.max(data)))
for frame in dataNorm:
  plt.imshow(frame,cmap='gray')
  plt.show()

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
from autocrop import Cropper

# We can also show the array from Matplotlib
loc = "/opt/ml/input/data/train/images/000001_female_Asian_45/mask1.jpg"
c = Cropper()
img_array = c.crop(loc)
plt.imshow(img_array)
plt.show()

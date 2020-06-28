import glob
import os
import numpy as np
from imageio import imsave
import argparse
from utils.dataset_processing.image import DepthImage

path = "/home/ubuntu/egad/depth-images"

pcds = glob.glob(os.path.join(path, '*.pcd'))

for pcd in pcds:
    di = DepthImage.from_pcd(pcd, (480, 640))
    di.inpaint()

    of_name = pcd.replace('.pcd', 'd.tiff')
    print(of_name)
    imsave(of_name, di.img.astype(np.float32))
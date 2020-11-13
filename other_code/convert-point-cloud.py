import glob
import os
import numpy as np
from imageio import imsave
import argparse
from utils.dataset_processing.image import DepthImage

path = "/home/ubuntu/egad/depth-images"

pcds = glob.glob(os.path.join(path, '*.pcd'))

for pcd in pcds:
    depIm = DepthImage.from_pcd(pcd, (270, 270))
    depIm.inpaint()
    depIm.normalise()

    of_name = pcd.replace('.pcd', 'd.tiff')
    print(of_name)
    imsave(of_name, depIm.img.astype(np.float32))
    
    
    
    

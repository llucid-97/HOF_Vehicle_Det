from skimage.feature import hog
from random import randint
from loadData import loadAll
import matplotlib.pyplot as plt
import cv2
import numpy as np

debug = False



def feature_extract(images,feature_vec=True):
    global debug
    hog_features = []
    for image in images:
        tmp_result = []
        norm_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        norm_image = cv2.normalize(norm_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        for i in range(3):
            tmp_result.append(None)

            if debug:
                tmp_result[-1], vis = hog(norm_image[:, :, i],
                                          orientations=11,
                                          pixels_per_cell=(16, 16),
                                          cells_per_block=(2, 2),
                                          transform_sqrt=False,
                                          visualise=debug,
                                          feature_vector=False)
                if i == 0:
                    plt.ion()
                    ax1 = plt.subplot(211)
                    ax1.imshow(image)
                    ax1.set_title('Car Image', fontsize=16)
                    ax2 = plt.subplot(212)
                    ax2.imshow(vis, cmap='gray')
                    ax2.set_title('Car HOG', fontsize=16)
                    plt.pause(1)
                    ax1.cla()
                    ax2.cla()
            else:
                tmp_result[-1] = hog(norm_image[:, :, i],
                                     orientations=11,
                                     pixels_per_cell=(16, 16),
                                     cells_per_block=(2, 2),
                                     transform_sqrt=False,
                                     visualise=debug,
                                     feature_vector=False)
        if feature_vec:
            hog_features.append(np.ravel(tmp_result))
        else:
            hog_features.append(tmp_result)

    return hog_features


if __name__ == "__main__":
    debug = 1

    _, pt, _, _ = loadAll()
    fig = plt.figure()
    feature_extract(pt)

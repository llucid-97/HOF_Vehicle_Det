# Load training data
import sys
import os
from urllib.request import urlretrieve
import glob
import cv2
from sklearn.utils import shuffle

url_cars = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"
url_notCars = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"
debug = False


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def loadAll():
    # Get paths
    pos = glob.glob('./dataset/vehicles/vehicles/**/*.png')
    neg = glob.glob('./dataset/non-vehicles/non-vehicles/**/*.png')

    # Load and augment the dataset
    pos_images = []
    neg_images = []
    for path in pos:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        pos_images.append(img)
        flip = cv2.flip(img, 1)
        pos_images.append(flip)
    for path in neg:
        neg_images.append(
            cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        )
    # Split train/val

    pDim = len(pos)
    nDim = len(neg)

    pos_images = shuffle(pos_images)
    neg_images = shuffle(neg_images)
    split = int(pDim * 0.8)

    pos_train = pos_images[:split]
    neg_train = neg_images[:split]

    pos_valid = pos_images[split:]
    neg_valid = neg_images[split:]

    global debug
    if debug:
        print(len(pos_images) + len(neg_images), " Images")
        print("80% training, 20% validation")

    return pos_train, neg_train, pos_valid, neg_valid


if __name__ == "__main__":
    debug = True
    if not os.path.isfile("vehicles.zip"):
        urlretrieve(url_cars, "vehicles.zip", reporthook)
    if not os.path.isfile("non-vehicles.zip"):
        urlretrieve(url_cars, "non-vehicles.zip", reporthook)

    loadAll()

# Load training data
import sys
import  os
from urllib.request import urlretrieve

url_cars = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"
url_notCars = "https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip"


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


if __name__ == "__main__":
    if not os.path.isfile("vehicles.zip"):
        urlretrieve(url_cars, "vehicles.zip", reporthook)
    if not os.path.isfile("non-vehicles.zip"):
        urlretrieve(url_cars, "non-vehicles.zip", reporthook)

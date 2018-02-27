# This code is largely adapted from Udacity's CarND lectures.
# Available: https://classroom.udacity.com/nanodegrees/nd013/

import matplotlib.pyplot as plt
import cv2
import numpy as np
from HOG import feature_extract
from train import loadModel, train
from scipy.ndimage.measurements import label
import glob

debug = False
showAll = False

orientations = 11
pixels_per_cell = 16
cells_per_block = 2

# Block split static params
window = 64
nFeaturesPerBlock = orientations * cells_per_block ** 2
nBlocksPerWindow = (window // pixels_per_cell) - cells_per_block + 1
stride = 2

lastHeatMap = None
if not debug:
    fig = plt.figure()


def getCars(image, scaleDiv, model, yStart=350, yStop=660, xStart=0, xStop=1280):
    """
    :param image: Cropped RGB image frame
    :param scaleDiv: Divides image dimensions by this factor
    :return:
    """
    global debug
    image = image[yStart:yStop, xStart:xStop]

    # 1: Scale
    if scaleDiv != 1:
        imshape = image.shape
        img = cv2.resize(image, (np.int(imshape[1] / scaleDiv),
                                 np.int(imshape[0] / scaleDiv)))
    else:
        img = np.copy(image)

    # Split into blocks
    nxblocks = (img.shape[1] // pixels_per_cell) - cells_per_block + 1
    nyblocks = (img.shape[0] // pixels_per_cell) - cells_per_block + 1

    nxsteps = (nxblocks - nBlocksPerWindow) // stride + 1
    nysteps = (nyblocks - nBlocksPerWindow) // stride + 1

    # HOG Transform
    HOG_image = feature_extract([img], False)
    HOG_image = np.array(HOG_image[0])

    rectangles = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * stride
            xpos = xb * stride
            # Extract HOG for this patch

            featCrop = HOG_image[:, ypos:ypos + nBlocksPerWindow, \
                       xpos:xpos + nBlocksPerWindow].ravel()

            xleft = xpos * pixels_per_cell
            ytop = ypos * pixels_per_cell

            # Extract the image patch
            test_prediction = model.predict(featCrop.reshape(1, -1))

            global showAll
            if test_prediction == 1 or showAll:
                xbox_left = np.int(xleft * scaleDiv)
                ytop_draw = np.int(ytop * scaleDiv)
                win_draw = np.int(window * scaleDiv)
                rectangles.append(
                    ((xbox_left + xStart, ytop_draw + yStart),
                     (xbox_left + win_draw + xStart, ytop_draw + yStart + win_draw)))

    return rectangles


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    # Copied from Udacity CarND lesson material
    :param img:
    :param bboxes:
    :param color:
    :param thick:
    :return:
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    """
    STOLEN from Udacity CarND
    :param heatmap:
    :param bbox_list:
    :return:
    """
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1] - 1:box[1][1] + 1, box[0][0] - 1:box[1][0] + 1] += 1
        heatmap[:, -1:1] = 0
        heatmap[-1:1, :] = 0

    # Return updated heatmap
    return heatmap


def heatBoxes(img):
    # Iterate through all detected cars
    # img = cv2.dilate(img, np.ones((3, 3)), iterations=1)
    labels = label(img)

    boxList = []
    for i in range(labels[1]):
        pass
        nonzero = (labels[0] == i + 1).nonzero()

        boxList.append(((np.min(nonzero[1]), np.min(nonzero[0])), (np.max(nonzero[1]), np.max(nonzero[0]))))
    return boxList


def interpreteFrame(img, model):
    numBoxes = 0
    heatMap = np.zeros_like(img[:, :, 0])
    boxes = getCars(img, 0.7, model, 400, 720, 400, 1000)
    numBoxes += len(boxes)
    heatMap = 0.6 * add_heat(heatMap, boxes)

    boxes = getCars(img, 1.3, model, 400, 550)
    numBoxes += len(boxes)
    heatMap = add_heat(heatMap, boxes)

    boxes = getCars(img, 1.5, model, 400, 600)
    numBoxes += len(boxes)
    heatMap = add_heat(heatMap, boxes)

    boxes = getCars(img, 1.7, model, 400, 650)
    numBoxes += len(boxes)
    heatMap = add_heat(heatMap, boxes)

    boxes = getCars(img, 2, model, 400, 720)
    numBoxes += len(boxes)
    heatMap = add_heat(heatMap, boxes)

    # 2D hysteresis thresholding
    blobs = label(heatMap)
    for i in range(blobs[1]):
        if not np.any(heatMap[blobs[0] == (i + 1)] > 1.5):
            heatMap[blobs[0] == i + 1] = 0

    # heatMap[heatMap <= 1] = 0

    print("Found ", numBoxes, "boxes")

    # Temporal Smoothing: Leaky Integrator
    global lastHeatMap

    updateStrength = 0.2
    if not (lastHeatMap is None):
        heatMap = ((1 - updateStrength) * lastHeatMap) + \
                  (updateStrength * heatMap)

    # Thresholding
    lastHeatMap = heatMap

    heatMap[heatMap <= 0.13] = 0

    boxes = heatBoxes(heatMap)
    img = draw_boxes(img, boxes, 'random')
    heatMap[0, 0] = 2
    plt.ion()
    ax1 = plt.subplot(211)
    ax1.imshow(img)
    ax2 = plt.subplot(212)
    ax2.imshow(lastHeatMap, cmap='hot')
    plt.pause(0.001)
    ax1.cla()
    ax2.cla()
    return img


if __name__ == "__main__":
    # showAll = True
    debug = True
    test_images = glob.glob('./test_images/test*.jpg')
    for path in test_images:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        model = loadModel()
        interpreteFrame(img, model)
        input()

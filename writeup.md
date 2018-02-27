#Vehicle Detection Project

The goals / steps of this project are to track vehicles in the test video by:

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
     >* **%TODO: Cut?** Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    >* **%TODO: Cut!** Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implementing a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog.jpg
[image2]: ./output_images/box_ranges.png
[image3]: ./output_images/detection.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[heatmap1]: ./output_images/frame1.png
[heatmap2]: ./output_images/frame2.png
[heatmap3]: ./output_images/Frame3.png
[heatmap4]: ./output_images/Frame4.png
[heatmap5]: ./output_images/Frame5.png
[heatmap6]: ./output_images/Frame6.png
[image9]: ./output_images/pipeline.gif
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

**Loading Dataset:**
I started by downloading the dataset in `loadData.py` (if main function)

Once downloaded, the function `loadAll()` from line 27 of `loadData.py` reads training data into a list of numpy array images,
and slightly augments the dataset through mirroring.

The data is shuffled and split into a training and validation set to easily catch over/under fitting

**Feature Extraction**
The HOG code is implemented in `feature_extract()` from line 11 of `HOG.py`.  
 
Here is an example of one of each of the `vehicle` and `non-vehicle` classes and the HOG transform of their first channel:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.


I had a "default" set to build the initial training pipeline connected with the SVM Classifier

I then changed the parameters one by one and retrained, each time keeping the only the changes that increased accuracy without sacrificing too much speed.

The features I settled on were all HOG channels of a YUV colorspace image normalized and ravelled into a single vector, and the parameter set was:
* orientations=11
* pixels_per_cell=(16, 16)
* cells_per_block=(2, 2)



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier was created and trained in the function `train()` in `train.py`.
It was an SVC classifier. My design methodology was:

* Make as simple a classifier as possible
* Get maximum accuracy you can with that and augmenting the dataset
* Increase complexity of classifier if you can't reach acceptable levels

I got up to 99% accuracy on my validation set with a linear SVC, so did not need to go further.

> **SideNote**: Perhaps this methodology was flawed because trying more complex classifiers SVC kernels didn't converge to a solution nearly as well as the linear when I experimented with them %TODO: Look into better design methodologies later
 

### Sliding Window Search



#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the function `getCars()` in `getCars.py`, I adapt the function given in the lectures.

Given sample image, the function:
* Crops to target y-area
* Scales up/down to target magnification
* Generates parameters for window positions based on HOG training dimensions and frame dimensions, desired size and stride/overlap of windows
* Takes the HOG transform of the cropped frame
* Iterates through windows, cropping the HOG transform to each window and running a prediction on it
* If true, its a window

Each window overlaps its neighbours halfway (in both x and y)
 



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The checking is done at 2 scales and 2 y-intervals because of perspective as shown below:
![alt text][image2]

The scales were found empirically: smaller scales gave more false positives. Larger scales gave more false negatives

This works out to show:
![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


![alt text][image9]

Here's a [link to my video result](./output_videos/out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This is done mainly in the function `interpreteFrame()` in `getCars.py`
 
I recorded the positions of positive detections in each frame of the video.

From the positive detections I created a heatmap.

Here's an example result showing the heatmap from a series of frames of video.
 
I then use `scipy.ndimage.measurements.label()` to separately label(number) clustered detections

I use the result from that to select each label and hysteresis threshold them

Then I temporally filter the result with a leaky integrator

### Here are six frames and their corresponding heatmaps:

![alt text][heatmap1]
![alt text][heatmap2]
![alt text][heatmap3]
![alt text][heatmap4]
![alt text][heatmap5]
![alt text][heatmap6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The classifier's accuracy was really good. Aside from expanding the dataset, not much could improve there.

My biggest mistake was committing to using only its "true"/"false" predictions as given.

I should have taken a confidence score instead (eg: Distance from the SVC decision boundary).

With this, it would have been easier to filter out low-confidence false positives.

It would also have boosted accuracy since I could then perform non-max suppresison on windows' confidence scores, meaning:

I would need far fewer sliding window scales: 2 would have sufficed, instead of the 6 my final version used because:

I could get an imprecise but accurate prediction, when doing a "broad search" (large scale/stride windows), then narrow it down to a small area around this "true" prediction.

This would also eliminate any false positives because they would be verified then and there.

___


Lastly, I'd separate my temporal filtering into 2 parts:
* Smoothing box dimensions
* Reinforcing probability of box presence being a true positive

The current implementation ties these two together, and finding parameters to ge tthem both working well was taking too long.

  


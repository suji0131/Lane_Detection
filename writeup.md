
**Detecting Lane Lines on the Road**

The goals of this project is as following:
* To make a pipeline that finds lane lines on the road

Pipeline:
(a) Convert image to grayscale
(b) Remove noise by applying gaussian blur
(c) Use canny edge detection algorithm to identify the edge points
(d) Then take points or lines from your region of interest
(e) Use Hough transformation to convert the points from canny edge detection to line
(f) Rather the averaging the slopes we got from the hough transformation, I applied 
    regression to points (from houg transform), this way we find the best possible fit 
    for the points as opposite to sub-optimal heuristic, averaging.
Note: to retain the information from previous line I took average of previous two slopes.


### Reflection

###1. Potential shortcomings with current pipeline


One potential shortcoming would be that we can't draw for curved lanes 

Another is we have to fine tune parameters for every possible colors or scenarios

The drawn lines are wiggling, not smoothly transitioning


###2. Suggest possible improvements to your pipeline

A possible improvement would be to use a polynomial curve to fit the points instead of a line.
OR we can break region of interest into small regions and fit linear for each divide region but
it will be computationally expensive. 


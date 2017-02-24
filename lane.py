import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2, math, moviepy
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import scipy.stats as sp

class lane:
    def __init__(self):
        'img_name is a string that indicates file path'
        self.m_ = np.array([0,0]) #holds previous slope value.
        self.x1_ = np.array([0,0])
        self.x2_ = np.array([0,0])

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #used BGR2GRAY if you read an image with cv2.imread()
        #use RGB2GRAY for mpimg.imread()

    def gaussian_blur(self, img, kernel_size):
        'Applies guassian blur, which clears noise and avg'
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def canny(self, img, low_threshold=65, high_threshold=195):
        'Applies the canny edge transform'
        return cv2.Canny(img, low_threshold, high_threshold)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[200, 0, 0], thickness=10):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def hough_lines_old(self, img, rho=1, theta=np.pi/180, threshold=50, min_line_len=20, max_line_gap=25):
        """
        `img` should be the output of a Canny transform.
        Returns an image with hough lines drawn.
        """
        #rho = distance resolution in pixels of the Hough grid
        #theta = angular resolution in radians of the Hough grid
        #threshold = minimum number of votes (intersections in Hough grid cell)
        #min_line_length = minimum number of pixels making up a line
        #max_line_gap = maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.copy(self.image)
        self.draw_lines(line_img, lines)
        return line_img

    def regression(self, points, rl):
        """
        points is set of points that form lines after hough_transformation.
        Instead of simply averaging slopes, fitting a regression line is a better
        and robust option
        rl: tells whether points form left or right side line.
        """
        if (len(points)==0): #when no points are detected, previous line will be used
            return np.array([self.x1_[rl],540,self.x2_[rl],340])
        else:
            x = np.append(points[:,0,0], points[:,0,2]) #x cordinates of every pts
            y = np.append(points[:,0,1], points[:,0,3])#y cordinates of every pts
            m, c, r, p, std = sp.linregress(x,y) #regression
        if (self.m_[rl] == 0): #self.m_ holds previous slope value.
            pass #for first time
        else:
            m = (m + self.m_[rl])/2 #averages previous slope and new slope
        self.m_[rl] = m #update
        y1 = 540        #find points that form the regression line
        x1 = int((y1-c)/m)
        self.x1_[rl] = np.copy(x1)
        y2 = 340
        x2 = int((y2-c)/m)
        self.x2_[rl] = np.copy(x2)
        return np.array([x1, y1, x2, y2]) #two points that form a line

    def hough_lines(self, img, rho=1, theta=np.pi/180, threshold=50, min_line_len=20, max_line_gap=25):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        y_ = lines[:,0,3] - lines[:,0,1]
        x_ = lines[:,0,2] - lines[:,0,0]
        slope = np.divide(y_, x_)

        new_lines = np.zeros((2,1,4), dtype = int)
        new_lines[0,0,:] = self.regression(lines[slope > 0.1], 0) #right side pts
        temp = lines[slope < -0.1]
        new_lines[1,0,:] = self.regression(temp, 1) #left side pts


        line_img = np.copy(self.image)
        self.draw_lines(line_img, new_lines)
        return line_img

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.
        `initial_img` should be the image before any processing.
        The result image is computed as follows:
        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def pipe_line(self, img):
        self.image = img
        vertices = np.array([[(0,540),(430, 320), (510, 320), (910, 540)]], dtype=np.int32)
        gray_img = self.grayscale(self.image) #gray scale
        gray_gauss = self.gaussian_blur(gray_img, 3) #guassian blur
        canny_img = self.canny(gray_gauss) #canny edge detection
        reg_interest = self.region_of_interest(canny_img, vertices) #region of int
        hough_img = self.hough_lines(reg_interest) #hough transformation
        return hough_img

if __name__== '__main__':
    trail = lane()
    white_output = 'yellow.mp4'
    clip1 = VideoFileClip("solidYellowLeft.mp4")
    white_clip = clip1.fl_image(trail.pipe_line) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

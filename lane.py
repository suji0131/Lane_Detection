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
        #self.image = cv2.imread(img_name)

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

        lef_pts = lines[slope > 0.1]
        if (len(lef_pts)==0):
            pass
        else:
            lef_x = np.append(lef_pts[:,0,0], lef_pts[:,0,2])
            lef_y = np.append(lef_pts[:,0,1], lef_pts[:,0,3])
            m_l, c_l, r_l, p_l, se_l = sp.linregress(lef_x,lef_y)
            yl1 = 540
            xl1 = int((yl1-c_l)/m_l)
            yl2 = 350
            xl2 = int((yl2-c_l)/m_l)
            new_lines[0,0,:] = np.array([xl1, yl1, xl2, yl2])

        rig_pts = lines[slope < -0.1]
        if (len(rig_pts) ==0):
            pass
        else:
            rig_x = np.append(rig_pts[:,0,0], rig_pts[:,0,2])
            rig_y = np.append(rig_pts[:,0,1], rig_pts[:,0,3])
            m_r, c_r, r_r, p_r, se_r = sp.linregress(rig_x,rig_y)
            yr1 = 540
            xr1 = int((yr1-c_r)/m_r)
            yr2 = 350
            xr2 = int((yr2-c_r)/m_r)
            new_lines[1,0,:] = np.array([xr1, yr1, xr2, yr2])
        #new_lines = np.zeros((2,1,4), dtype = int)
        #new_lines[0,0,:] = np.array([xl1, yl1, xl2, yl2])
        #new_lines[1,0,:] = np.array([xr1, yr1, xr2, yr2])

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
        gray_img = self.grayscale(self.image)
        gray_gauss = self.gaussian_blur(gray_img, 3)
        canny_img = self.canny(gray_gauss)
        reg_interest = self.region_of_interest(canny_img, vertices)
        hough_img = self.hough_lines(reg_interest)
        #result = self.region_of_interest(hough_img, vertices)
        return hough_img

if __name__== '__main__':
    trail = lane()
    white_output = 'chall.mp4'
    clip1 = VideoFileClip("challenge.mp4")
    white_clip = clip1.fl_image(trail.pipe_line) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

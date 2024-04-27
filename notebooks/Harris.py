from Image import Image
import numpy as np
import cv2

class Harris(Image):
    def __init__(self):
        super().__init__()

    def calculate_Ix_Iy(self):
        Ix = cv2.Sobel(self.img_copy, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(self.img_copy, cv2.CV_64F, 0, 1, ksize=3)
        return Ix, Iy

    def calculate_Ixx_Iyy_Ixy(self, Ix, Iy):
        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy
        return Ixx, Iyy, Ixy

    def calculate_matrix_M(self, Ixx, Iyy, Ixy, window_size=3, k=0.04):
        offset = window_size//2
        height, width = self.img_copy.shape
        R = np.zeros_like(self.img_copy, dtype=np.float32)
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                Sxx = np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
                Syy = np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
                Sxy = np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])
                det = Sxx*Syy - Sxy**2
                trace = Sxx + Syy
                R[y, x] = det - k*(trace**2)
        return R

    def harris_corner_detection(self, R, alpha=0.01, window_size=3):
        offset = window_size//2
        height, width = self.img_copy.shape
        threshold = alpha * R.max()
        cornerList = []
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                value=R[y, x]
                if value>threshold:
                    cornerList.append([x, y, value])
        return cornerList

    def draw_corners(self, cornerList):
        for corner in cornerList:
            cv2.circle(self.img_copy,(corner[0],corner[1]),4,(0,255,0))
        return self.img_copy

    def harris_corner_detection_main(self, window_size=3, k=0.04, alpha=0.01):
        Ix, Iy = self.calculate_Ix_Iy()
        Ixx, Iyy, Ixy = self.calculate_Ixx_Iyy_Ixy(Ix, Iy)
        R = self.calculate_matrix_M(Ixx, Iyy, Ixy, window_size, k)
        cornerList = self.harris_corner_detection(R, alpha, window_size)
        img_corners = self.draw_corners(cornerList)
        return img_corners




# # Load the image
# img = cv2.imread('Harris.jpg', cv2.IMREAD_GRAYSCALE)
# color_image = cv2.imread('Harris.jpg')
#
# # Calculate Ix and Iy
# Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#
# # Calculate Ixx, Iyy, Ixy
# Ixx = Ix**2
# Iyy = Iy**2
# Ixy = Ix*Iy
#
# # Calculate the matrix M
# window_size = 3
# offset = window_size//2
# height, width = img.shape
# k = 0.04
# R = np.zeros_like(img, dtype=np.float32)
#
# for y in range(offset, height-offset):
#     for x in range(offset, width-offset):
#         Sxx = np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
#         Syy = np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
#         Sxy = np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])
#         det = Sxx*Syy - Sxy**2
#         trace = Sxx + Syy
#         R[y, x] = det - k*(trace**2)
#
#
#
# threshold = 0.02 * R.max()
# #
# for y in range(offset, height-offset):
#     for x in range(offset, width-offset):
#         value=R[y, x]
#         if value>threshold:
#             # cornerList.append([x, y, value])
#             cv2.circle(color_image,(x,y),4,(0,255,0))
#
#
# # # color_image[R_normalized > 0.3] = [255, 255, 255]  # Mark corners in red
# # # Plot R
# plt.figure(figsize=(8, 6))
# plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
# # plt.imshow(R, cmap='gray')
# plt.title('Color Image with Corners Marked')
# plt.axis('off')
# plt.show()


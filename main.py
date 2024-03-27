
#import the image and image processing from model folder
from model.image_ import Image
from model.image_processor import imageProcessor
import cv2
import numpy as np

#initialize the image object

def main():
    # Create an instance of ImageProcessor
    processor = imageProcessor()

    # Load an image
    image_obj = Image(data=cv2.imread('cat.jpg'), width=500, height=500)

    # Apply Canny edge detection using the ImageProcessor
    result = processor.apply_canny(image_obj, kernel_size=2, low_threshold=20, high_threshold=100)

    # Display the resulting image
    cv2.imshow("Canny Image", result.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
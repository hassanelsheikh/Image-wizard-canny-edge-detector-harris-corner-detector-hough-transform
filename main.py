from model import image
from model import image_processor
import cv2

processor = image_processor

image_obj = image(data = cv2.imread('image.jpg'), width = 500, height = 500)

result = processor.apply_canny(image_obj, kernel_size=5, low_threshold=50, high_threshold=150)

cv2.imshow('Canny Edge Detection', result)
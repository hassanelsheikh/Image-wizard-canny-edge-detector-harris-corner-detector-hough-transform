### Canny Edge Detector

#### Project Overview
This project implements an image processing application with a graphical user interface (GUI) using PyQt5. The application provides functionalities such as applying the Canny edge detector, Harris corner detector, and Hough line detector to images loaded from the user's filesystem. It aims to provide a user-friendly interface for performing basic image processing operations.

#### Dependencies
- Python 3.x
- PyQt5
- OpenCV (cv2)
- NumPy

#### Installation
1. Make sure you have Python 3.x installed on your system.
2. Install the required dependencies using pip:
    ```
    pip install PyQt5 opencv-python numpy
    ```
3. Clone or download the project repository to your local machine.

#### Usage
1. Navigate to the project directory.
2. Run the `main.py` script:
    ```
    python main.py
    ```
3. The application window will open, allowing you to browse for an image file using the "Browse" button.
4. Once an image is selected, you can apply various image processing operations using the provided buttons:
    - **Apply Canny**: Applies the Canny edge detector to the image.
    - **Apply Harris Transform**: Applies the Harris corner detector to the image.
    - **Apply Hough Transform**: Applies the Hough line detector to the image.
5. Adjust parameters as needed using the input fields provided.
6. The processed image will be displayed in the application window.

#### Notes
- Ensure that the image files you intend to process are in `.jpg` or `.png` format.
- This project serves as a basic demonstration of image processing techniques and can be extended with additional functionalities or optimizations as needed.

#### Contributors
- Hassan Elsheikh
- Asmaa Khalid
- Ammar Yasser
- Nada Alfowey

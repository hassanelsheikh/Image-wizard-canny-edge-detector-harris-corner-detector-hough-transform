import numpy as np

def apply_harris(image, k=0.04, threshold=0.01):
    gray = image.astype(np.float32)
    Ix = np.gradient(gray, axis=1)
    Iy = np.gradient(gray, axis=0)
    
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    
    # Gaussian filter for summing within a window
    Sxx = cv2.GaussianBlur(Ixx, (5, 5), 1)
    Syy = cv2.GaussianBlur(Iyy, (5, 5), 1)
    Sxy = cv2.GaussianBlur(Ixy, (5, 5), 1)
    
    # Harris response calculation
    detM = Sxx * Syy - Sxy**2
    traceM = Sxx + Syy
    R = detM - k * (traceM**2)
    
    # Threshold and non-max suppression
    corners = (R > threshold * R.max()) & (R == cv2.dilate(R, None))
    
    return corners

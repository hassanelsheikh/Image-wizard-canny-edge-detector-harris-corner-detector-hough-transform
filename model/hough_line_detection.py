import numpy as np

def apply_hough(edges, rho_res=1, theta_res=np.pi/180, threshold=100):
    height, width = edges.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.arange(-np.pi, np.pi, theta_res)
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # Fill the accumulator
    y_idxs, x_idxs = np.nonzero(edges)
    for (x, y) in zip(x_idxs, y_idxs):
        for idx_theta, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, idx_theta] += 1
    
    # Find lines
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator > threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
    
    return lines

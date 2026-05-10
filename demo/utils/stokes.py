import numpy as np

def process_images_stokes(I0, I45, I90, I135, A=None):
    """
    Recover 3 Stokes parameters from 4 states of polarization.
    DOLP, angle of linear polarization are also provided.
    All images are mapped to [0 1] range.
    
    Parameters:
    I0, I45, I90, I135: Input polarization images
    A: measurement matrix from calibration (optional, uses ideal matrix if None)
        Ideal matrix A = [[1, 1, 0, 0],
                         [1, 0, 1, 0], 
                         [1, -1, 0, 0],
                         [1, 0, -1, 0]]
    
    Returns:
    S0_scaled, S1_scaled, S2_scaled, DOLP_scaled, AOLP_scaled
    """
    
    # Define ideal measurement matrix from MATLAB comments
    if A is None:
        A = np.array([[1, 1, 0, 0],
                      [1, 0, 1, 0], 
                      [1, -1, 0, 0],
                      [1, 0, -1, 0]], dtype=np.float32)
    
    # Calculate Stokes parameters using calibration matrix A
    # Reshape images for matrix multiplication
    height, width = I0.shape[:2]
    if len(I0.shape) == 3:
        # For color images, process first channel
        I_reshaped = np.stack([I0[:,:,0].flatten(), 
                              I45[:,:,0].flatten(), 
                              I90[:,:,0].flatten(), 
                              I135[:,:,0].flatten()], axis=0)
    else:
        # For grayscale images
        I_reshaped = np.stack([I0.flatten(), 
                              I45.flatten(), 
                              I90.flatten(), 
                              I135.flatten()], axis=0)
    
    # Apply calibration matrix
    S = np.dot(A.T, I_reshaped).T
    S0 = S[:, 0].reshape(height, width) / 4
    S1 = S[:, 1].reshape(height, width)
    S2 = S[:, 2].reshape(height, width)
    
    # Scaling to [0, 1] range
    scale = [0, 1]
    S0_scaled = S0 * (scale[1] - scale[0]) / scale[1] + scale[0]
    S1_scaled = (S1 + scale[1]) * (scale[1] - scale[0]) / (2 * scale[1]) + scale[0]
    S2_scaled = (S2 + scale[1]) * (scale[1] - scale[0]) / (2 * scale[1]) + scale[0]
    
    # Calculate DOLP and AOLP
    a = S1**2
    b = S2**2
    DOLP_2 = np.sqrt(a + b)
    
    # Avoid division by zero
    DOLP = np.divide(DOLP_2, S0, out=np.zeros_like(DOLP_2), where=S0!=0)
    DOLP_scaled = DOLP * (scale[1] - scale[0]) / 2 + scale[0]
    
    AOLP = 0.5 * np.arctan2(S2, S1)
    AOLP_scaled = (AOLP + np.pi/2) * (scale[1] - scale[0]) / np.pi + scale[0]
    
    # Remove NaN and inf values, clamp DOLP to [0, 1]
    DOLP_scaled = np.clip(DOLP_scaled, 0, 1)
    DOLP_scaled = np.nan_to_num(DOLP_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    
    return S0_scaled, S1_scaled, S2_scaled, DOLP_scaled, AOLP_scaled

def psnr(x, y):
    return 10 * np.log10(255**2 / np.mean((x - y) ** 2))

def psnr_1(x, y):
    return 10 * np.log10(1 / np.mean((x - y) ** 2))

def mean_angle_error(x, y):
    x = np.mod(x, 1)
    y = np.mod(y, 1)

    d0 = (x - y)**2
    dp = (x + 1 - y)**2
    dn = (x - 1 - y)**2

    dif = np.minimum(np.minimum(d0, dp), dn)

    mse = np.mean(dif)
    return np.sqrt(mse) * 180
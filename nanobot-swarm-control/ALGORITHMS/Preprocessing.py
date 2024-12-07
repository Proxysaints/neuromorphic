import numpy as np
import cupy as cp
from cuml.dnn import Conv3D, MaxPool3D, Relu, Flatten, Dense
from cuml.dnn import Sequential
from scipy.spatial.transform import Rotation as R
from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler, StandardScaler as cuStandardScaler
from cuml.decomposition import PCA as cuPCA
from concurrent.futures import ThreadPoolExecutor

class Preprocessor:
    """
    A modular preprocessing pipeline for 3D position data, optimized for memory and computational efficiency.
    """
    def __init__(self, methods=None, use_gpu=False):
        self.methods = methods if methods else []
        self.use_gpu = use_gpu

    def add_method(self, method, params=None):
        """
        Add a preprocessing method to the pipeline.
        """
        self.methods.append((method, params if params else {}))

    def apply(self, data):
        """
        Apply the preprocessing methods sequentially on the data.
        """
        for method, params in self.methods:
            data = method(data, **params)
            if self.use_gpu:
                cp.cuda.Device(0).mem_free()  # Clear GPU memory after each step
        return data

# Optimized Preprocessing Methods with GPU Integration

def cudnn_gaussian_filter(data, sigma=1):
    """
    Apply a Gaussian filter using cuDNN for GPU acceleration.
    """
    # Define Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # Ensure kernel size covers most of the Gaussian distribution
    x = cp.linspace(-3 * sigma, 3 * sigma, kernel_size)
    kernel = cp.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()

    # Convert kernel to 2D for convolution
    kernel = cp.outer(kernel, kernel)
    kernel = kernel[:, :, cp.newaxis, cp.newaxis]  # Shape for cuDNN: (H, W, C_in, C_out)

    # Prepare data for 2D convolution
    data = cp.asarray(data, dtype=cp.float32)
    data = data[:, :, cp.newaxis, cp.newaxis]  # Shape: (N, H, W, C)

    # Initialize cuDNN convolution layer
    conv = Conv3D(kernel_size=(kernel_size, kernel_size, kernel_size),
                  stride=(1, 1, 1),
                  padding=(kernel_size // 2, kernel_size // 2, kernel_size // 2),
                  channels=1)

    # Apply Gaussian smoothing
    smoothed = conv.forward(data, kernel)

    if cp.cuda.is_available():
        cp.cuda.Device(0).mem_free()  # Clear GPU memory after convolution
    return smoothed.reshape(data.shape[:-1])

def min_max_normalize(data):
    """
    Normalize data to a [0, 1] range using cuML (GPU accelerated).
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    scaler = cuMinMaxScaler()
    normalized_data = scaler.fit_transform(data).get()  # Convert back to CPU array after transformation

    if cp.cuda.is_available():
        cp.cuda.Device(0).mem_free()  # Clear GPU memory after normalization
    return normalized_data

def z_score_normalize(data):
    """
    Standardize data to have a mean of 0 and a standard deviation of 1 using cuML.
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    scaler = cuStandardScaler()
    standardized_data = scaler.fit_transform(data).get()  # Convert back to CPU array after transformation

    if cp.cuda.is_available():
        cp.cuda.Device(0).mem_free()  # Clear GPU memory after standardization
    return standardized_data

def apply_pca(data, n_components=2):
    """
    Apply PCA using cuML for GPU acceleration.
    """
    data = cp.asarray(data, dtype=cp.float32)  # Ensure GPU compatibility
    pca = cuPCA(n_components=n_components)
    pca_result = pca.fit_transform(data).get()  # Convert back to CPU array after transformation

    if cp.cuda.is_available():
        cp.cuda.Device(0).mem_free()  # Clear GPU memory after PCA
    return pca_result

def interpolate_missing(data, method='linear'):
    """
    Interpolate missing data points in 3D position data.
    """
    for dim in range(data.shape[1]):
        mask = np.isnan(data[:, dim])
        indices = np.arange(len(data))
        if np.any(mask):
            if method == 'linear':
                interpolator = interp1d(indices[~mask], data[~mask, dim], kind='linear', fill_value='extrapolate')
            elif method == 'spline':
                interpolator = UnivariateSpline(indices[~mask], data[~mask, dim], s=0)
            data[mask, dim] = interpolator(indices[mask])
    return data

def normalize_positions(data):
    """
    Translate 3D positions to the origin (center the data).
    """
    centroid = np.mean(data, axis=0)
    return data - centroid

def rotate_coordinates(data, axis, angle):
    """
    Rotate 3D coordinates around a given axis.
    """
    rotation_matrix = {
        'x': R.from_euler('x', angle, degrees=True).as_matrix(),
        'y': R.from_euler('y', angle, degrees=True).as_matrix(),
        'z': R.from_euler('z', angle, degrees=True).as_matrix()
    }.get(axis)

    if rotation_matrix is None:
        raise ValueError("Invalid axis. Choose from 'x', 'y', 'z'.")

    return np.dot(data, rotation_matrix.T)


# Define a Hook to capture and display output during training
class Hook:
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.outputs = []

    def __call__(self, layer_input, layer_output):
        self.outputs.append(layer_output)

    def get_outputs(self):
        return self.outputs


# Create a simple 3D CNN model
def create_cnn_model(input_shape):
    model = Sequential()

    # Adding Conv3D layer
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    
    # Hook to capture the output of Conv3D
    conv3d_hook = Hook("Conv3D")
    model.add(HookLayer(conv3d_hook))

    # Adding Max Pooling layer
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    # Adding another Conv3D layer
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))

    # Hook to capture the output of second Conv3D
    conv3d_hook2 = Hook("Conv3D-2")
    model.add(HookLayer(conv3d_hook2))

    # Adding Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model, conv3d_hook, conv3d_hook2


# Example Usage with Optimizations

# Define input shape for 3D CNN (e.g., 64x64x64x1)
input_shape = (64, 64, 64, 1)

# Create Preprocessor
methods = [
    (cudnn_gaussian_filter, {'sigma': 2}),
    (min_max_normalize, {}),
    (apply_pca, {'n_components': 3}),
]
preprocessor = Preprocessor(methods, use_gpu=True)

# Create CNN model
model, conv3d_hook, conv3d_hook2 = create_cnn_model(input_shape)

# Example data for training (random data for demonstration)
data = cp.random.random((10, 64, 64, 64, 1))  # 10 samples of 64x64x64 3D data

# Apply preprocessing
processed_data = preprocessor.apply(data)

# Perform a forward pass through the CNN (dummy forward pass)
model.predict(processed_data)

# Display the output from the hooks
print("Output from Conv3D layer:")
print(conv3d_hook.get_outputs())

print("Output from second Conv3D layer:")
print(conv3d_hook2.get_outputs())
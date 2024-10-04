#utils_semi.py
# Utils for semi-supervised learning
from utils.Hilbert_curve import HilbertCurve
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
'''
This code (idea) is from Hugo Gangloff's github repo:
'''

'''
1D sequence <-> image, using an hilbert curve
requires the sequence have length equal to a power of 2
based on hilbertcurve.py from the package https://github.com/galtay/hilbertcurve
'''
np.random.seed(19594)
def generate_noisy_image(image_file_path, output_size, noise_mean, noise_stddev, use_multiplicative_noise):
    """
    Reads an image file, creates a noisy version of the image, and generates a mask for missing pixels.

    Parameters:
    image_file_path (str): Path to the input image file.
    output_size (int): Size of the output image.
    noise_mean (float): Mean of the noise.
    noise_stddev (float): Standard deviation of the noise.
    missing_pixel_prob (float): Probability of a pixel being missing.
    use_multiplicative_noise (bool): Whether to use multiplicative noise.

    Returns:
    noisy_image (ndarray): Noisy version of the input image.
    original_image (ndarray): Original input image (binary)
    """
    # Check if image file exists
    if not os.path.isfile(image_file_path):
        raise FileNotFoundError(f"File '{image_file_path}' does not exist.")

    # Load image
    original_image = plt.imread(image_file_path)
    original_image = cv2.resize(original_image, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    _, original_image = cv2.threshold(original_image, 127, 1, cv2.THRESH_BINARY)

    # Convert image to chain
    chain = image_to_chain(original_image)

    # Generate noisy chain
    noisy_chain = np.zeros(len(chain))
    if use_multiplicative_noise:
        z = np.random.randn(len(chain))
        for t in range(len(chain)):
            noisy_chain[t] = z[t] * np.random.normal(noise_mean * chain[t], noise_stddev, 1)
    else:
        for t in range(len(chain)):
            if t == 0:
                noisy_chain[t] = np.random.normal(np.sin(noise_mean * chain[t]), noise_stddev, 1)
            else:
                noisy_chain[t] = np.random.normal(np.sin(noise_mean * chain[t] + noisy_chain[t-1]), noise_stddev, 1)

    # Convert noisy chain to image
    noisy_image = chain_to_image(noisy_chain).reshape(output_size, output_size)
    noisy_image = cv2.normalize(noisy_image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    return noisy_image, original_image

def get_input(prompt, input_type):
    ''' 
    This function is used to get an input from the user and check if it is valid
    '''
    while True:
        try:
            user_input = input_type(input(prompt))
            return user_input
        except ValueError:
            print('Invalid input. Please try again.')
            
def read_create_x_y_ymiss(image_file, size, mu, sigma, p, multi_noise):
    '''
    image_file: path to the image
    size: size of the image
    mu: mean of the normal distribution
    sigma: std of the normal distribution
    p: probability of a pixel to be missing
    multi_noise: boolean, if True we use multiplicative noise, otherwise additive noise
    return:
    image_xx: image with noise (size*size)
    image: original image (size*size)
    label_miss: input image with missing labels (size*size)
    '''
    img = plt.imread(image_file)//255
    image = cv2.resize(img, (size, size)).astype('int16')
    chain = image_to_chain(image)
    x = np.zeros(len(chain))
    if multi_noise:
        z = np.random.normal(size = len(chain))
        for t in range(len(chain)):
            x[t] = z[t]*np.random.normal(mu*(chain[t]), sigma, 1)
    else:
        for t in range(len(chain)):
            if t == 0:
                x[t] = np.random.normal(np.sin(mu*(chain[t])), sigma, 1)
            else:
                x[t] = np.random.normal(np.sin(mu*(chain[t]) + x[t-1]), sigma, 1)

    image_x = chain_to_image(x).reshape(size, size)
    image_xx = (image_x.astype('float') - np.min(image_x)) / (np.max(image_x) - np.min(image_x))
    mask_missing = np.random.choice([0,1], size=image.shape, p=[p, 1-p])
    label_miss = image.copy()
    label_miss[mask_missing==0] = -1
    return image_xx, image, label_miss

def create_missing_labels(img, p):
    '''
    p = probability of missing a pixel
    '''
    mask_missing = np.random.choice([0,1], size=img.shape, p=[p, 1-p])
    label_miss = img.copy()
    label_miss[mask_missing==0] = -1
    return label_miss

def creation_noisy_image(img, mu, sigma,p,  size = 28):
    '''
    image: image of size 28*28
    Binary mask: 0 if the pixel is the background, 1 otherwise (part of the number)
    
    chain: chain of the image (Hilbert curve) which is a vector of size size*size 
    z =  N(0, 1) # random variable wich
    x = z* N(mu, sigma)

    return:
    image_x: image with noise (28*28)
    label_miss: input image with missing labels (28*28)
    '''
    # We use np.pad to add a border of 0 to the image and increase the size to 32*32 beacause:
    # 1D sequence <-> image, using an hilbert curve requires the sequence have length equal to a --power of 2--    
    # This is not a general solution, but it works for the MNIST dataset
    image = np.pad(img.reshape(size, size), ((2,2), (2,2)), 'constant')
    chain = image_to_chain(image)
    # More complex noisy image
    # mu*(chain[i]) is the mean of the normal distribution and it changes for each pixel with respect to the label at that pixel
    # z = np.random.normal(size = len(chain))
    # x = np.array([  z[i]*np.random.normal(mu*(chain[i]), sigma, 1) for i in range(len(chain)) ])
    x = np.array([np.random.normal(mu*(chain[i]), sigma, 1) for i in range(len(chain)) ])
    # In order to recuperate the original image we need to remove the first 2 and last 2 elements of the chain
    image_x = chain_to_image(x)[2:(2+size), 2:(2+size)]
    mask_missing = np.random.choice([0,1], size=(size, size), p=[p, 1-p])
    label_miss = img.reshape(size, size).copy()
    label_miss[mask_missing==0] = -1

    return [image_x, label_miss]

def semi_sup_preprocessing(list_images,p, mu, sigma, size):
    '''
    list_images: list of images (1, size*size)
    p: probability of a pixel to be missing
    mu: mean of the normal distribution
    sigma: std of the normal distribution

    return:
    x: list of noisy images of length size*size
    y: list of images with missing labels  of length size*size
    '''
    x_y = [creation_noisy_image(im, mu, sigma,p, size ) for im in list_images]
    return x_y
    #return [ x[0] for x in x_y], [ y[1] for y in x_y]



def dim_image(list_image, size = 28):
    '''
    This function is used to reshape the images specially for the particular case of the MNIST dataset
    list_image: list of images (1, size*size)
    return: list of images (size, size)
    '''
    return [img.reshape(size, size) for img in list_image]



def get_hilbertcurve_path(image_border_length):
    '''
    Given image_border_length, the length of a border of a square image, we
    compute path of the hilbert peano curve going through this image

    Note that the border length must be a power of 2.

    Returns a list of the coordinates of the pixel that must be visited (in
    order !)
    '''
    path = []
    p = int(np.log2(image_border_length))
    hilbert_curve = HilbertCurve(p, 2)
    path = []
    #print("Compute path for shape ({0},{1})".format(image_border_length,
        # image_border_length))
    for i in range(image_border_length ** 2):
        coords = hilbert_curve.coordinates_from_distance(i)
        path.append([coords[0], coords[1]])

    return path

def chain_to_image(X_ch, masked_peano_img=None):
    '''
    X_ch is an unidimensional array (a chain !) whose length is 2^(2*N) with N non negative
    integer.
    We transform X_ch to a 2^N * 2^N image following the hilbert peano curve
    '''
    if masked_peano_img is None:
        image_border_length = int(np.sqrt(X_ch.shape[0]))
        path = get_hilbertcurve_path(image_border_length)
        masked_peano_img = np.zeros((image_border_length, image_border_length))
    else:
        image_border_length = masked_peano_img.shape[0]
        path = get_hilbertcurve_path(image_border_length)
        
    X_img = np.empty((image_border_length, image_border_length))
    offset = 0
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_img[coords[0], coords[1]] = X_ch[idx - offset]
        else:
            offset += 1
            X_img[coords[0], coords[1]] = -1

    return X_img

def image_to_chain(X_img, masked_peano_img=None):
    '''
    X_img is a 2^N * 2^N image with N non negative integer.
    We transform X_img to a 2^(2*N) unidimensional vector (a chain !)
    following the hilbert peano curve
    '''
    path = get_hilbertcurve_path(X_img.shape[0])

    if masked_peano_img is None:
        masked_peano_img = np.zeros((X_img.shape[0], X_img.shape[1]))

    X_ch = []
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_ch.append(X_img[coords[0], coords[1]])

    return np.array(X_ch)


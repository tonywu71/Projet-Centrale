import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter

from scipy.ndimage import gaussian_filter

import os

from itertools import permutations

from IPython.display import clear_output


# ---------------- Image utilities ----------------

def read_img(filename):
    '''Gets the array of an image file.'''
    return Image.open(filename)


def split_img(im_shuffled, nb_lines, nb_cols, x_margin=0, y_margin=0):
    '''Returns a dictionary of all the pieces of the puzzle.
    x_margin and y_margin permits to get smaller crops to take into account
    the unclean puzzles. The final image will have 2*x_margin less pixels
    wrt to x and 2*y_margin less pixels wrt to y.

    
    Args:
    - im_suffled (Image object)
    - nb_lines (int)
    - nb_cols (int)
    - x_margin (positive int)
    - y_margin (positive int)

    Returns:
    - cropped (dict)
    '''

    assert x_margin >= 0, 'x_margin should be positive'
    assert y_margin >= 0, 'y_margin should be positive'


    w, h = im_shuffled.size # w, h = width, height
    
    # For one piece of the puzzle
    w_piece = w / nb_cols
    h_piece = h / nb_lines
    
    cropped = {}
    
    for i in range(nb_lines):
        for j in range(nb_cols):
            left = i * w_piece + x_margin
            top = j * h_piece + y_margin
            right = (i + 1) * w_piece - x_margin
            bottom = (j + 1) * h_piece - y_margin
            
            cropped[(i,j)] = im_shuffled.crop((left, top, right, bottom))
    
    return cropped

def save_cropped(cropped, filtered=False):
    '''Save as files all the pieces of the puzzle in the cropped directory.
    The files are named accordingly to 'i-j.jpg' where i and j are the coordinates
    of the pieces of the puzzle in the PIL coods system.

    Args:
    - cropped ({key: image})
    Returns:
    - None
    '''
    
    if filtered:
        for (i,j), im in cropped.items():
            filename = f'{i}-{j}.jpg'
            
            # For the untouched image:
            filepath = os.path.join('cropped', filename)
            im.save(filepath)

            # For the filtered image:
            filepath_filtered = os.path.join('cropped_filtered', filename)
            arr = gaussian_filter(np.array(im), sigma=1)
            im_filtered = Image.fromarray(arr)
            im_filtered.save(filepath_filtered)
            
    
    else:
        for (i,j), im in cropped.items():
            filename = f'{i}-{j}.jpg'
            filepath = os.path.join('cropped', filename)
            im.save(filepath)
    
    print('Images successfully saved.')
    return



# ---------------- Operations on images ----------------

def get_current_permutations(cropped):
    ''' Generator that yields a dictionary giving the mapping from the current
    configuration to the shuffled puzzle.
    
    Args:
    - cropped (Image object)

    Returns:
    - generator object
    '''
    
    list_keys = list(cropped.keys())
    
    for config in permutations(list_keys):
        map_config = dict(zip(list_keys, config))
        yield map_config

def grad_x(im1, im2):
    '''Return the discrete horizontal gradient. im2 must be to the right of im1.
    Args:
    - im1 (Image object)
    - im2 (Image object)

    Returns:
    - grad_x_val (float)

    NB: numpy and PIL don't share the same coordinate system! '''
    
    ## Conversion from Image object into numpy arrays
    arr1 = np.array(im1)
    arr2 = np.array(im2)
    
    min_x = min(arr1.shape[0], arr2.shape[0])
    min_y = min(arr1.shape[1], arr2.shape[1])
    
    arr1 = arr1[:min_x,:min_y,:]
    arr2 = arr2[:min_x,:min_y,:]
    
    ## Computation of the horizontal gradient at the frontier
    return np.sum(np.square(arr1[-1,:,:] - arr2[0,:,:]))

def grad_y(im1, im2):
    '''Return the discrete horizontal gradient. im2 must be to the right of im1.
    Args:
    - im1 (Image object)
    - im2 (Image object)

    Returns:
    - grad_y_val (float)

    NB: numpy and PIL don't share the same coordinate system! '''
    
    ## Conversion into numpy arrays
    arr1 = np.array(im1)
    arr2 = np.array(im2)
    
    min_x = min(arr1.shape[0], arr2.shape[0])
    min_y = min(arr1.shape[1], arr2.shape[1])
    
    arr1 = arr1[:min_x,:min_y,:]
    arr2 = arr2[:min_x,:min_y,:]
    
    ## Computation of the horizontal gradient at the frontier
    return np.sum(np.square(arr1[:,0,:] - arr2[:,-1,:]))

def mean_grad(cropped, nb_lines, nb_cols):
    '''Returns the mean of the grad in both horizontally and vertically.'''
    res = 0
    for j in range(nb_lines):
        for i in range(nb_cols-1):  
            res += grad_x(cropped[(i,j)], cropped[(i+1,j)])
    return res / (nb_lines * nb_cols)

def read_cropped_im(i, j):
    ''' Returns the given image loaded from the cropped folder
    as an Image object.'''
    im = Image.open(os.path.join('cropped', f'{i}-{j}.jpg'))
    return im

def get_concat_h(im1, im2):
    ''' Returns the horizontal concatenation of im1 and im2.'''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    ''' Returns the vertical concatenation of im1 and im2.'''
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def create_config(map_config, nb_lines, nb_cols):
    ''' Returns an image according to the given configuration.

    Strategy:
    1) We'll start concatenate each line of the final configuration.
    2) Only then are we going to concatenate those lines vertically.
    
    Args:
    - map_config (dict): dictionary mapping from the current configuration
        to the shuffled puzzle.
    - nb_lines (int)
    - nb_cols (int)

    Returns:
    - an Image object
    '''
    
    ## Step 1:
    list_lines = []
    
    for j in range(nb_lines): # We process line by line...
        # We start from the left-most image.
        current_im = read_cropped_im(*map_config[(0,j)]) # NB: The * allows to unpack the given tuple
        
        for i in range(1, nb_cols): # For each piece of the line...
            new_piece = read_cropped_im(*map_config[(i,j)]) # we get the juxtaposed piece just right to the previous one
            
            current_im = get_concat_h(current_im, new_piece)
        
        list_lines.append(current_im)
    
    # Now we can vertically concatenate the obtained lines.
    current_im = list_lines[0]
    
    for idx, img_line in enumerate(list_lines):
        if idx == 0:
            pass
        else:
            current_im = get_concat_v(current_im, img_line)
    
    return current_im

def brute_force(cropped):
    ''' Brute force solve. VERY SLOW!!!
    Saves all possibles configurations in the 'output' folder.
    
    Args:
    - cropped {dict}

    Returns:
    - None
    '''
    for idx, map_config in enumerate(get_current_permutations(cropped)):
        print(f'Current configuration: {idx}')
        im_config = create_config(map_config, nb_lines, nb_cols)
        filename = f'{idx}.jpg'
        filepath = os.path.join('outputs', filename)
        im_config.save(filepath)
        clear_output(wait=True)
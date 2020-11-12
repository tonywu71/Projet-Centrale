import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import os
from itertools import permutations
from IPython.display import clear_output
from copy import deepcopy
from collections import namedtuple

# ---------------- Image utilities ----------------

def read_img(filename):
    '''Gets the array of an image file.'''
    return Image.open(filename)

def split_img(im_shuffled, nb_lines, nb_cols, margin=(0, 0)):
    '''Returns a dictionary of all the pieces of the puzzle.
    Use optional argument margin in order to have more smooth cuts.
    
    Args:
    - im_suffled (Image object)
    - nb_lines (int)
    - nb_cols (int)
    - margin ((x_margin, y_margin))

    Returns:
    - cropped (dict)
    '''
    w, h = im_shuffled.size # w, h = width, height
    
    # For one piece of the puzzle
    w_piece = (w / nb_cols)
    h_piece = (h / nb_lines)
    
    cropped = {}

    x_margin, y_margin = margin
    
    for i in range(nb_lines):
        for j in range(nb_cols):
            left = i * w_piece + x_margin / 2
            top = j * h_piece + y_margin / 2
            right = (i + 1) * w_piece - x_margin / 2
            bottom = (j + 1) * h_piece -  y_margin / 2
            
            cropped[(i,j)] = im_shuffled.crop((left, top, right, bottom))
    
    return cropped

def display_image(img, nb_lines, nb_cols, title='', figsize=(5,6)):
    '''Show the image with custom ticks for both x and y axis, making piece 
    identification easier.
    
    Args:
    - img (Image object)
    - nb_lines (int)
    - nb_cols (int)
    Returns:
    - None
    '''

    plt.figure(figsize=figsize)

    xticks_location = (img.width / nb_cols) / 2 + np.linspace(0, img.width, nb_cols+1)
    yticks_location = (img.height / nb_lines) / 2 + np.linspace(0, img.height, nb_lines+1)
    plt.xticks(xticks_location, range(nb_cols))
    plt.yticks(yticks_location, range(nb_lines))
    if title:
        plt.title(title)

    plt.imshow(img)
    return


def display_cropped(cropped, nb_lines, nb_cols, title='', figsize=(5,6)):
    '''Show the image with custom ticks for both x and y axis, making piece 
    identification easier.
    
    Args:
    - cropped ({key: image})
    - nb_lines (int)
    - nb_cols (int)
    Returns:
    - None
    '''

    img = cropped_to_img(cropped, nb_lines, nb_cols)
    display_image(img, nb_lines, nb_cols, title='', figsize=figsize)
    return




def save_cropped(cropped):
    '''Save as file all the pieces of the puzzle in the cropped directory.
    The files are named accordingly to 'i-j.jpg' where i and j are the coordinates
    of the pieces of the puzzle in the PIL coods system.

    Args:
    - cropped ({key: image})
    Returns:
    - None
    '''
    
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
    - cropped ({key: image})

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
    '''Return the discrete horizontal gradient. im2 must be below im1.
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
    
    ## Computation of the vertical gradient at the frontier
    return np.sum(np.square(arr1[:,0,:] - arr2[:,-1,:]))

def mean_grad(cropped, nb_lines, nb_cols):
    '''Returns the mean of the gradient both horizontally and vertically.'''
    res = 0
    for j in range(nb_lines):
        for i in range(nb_cols-1):  
            res += grad_x(cropped[(i,j)], cropped[(i+1,j)])
    
    for i in range(nb_cols):
        for j in range(nb_lines-1):  
            res += grad_y(cropped[(i,j)], cropped[(i,j+1)])
    
    return res / 2

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

def config_to_img(map_config, nb_lines, nb_cols):
    ''' Returns an image according to the given configuration.

    Strategy:
    1) We'll start concatenate each line of the final configuration.
    2) Only then are we going to concatenate those lines vertically.
    
    Args:
    - map_config ({(old_coords): (new_coords), ...}): dictionary mapping from the current configuration
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

def cropped_to_img(cropped, nb_lines, nb_cols):
    ''' Returns an image according to the given configuration.

    Strategy:
    1) We'll start concatenate each line of the final configuration.
    2) Only then are we going to concatenate those lines vertically.
    
    Args:
    - cropped ({(x, y): Image Object, ...}): dictionary mapping from the current configuration
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
        current_im = cropped[(0,j)] # NB: The * allows to unpack the given tuple
        
        for i in range(1, nb_cols): # For each piece of the line...
            new_piece = cropped[(i,j)] # we get the juxtaposed piece just right to the previous one
            
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

def get_grad_orientation(im_1, im_2, orientation):
    '''Returns the gradient considering im_1 as the reference image and im_2
    concatenated right next to im_1 with the given orientation. The gradient is
    calculated at the limit.
    
    Orientation must be in ['N', 'E', 'W', 'S'].
    
    Args:
    - im_1 (Image object)
    - im_2 (Image object)
    - orientation (str)

    Returns:
    - grad (float)
    
    '''
    
    assert orientation in ['N', 'E', 'W', 'S'], 'Given input for orientation not understood.'
    
    if orientation == 'E':
        return grad_y(im_1, im_2)
    elif orientation == 'W':
        return grad_y(im_2, im_1)
    elif orientation == 'S':
        return grad_x(im_1, im_2)
    elif orientation == 'N':
        return grad_x(im_2, im_1)

def getBestConfig(cropped, nb_lines, nb_cols):
    '''Returns a dictionary that contains another dictionary that gives
    the ID of the piece with the best gradient according to the current direction
    ('N' for North, 'E' for East, 'W' for West and 'S' for South) which is used
    as the key. Moreover, one can access the gradient value using the 'grad_N'
    (or grad_E, etc...) key.
    
    Args:
    - cropped {key: image}
    - nb_lines (int)
    - nb_cols (int)

    Returns:
    - dicBestConfig {curr_piece: {'N': (x_best_N, y_best_N), ..., 'grad_N': min_grad_N, ...}}
    '''
    
    dicBestConfig = {}
    orientations = ['N', 'E', 'W', 'S']
    
    for curr_piece_ID in cropped.keys(): # For every piece of the puzzle...
        # Creating an empty dict for the current piece.
        dicBestConfig[curr_piece_ID] = {}
        
        for orientation in orientations: # For every single of the 4 orientation...
            # Preparing the key for storing the gradient.
            grad_orientation = 'grad_' + orientation
            
            # Variables to store the best candidate.
            min_grad = np.inf
            best_piece_ID = None
            
            for piece_ID in cropped.keys(): # For every piece of the puzzle...
                if piece_ID == curr_piece_ID: # We skip duplicates...
                    continue
                else: # If we have two different pieces...
                    curr_grad = get_grad_orientation(
                        im_1=cropped[curr_piece_ID], 
                        im_2=cropped[piece_ID], 
                        orientation=orientation)
                    if curr_grad < min_grad: # If it's a better candidate...
                        # Overwriting the previous variables.
                        min_grad = curr_grad
                        best_piece_ID = piece_ID
            
            dicBestConfig[curr_piece_ID][orientation] = best_piece_ID
            dicBestConfig[curr_piece_ID][grad_orientation] = min_grad
    
    return dicBestConfig

def getOrderedConfigsByConfig(dicBestConfig, orientation, reverse=False):
    '''Returns a sorted list of elements from dicBestConfig.values().
    
    Args:
    - dicBestConfig (dict)
    - orientation (str)
    - reverse (bool)

    Returns:
    - ordered_list [(value from dicBestConfig.values()) ordered by the
    gradient according to the given orientation]
    '''

    orientations = ['N', 'E', 'W', 'S']
    assert orientation in ['N', 'E', 'W', 'S'], 'Given input for orientation not understood.'
    
    grad_orientation_key = 'grad_' + orientation
    return sorted(dicBestConfig.items(), key=lambda x: x[1][grad_orientation_key], reverse=reverse)

def getOrderedConfigs(dicBestConfig, reverse=False):
    """Returns a sorted list of elements from dicBestConfig.values().
    We don't consider orientation in this function.
    
    Args:
    - dicBestConfig (dict)
    - reverse (bool)

    Returns:
    - ordered_list (list): list of named tuples of the form 
        (start, end, orientation, score)
    """

    list_temp = []
    list_orientations = ['N', 'E', 'W', 'S']

    # Creating a namedtuple for convenience:
    Config = namedtuple('Config', ['start', 'end', 'orientation', 'score'])

    for start, val in dicBestConfig.items():
        for orientation in list_orientations:
            end = val[orientation]
            score = val['grad_' + orientation]

            list_temp.append(Config(start, end, orientation, score))
    
    ordered_list = sorted(list_temp, key=lambda x: x.score, reverse=reverse)
    return ordered_list





# ---------------- Brute force ----------------

def brute_force(cropped, nb_lines, nb_cols):
    ''' Brute force solve. VERY SLOW!!!
    Saves all possibles configurations in the 'output' folder.
    
    Args:
    - cropped {dict}
    - nb_lines (int)
    - nb_cols (int)

    Returns:
    - None
    '''
    for idx, map_config in enumerate(get_current_permutations(cropped)):
        print(f'Current configuration: {idx}')
        im_config = config_to_img(map_config, nb_lines, nb_cols)
        filename = f'{idx}.jpg'
        filepath = os.path.join('outputs', filename)
        im_config.save(filepath)
        clear_output(wait=True)



# ---------------- Manual solve ----------------
def config_switcher(cropped, nb_lines, nb_cols, coords_1, coords_2):
    '''Switch places for two pieces and return a new cropped dictionary.
    
    Args:
    - cropped
    - nb_lines
    - nb_cols
    - coords_1 (2-tuple): 1st piece to move
    - coords_2 (2-tuple): 2nd piece to move

    Returns:
    - new_cropped
    '''
    
    new_cropped = deepcopy(cropped)
    new_cropped[coords_1], new_cropped[coords_2] = new_cropped[coords_2], new_cropped[coords_1]
    
    return new_cropped

def config_switcher_helper(cropped, nb_lines, nb_cols, coords_1, coords_2):
    '''Show on the same plot the previous image and the new one after having
    the pieces switched places.
    
    Args:
    - cropped
    - nb_lines
    - nb_cols
    - coords_1 (2-tuple): 1st piece to move
    - coords_2 (2-tuple): 2nd piece to move

    Returns:
    - new_cropped
    '''
    
    plt.figure(figsize=(12, 10))



    plt.subplot(1, 2, 1)
    old_image = cropped_to_img(cropped, nb_lines, nb_cols)

    xticks_location = (old_image.width / nb_cols) / 2 + np.linspace(0, old_image.width, nb_cols+1)
    yticks_location = (old_image.height / nb_lines) / 2 + np.linspace(0, old_image.height, nb_lines+1)
    
    plt.xticks(xticks_location, range(nb_cols))
    plt.yticks(yticks_location, range(nb_lines))
    plt.imshow(old_image)
    plt.title('Old image')
    
    plt.subplot(1, 2, 2)
    new_cropped = config_switcher(cropped, nb_lines, nb_cols, coords_1, coords_2)
    new_image = cropped_to_img(new_cropped, nb_lines, nb_cols)
    plt.xticks(xticks_location, range(nb_cols))
    plt.yticks(yticks_location, range(nb_lines))
    plt.imshow(new_image)
    plt.title('New image')
    return


# ---------------- Backtracking ----------------
def get_next_location(nb_pieces, nb_lines, nb_cols):
    '''Returns the next coords (i,j) of the piece according to the
    completion strategy.

    Completion strategy:
        Adds a piece with increasing x and, if the x are the same,
        with increasing y. In other terms, we complete the puzzle from
        left to right and from top to bottom.'''
    
    # Get the previous coords (not trivial)
    if nb_pieces % nb_cols == 0: # If we have a full line...
        y = (nb_pieces // nb_cols) - 1
        x = nb_cols - 1
    else: #If the line isn't fully completed yet...
        y = nb_pieces // nb_cols
        x = (nb_pieces % nb_cols) - 1
    
    if x == nb_cols - 1: # If we are already at the end of a line...
        x_new = 0
        y_new = y + 1
    else: # If there is still some room on the line...
        x_new = x + 1
        y_new = y
    
    print(f'Added new piece at: ({x_new}, {y_new})')
    assert 0 <= x_new < nb_cols, 'Error with the x axis!'
    assert 0 <= y_new < nb_lines, 'Error with the y axis!'
    
    return (x_new, y_new)

def score(config, cropped, nb_lines, nb_cols):
    '''Computes the score of the current config, which is in this case
    the squared mean gradient with respect to x and y divided by the 
    total number of pieces in the puzzle.'''

    # In order to call the mean_grad function, we first
    # have to generate a dictionary that has the same
    # format as 'cropped': {(0, 0): <PIL.Image.Image>, ...}.

    # Currently, 'config' has the shape {(new_coords): (old_coords), ...}.

    # Next line allows to obtain the wanted dictionary.
    new_cropped = get_config_mapped(config=config, cropped=cropped)
    
    score = mean_grad(new_cropped, nb_lines, nb_cols)**2 / 2
    
    return score

def get_config_mapped(config, cropped):
    '''Converts a config dictionary to the same format as a cropped dictionary.
    Args:
    - config ({(new_coords): (old_coords), ...}): current configuration (not necessarily 
    completed)
    - cropped ({(0, 0): <PIL.Image.Image>, ...}): dictionary of every single piece 
    of the puzzle
    '''
    
    return {new_coords: cropped[old_coords] for new_coords, old_coords in config.items()}

def partial_score(partial_config, cropped, nb_lines, nb_cols):
    '''Computes the score of a partial configuration.'''
    
    res = 0
    config_mapped = get_config_mapped(config=partial_config, cropped=cropped)
    
    # Gradient wrt to x:
    for j in range(nb_lines):
        for i in range(nb_cols-1):
            if (i,j) in config_mapped.keys() and (i+1,j) in config_mapped.keys():
                res += grad_x(config_mapped[(i,j)], config_mapped[(i+1,j)])
    
    # Gradient wrt to y:
    for i in range(nb_cols):
        for j in range(nb_lines-1):
            if (i,j) in config_mapped.keys() and (i,j+1) in config_mapped.keys():
                res += grad_y(config_mapped[(i,j)], config_mapped[(i,j+1)])
    
    return (res/2)**2 / (nb_lines * nb_cols)

def solve_backtracking(cropped, nb_lines, nb_cols):
    '''
    Applies backtracking for building the puzzle.
    
    In what follows, 'config' is a dictionary with the
    shape {(x, y): (i, j), ...}, ie that links the current
    configuration to the suffled one.
    
    Args:
    - cropped ({(0, 0): <PIL.Image.Image>, ...}): dictionary of
    every single piece of the puzzle
    '''
    
    bestScore = np.inf
    bestSol = None
    
    nb_pieces_total = len(cropped)
    config = {}
    
    # ------ Auxiliary functions ------
    
    
    def is_terminal(config):
        '''Returns True if we have generated a complete
        solution for the puzzle.'''
        return len(config) == nb_pieces_total
    
    def is_promising(partial_config, bestScore):
        '''Returns True iif the gradient score of the partial configuration
        is lower or equal to bestScore.'''
        current_score = partial_score(partial_config, cropped, nb_lines, nb_cols)
        print(f'current_score: {current_score}')
        return current_score < bestScore
    
    def children(config, cropped, bestScore):
        '''Generator for a list of configurations that have one supplementary piece
        when compared to 'config'.
        
        Args:
        - config ({new_coords: old_coords, ...})
        - cropped ({(i,j): Image object, ...})
             
        Completion strategy:
        Adds a piece with increasing x and, if the x are the same,
        with increasing y. In other terms, we complete the puzzle from
        left to right and from top to bottom.'''
        
        # We get the location (i, j) of the next piece.
        nb_pieces = len(config)
        next_location = get_next_location(nb_pieces=nb_pieces, nb_lines=nb_lines, nb_cols=nb_cols)
        
        # config.values() contains the old coords that have already been used
        # cropped.keys() contains all the possible coords
        remaining_pieces = [coords for coords in cropped.keys() if coords not in config.values()]
        
        for next_piece in remaining_pieces:
            config_copy = deepcopy(config)
            
            assert next_location not in config_copy.keys(), 'issue when completing the current config'
            
            config_copy[next_location] = next_piece
            # print(f'config_copy = {config_copy}\n')
            
            if is_promising(config_copy, bestScore):
                print('Promising branch.\n')
                yield config_copy
            else:
                print('Not promising branch.\n')
                continue # get directly to next iteration
        
    
    def backtracking(config, cropped, bestScore, bestSol):
        '''
        Backtracking for building the puzzle (recursive).

        Args:
        - config: dictionary giving the mapping from the current
        configuration to a given configuration of the puzzle
        (the dictionary doesn't have to contain alle the puzzle pieces
        since it's being built on the moment)
        - cropped
        - bestScore (float)
        - bestSol (dict)

        Returns:
        - new_bestScore
        - new_bestSol
        '''
        
        
        if is_terminal(config):
            # print('Viewing current configuration:')
            # current_img = create_config(config, nb_lines, nb_cols)
            # plt.figure(figsize = (5,2))
            # plt.imshow(current_img)
            
            # pdb.set_trace()            
            
            print('is_terminal')
            current_score = score(config, cropped, nb_lines, nb_cols)
            
            # clear_output(wait=True)
            print(f'current_score: {current_score}\n')
            
            if current_score < bestScore:
                new_bestScore = current_score
                new_bestSol = deepcopy(config)
                
                print(f'New bestScore: {new_bestScore}\n')
                
        else:
            print(f'not terminal, current nb of pieces: {len(config)}')
            for new_config in children(config, cropped, bestScore):
                new_bestScore, new_bestSol = backtracking(new_config, cropped, bestScore, bestSol)
        
        return new_bestScore, new_bestSol
    
    
    # ------ Main ------
    return backtracking(config, cropped, bestScore, bestSol)
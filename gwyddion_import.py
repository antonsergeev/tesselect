# Script loads 3d data from text file (after Gwyddion text importing of AFM file)

import re
import numpy as np

def ReadData(file_name):
    '''
    Load 3d data array from a text file. The text file is imported from Gwyddion (free SPM data analysis software).
    
    Parameters
    ----------
    file_name : str
        Relative path to a text file
    
    Returns
    -------
    data : ndarray
        MxM matrix of SPM data
    width : float
        Width of image (in meters)
    height : float
        Height of image (in meters)
    pixel_height : float
        Height of one pixel (in meters)
    height_unit : float
        Measurement unit coefficient (in unit/meter)
    '''
    
    comments = []       # List of comments in text file
    f = open(file_name)
    for line in f:
        if line.startswith('#'):
            comments.append(line)
        else:
            break
    
    f.close()
    rex = r"(\d+[.]\d+)\s(\S+)"         # regular expression for image size searching
    width_match = re.search(rex, comments[1])
    height_match = re.search(rex, comments[2])
    
    if (width_match.group(2) == 'µm') and (height_match.group(2) == 'µm'):
        width_unit = 1e-6
        height_unit = 1e-6
    else:
        raise ValueError("Attention! The measurement units aren't micrometers!")   # My data was only in micrometers :)
    
    width = float(width_match.group(1)) * width_unit
    height = float(height_match.group(1)) * height_unit
    
    data = np.genfromtxt(file_name)     # NumPy function for data importing
    M = np.shape(data)[0]               # ---!!--- Needs to add rectangular area ---!!--- 
    pixel_height = height/M
    
    return data, width, height, pixel_height, height_unit

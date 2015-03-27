# -*- coding: UTF-8 -*-

"""
Tesselect module based on Delaunay tesselation and provides 
some grain analysis techniques for microscopy images.


Dependencies:
-------------
Python 3.x
NumPy
SciPy
Matplotlib
Shapely (optional - only for one function)


Available functions:
--------------------
- MinusND   - substract N-degree curve from image line-by-line.
...

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2014-2015, Anton Sergeev.
#
# Distributed under the terms of the MIT License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

# Release data
__version__ = '0.1'
__author__ = "Anton Sergeev <antonsergeevphd@gmail.com>"
__license__ = "MIT"

import copy

import numpy as np
import scipy as sp
from scipy import ndimage, spatial
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.widgets import Cursor, MultiCursor, RectangleSelector, Slider
from matplotlib.patches import Rectangle

from shapely.geometry import MultiPoint

########################################################################


def MinusND(input_data, degree=2, direction='h'):
    """
    Substract n-degree curve from image line-by-line.
    
    Parameters
    ----------
    input_data : ndarray
        Array to substract from
    degree : int, optional
        Degree of polynome to substract
        Default: 2
    direction : str, optional
        Direction of substraction lines
        Legal values: 'h' (horizontal, default), 
                      'v' (vertical), 
                      'hv' (horizontal, then vertical), 
                      'vh' (vertical, then horizontal)
    
    Returns
    -------
    data : ndarray
        Smoothened 3d array
    
    """
    if type(direction) != str:
        raise ValueError("Type of parameter 'direction' must be string ('h', 'v', 'hv' or 'vh')")
    direction = direction.lower()
    if direction not in ['h', 'v', 'hv', 'vh']:
        raise ValueError("Parameter 'direction' must be equal to 'h', 'v', 'hv' or 'vh'")
    
    M, N = np.shape(input_data)
    line_data = np.zeros((M, N))
    
    if direction[0] == 'h':
        for i in np.arange(M):
            line_func = np.poly1d(np.polyfit(np.arange(N), input_data[i, :], degree))
            line_data[i, :] = line_func(np.arange(N))
        if len(direction) == 2:    # for parameter 'hv'
            input_data = input_data - line_data
            for i in np.arange(N):
                line_func = np.poly1d(np.polyfit(np.arange(M), input_data[:, i], degree))
                line_data[:, i] = line_func(np.arange(M))
    elif direction[0] == 'v':
        for i in np.arange(N):
            line_func = np.poly1d(np.polyfit(np.arange(M), input_data[:, i], degree))
            line_data[:, i] = line_func(np.arange(M))
        if len(direction) == 2:    # for parameter 'vh'
            input_data = input_data - line_data
            for i in np.arange(M):
                line_func = np.poly1d(np.polyfit(np.arange(N), input_data[i, :], degree))
                line_data[i, :] = line_func(np.arange(N))
    
    data = input_data - line_data
    data -= np.amin(data)
    
    return data


########################################################################


def FindTriangulation(x, y, ratio=0.4):
    """
    Function
    """
    tri = mtri.Triangulation(x, y)
    mask = mtri.TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio=ratio)
    tri.set_mask(mask)
    filtered_tri = mtri.Triangulation(x, y, tri.get_masked_triangles())
    
    return filtered_tri


########################################################################


def TriEdges(tri):
    """
    Fuction
    """
    x = tri.x
    y = tri.y
    # sort point numbers of triangulation edges in ascending order:
    edges = np.sort(tri.edges[np.lexsort(tri.edges.T)])
    
    edge_lengths = np.zeros(len(edges))
    for k in range(len(edges)):
        edge = edges[k]
        edge_lengths[k] = np.hypot(x[edge[0]] - x[edge[1]], 
                                   y[edge[0]] - y[edge[1]])
    
    vertex_neighbours = [[] for k in range(len(x))]
    for edge in edges:
        vertex_neighbours[edge[0]].append(edge[1])
        vertex_neighbours[edge[1]].append(edge[0])
    
    coordination_numbers = np.asarray([len(k) for k in vertex_neighbours])
    
    return edges, edge_lengths, coordination_numbers, vertex_neighbours


########################################################################


def BoundaryVertices(coordination_numbers, vertex_neighbours):
    """
    Fuction
    """
    mask1_bool = (coordination_numbers <= 4)
    mask2_nonbool = [k for i in range(len(coordination_numbers)) if mask1_bool[i] for k in vertex_neighbours[i]]
    mask2_nonbool = np.array(list(set(mask2_nonbool)))
    mask2_bool = np.zeros_like(mask1_bool)
    for k in mask2_nonbool:
        mask2_bool[k] = True
    
    mask_boundary = (mask1_bool + mask2_bool)
    mask_center = ~mask_boundary
    
    return mask_center, mask_boundary


########################################################################


def FindAngles(x, y, edges, vertex_neighbours):
    """
    Fuction
    """
    edge_angles = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        edge_angles[i] = 180 / np.pi * np.arctan2(y[edge[1]] - y[edge[0]], x[edge[1]] - x[edge[0]])
    
    grain_angles_all = copy.deepcopy(vertex_neighbours)    # ---!!--- not returned ---!!---
    grain_angles_all_norm = copy.deepcopy(vertex_neighbours)    # ---!!--- not returned ---!!---
    grain_angles_mean = np.zeros(len(x))
    
    for i, neigs in enumerate(vertex_neighbours):
        for k, neig in enumerate(neigs):
            # first if
            if i < neig:
                grain_angles_all[i][k] = edge_angles[ np.all(edges==[i, neig], axis=1) ]
            elif edge_angles[ np.all(edges==[neig, i], axis=1) ] > 0:
                grain_angles_all[i][k] = edge_angles[ np.all(edges==[neig, i], axis=1) ] - 180
            elif edge_angles[ np.all(edges==[neig, i], axis=1) ] < 0:
                grain_angles_all[i][k] = edge_angles[ np.all(edges==[neig, i], axis=1) ] + 180
            else:
                grain_angles_all[i][k] = edge_angles[ np.all(edges==[neig, i], axis=1) ]
            # second if
            if grain_angles_all[i][k] > 150:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] - 180
            elif grain_angles_all[i][k] > 90:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] - 120
            elif grain_angles_all[i][k] > 30:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] - 60
            elif grain_angles_all[i][k] < -150:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] + 180
            elif grain_angles_all[i][k] < -90:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] + 120
            elif grain_angles_all[i][k] < -30:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k] + 60
            else:
                grain_angles_all_norm[i][k] = grain_angles_all[i][k]
        
        grain_angles_mean[i] = np.mean(grain_angles_all_norm[i])
    
    return edge_angles, grain_angles_mean


########################################################################


def FindMinsTri(data, label_data, x, y, edges):
    """
    Функция находит минимумы двумерного массива "data" в точках с координатами (x,y), соединенных ребрами "edges"
    Возвращает локальный минимум для каждого ребра в "edges"
    
    """
    label_data = label_data.copy() - 1    # совпадение номеров с точками
    
    min_list = [[] for k in range(len(x))]
    argmin_array = np.zeros(len(edges))
    argmin_xy = np.zeros((2, len(edges)))
    
    for k, edge in enumerate(edges):
        x0, y0 = x[edge[0]], y[edge[0]]
        x1, y1 = x[edge[1]], y[edge[1]]
        
        num = max(abs(x1-x0), abs(y1-y0)) + 1    # максимальное количество пикселей (выбор между x и y) между двумя точками
        X, Y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        X, Y = np.rint(X).astype(int), np.rint(Y).astype(int)
        data_cs = data[Y, X]    # важно поменять местами X и Y
        
        label_data_cs = label_data[Y, X]
        if np.any(label_data_cs == edge[0]):
            cs_start = np.where(label_data_cs == edge[0])[0][-1]
        else:
            cs_start = 0
        if np.any(label_data_cs == edge[1]):
            cs_end = np.where(label_data_cs == edge[1])[0][0]
        else:
            cs_end = len(label_data_cs) - 1
        
        local_min = np.amin(data_cs[cs_start:cs_end+1])
        min_list[edge[0]].append(local_min)
        min_list[edge[1]].append(local_min)
        
        local_argmin = np.argmin(data_cs[cs_start:cs_end+1]) + cs_start
        argmin_array[k] = local_argmin
        argmin_xy[0, k] = X[local_argmin]
        argmin_xy[1, k] = Y[local_argmin]
        
    return min_list, argmin_array, argmin_xy


########################################################################


def FindOutEdges(edges, num_of_points=-1):
    """
    Fuction
    """
    if type(num_of_points) != int:
        raise ValueError("Type of parameter 'num_of_points' must be integer")
    if num_of_points == -1:
        num_of_points = max([max(i) for i in edges]) + 1
    elif num_of_points < 2:
        raise ValueError("Value of parameter 'num_of_points' must be greater than 1")
    
    out_edges = [[] for k in range(num_of_points)]
    for k, edge in enumerate(edges):
        out_edges[edge[0]].append(k)
        out_edges[edge[1]].append(k)
    
    return out_edges


########################################################################


def PreciseLabels(data_shape, argmin_xy, out_edges, mask_center):
    """
    Fuction
    """
    mesh_x, mesh_y = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
    coords = np.vstack((mesh_x.ravel(), mesh_y.ravel())).T
    coords = MultiPoint(coords)
    label_data_prec = np.zeros(data_shape, dtype=int)
    
    num = np.sum(mask_center)  # number of precise labels
    percentage = np.rint(np.linspace(0,num,21)).astype(int)
    count = 0  # number of calculated labels
    print('Calculated: ', end='')
    
    for i, outs in enumerate(out_edges):
        if mask_center[i] == True:
            poly = MultiPoint(argmin_xy.T[outs]).convex_hull
            inpoints = [point for point in coords if poly.contains(point)]
            for point in inpoints:
                label_data_prec[point.y, point.x] = i + 1
            
            if count in percentage:
                print('{}%... '.format(np.argwhere(percentage==count)[0,0]*5), end='')
            elif count == num - 1:
                print('100%')
            count += 1
    
    return label_data_prec


########################################################################


def FindDiameters(input_data, label_data, half_height, data_step):
    """
    Fuction
    """
    labels = np.unique(label_data)[1:]
    grain_area = np.zeros(len(half_height))
    grain_area_half = np.zeros(len(half_height))
    
    for k in labels:
        grain = input_data * (label_data == k)
        grain_area[k-1] = np.sum(grain > 0) * data_step**2
        grain_area_half[k-1] = np.sum(grain > half_height[k]) * data_step**2
    
    return grain_area, grain_area_half


########################################################################


def RDF(x, y, data_step, area_total):
    """
    Fuction
    """
    grain_pdist = spatial.distance.pdist(np.vstack((x, y)).T)
    rdf = np.zeros(np.ceil(np.max(grain_pdist)).astype(int), dtype=int)
    for dis in np.floor(grain_pdist):
        rdf[dis] += 1
    
    area_r = np.zeros(len(rdf))
    for k in range(len(rdf)):
        area_r[k] = np.pi*((k+1)**2 - k**2)
    area_r *= data_step**2
    
    normarea = area_r / area_total
    normrdf = rdf / np.sum(rdf)
    rdf_y = normrdf / normarea
    rdf_x = np.arange(len(rdf))*data_step
    
    return rdf_x, rdf_y


########################################################################





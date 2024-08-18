import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2 as cv
from matplotlib.colors import hex2color

import warnings
from matplotlib.collections import LineCollection


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return ax.add_collection(lc)

def function_timer(num_runs=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            for i in range(num_runs):
                start = time.time()
                func(*args, **kwargs)
                end = time.time()
                times.append(end-start)
            return times
        return wrapper
    return decorator

def nparray_to_video(fname, data, clim='auto', cmap='gray', codec='h264', fps=24, ffmpeg_params=['-pix_fmt', 'yuv420p']):
    """
    Save a 3D (t, y, x) numpy array to disk as movie after applying a colormap
    
    Warning : this function duplicates the input data in memory
    
    """
    
    def apply_cmap(data, cmap='gray', clim='auto'):

        if clim == 'auto':
            clim = data.min(), data.max()

        sm = ScalarMappable(Normalize(*clim, clip=True), cmap)    
        rgba = [sm.to_rgba(d, bytes=True) for d in data]

        return rgba
    
    dur = data.shape[0] / fps
    
    data_rgba = apply_cmap(data, cmap=cmap, clim=clim)
    clip = ImageSequenceClip([d for d in data_rgba], fps=fps)
    clip.set_duration(dur).write_videofile(fname, audio=False, codec=codec, fps=fps, ffmpeg_params=ffmpeg_params)

def find_contour(img, contour_color: str = '#000000', background_color: str = '#FFFFFF'):
    N = img.shape[1]
    background_rgb = hex2color(background_color)
    contour_rgb = tuple(int(255*color) for color in hex2color(contour_color))
    c, hierarchy = cv.findContours((img>0.5).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cv.drawContours((np.ones((N,N,3))*background_rgb*255).astype(np.uint8), c, -1, contour_rgb, 2)

def nparray_to_contour_video(fname, data, clim='auto', cmap='gray', codec='h264', fps=24, ffmpeg_params=['-pix_fmt', 'yuv420p'], contour_color: str = '#000000', background_color: str = '#FFFFFF'):
    """
    Save a 3D (t, y, x) numpy array to disk as movie after applying a colormap
    
    """
    dur = data.shape[0] / fps
    ImageList = [find_contour(data[i], contour_color, background_color) for i in range(data.shape[0])]
    
    clip = ImageSequenceClip(ImageList, fps=fps)
    clip.set_duration(dur).write_videofile(fname, audio=False, codec=codec, fps=fps, ffmpeg_params=ffmpeg_params)
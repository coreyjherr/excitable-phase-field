import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2 as cv
from matplotlib.colors import hex2color

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
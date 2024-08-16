#!/home/cherr/workspace/stochastic_phase_field_model/.venv/bin/python3
#%%
from util import nparray_to_video
import numpy as np

if __name__ == '__main__':
    # read the numpy array from the binary file
    filename_A = 'A.dat'
    filename_phi = 'phi.dat'

    A = np.fromfile(filename_A, dtype=np.float32).reshape(-1, 256, 256)
    phi = np.fromfile(filename_phi, dtype=np.float32).reshape(-1, 256, 256)
    #%%
    nparray_to_video('output.mp4', A*phi, fps=120)

    
# %%

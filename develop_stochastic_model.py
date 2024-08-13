# This file is used to test an implementation of a stochastic model found in the
# paper "From single to Collective Motion: From Single to Collective Motion of
# Social Amoebae: A Computational Study of Interacting Cells" by Alonso et al

# %%
import cupy as cp
from cupyx.scipy.signal import convolve2d
from matplotlib.pylab import f
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
from util import nparray_to_video, nparray_to_contour_video

from dataclasses import dataclass

@dataclass
class ExcitableCellParams:
    M: float = 0.045
    k_n: float = 0.1
    alpha: int = 3
    sigma: float = 0.15
    c0: int = 28
    dx: float = 0.15
    dt: float = 0.002
    N: int = 256
    beta: float = 22.22
    A0: float = 113
    gamma: float = 2
    eps: float = 0.75
    D: float = 0.5
    k_a: float = 1.5
    rho: float = 0.02
    delta0: float = 0.5
    tau: float = 2

import cupy as cp
from cupyx.scipy.signal import convolve2d

class ExcitableCell2d:
    def __init__(self, params: ExcitableCellParams):
        self.pp = params
        self.k = 2 * cp.pi * cp.fft.fftfreq(self.pp.N, d=self.pp.dx)
        self.k = self.k[:, cp.newaxis]
        self.k2 = self.k**2 + self.k.T**2
        self.phi = cp.zeros((self.pp.N, self.pp.N))
        self.c = cp.zeros((self.pp.N, self.pp.N))
        x = cp.linspace(-self.pp.N // 2, self.pp.N // 2, self.pp.N) * self.pp.dx
        X, Y = cp.meshgrid(x, x)

        self.phi[X**2 + Y**2 < 6**2] = 1
        self.c[X**2 + Y**2 < 6**2] = self.pp.c0 / self.pp.A0
        self.noise = cp.random.normal(size=(self.pp.N, self.pp.N), scale=self.pp.sigma * cp.sqrt(self.pp.dt))

        self.grid_offset_x = 0
        self.grid_offset_y = 0

    def get_center_of_mass_index(self):
        """ Gets the unitless center of mass of the cell"""
        i_com = cp.sum(self.phi * cp.arange(self.pp.N)) / (self.get_area(self.phi) / self.pp.dx**2)
        j_com = cp.sum(self.phi * cp.arange(self.pp.N).reshape(-1, 1)) / (self.get_area(self.phi) / self.pp.dx**2)
        return i_com, j_com

    def get_distance_to_boundary(self):
        """gets the distance of each point to the nearest boundary"""
        i, j = cp.where(self.phi > 0.5)
        i_min = cp.min(cp.minimum(i, self.pp.N - i))
        j_min = cp.min(cp.minimum(j, self.pp.N - j))
        return i_min, j_min

    def move_phase_fields(self, field,  axis, shift) -> None:
        """ Roll the phase fields in the direction specified """
        if axis == 0 and shift > 0:
            field[:self.pp.N-shift, :] = field[shift:, :]
            field[self.pp.N-shift:, :] = 0
        elif axis == 0 and shift < 0:
            field[-shift:, :] = field[:self.pp.N+shift, :]
            field[:-shift, :] = 0
        elif axis == 1 and shift > 0:
            field[:, :self.pp.N-shift] = field[:, shift:]
            field[:, self.pp.N-shift:] = 0
        elif axis == 1 and shift < 0:
            field[:, -shift:] = field[:, :self.pp.N+shift]
            field[:, :-shift] = 0

    def update_grid_location(self, i_cutoff, j_cutoff):
        i_com, j_com = self.get_distance_to_boundary()
        shift = self.pp.N//2 - i_cutoff
        if i_com <= i_cutoff:
            self.grid_offset_x -= shift
            print(shift)
            self.move_phase_fields(self.phi, 1, -shift)
            self.move_phase_fields(self.c, 1, -shift)
            self.move_phase_fields(self.noise, 1, -shift)
        elif i_com >= self.pp.N-i_cutoff:
            print(shift)
            self.grid_offset_x += shift
            self.move_phase_fields(self.phi, 1, shift)
            self.move_phase_fields(self.c, 1, shift)
            self.move_phase_fields(self.noise, 1, shift)

        shift = self.pp.N//2 - j_cutoff
        if j_com <= j_cutoff:
            print(shift)
            self.grid_offset_y -= shift
            self.move_phase_fields(self.phi, 0, -shift)
            self.move_phase_fields(self.c, 0, -shift)
            self.move_phase_fields(self.noise, 0, -shift)
        elif j_com >= self.pp.N-j_cutoff:
            print(shift)
            self.grid_offset_y += shift
            self.move_phase_fields(self.phi, 0, shift)
            self.move_phase_fields(self.c, 0, shift)
            self.move_phase_fields(self.noise, 0, shift)

    def get_area(self, field):
        return cp.sum(field) * self.pp.dx**2

    def laplacian(self, f):
        return (
            convolve2d(
                f,
                cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                mode="same",
                boundary="wrap",
            )
            / self.pp.dx**2
        )

    def diffx(self, field):
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=0)[1:-1, 1:-1]

    def diffy(self, field):
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=1)[1:-1, 1:-1]

    def gprime(self):
        # G = 18*phi**2*(1-phi)**2
        return 36 * self.phi * (1 - self.phi) ** 2 - 36 * self.phi**2 * (1 - self.phi)

    def update_phi(self, phi_x, phi_y):
        d_phi = (
            self.pp.gamma*self.laplacian(self.phi)
            - self.pp.gamma / self.pp.eps**2 * self.gprime()
            + (-self.pp.beta * (self.get_area(self.phi) - self.pp.A0) + self.pp.alpha * self.c * self.phi)
            * cp.sqrt((phi_x**2 + phi_y**2))
        )
        if cp.any(cp.isnan(d_phi)):
            print("nan")
        self.phi += self.pp.dt / self.pp.tau * d_phi

    def update_phi_spectral(self, phi_x, phi_y):
        d_phi = (
            - self.pp.gamma / self.pp.eps**2 * self.gprime()
            + (-self.pp.beta * (self.get_area(self.phi) - self.pp.A0) + self.pp.alpha * self.c * self.phi)
            * cp.sqrt((phi_x**2 + phi_y**2))
        )
        phi_hat = cp.fft.fft2(self.phi + self.pp.dt / self.pp.tau * d_phi)
        self.phi = cp.fft.ifft2(phi_hat / (1 + self.pp.gamma * self.k2 * self.pp.dt)).real

    def update_c(self, phi_x, phi_y):
        c_x = self.diffx(self.c)
        c_y = self.diffy(self.c)
        d_c = (
            self.pp.D * (self.laplacian(self.c) + cp.where(self.phi > 0.01, (phi_x * c_x + phi_y * c_y) / self.phi, 0))
            + self.pp.k_a * self.c * (1 - self.c) * (self.c - (self.pp.delta0 + self.pp.M * (self.get_area(self.c) - self.pp.c0)))
            - self.pp.rho * self.c
        )
        self.c += self.pp.dt * d_c + cp.sqrt(self.pp.dt) * self.phi*(1-self.phi) * self.noise

    def update_c_spectral(self, phi_x, phi_y):
        c_x = self.diffx(self.c)
        c_y = self.diffy(self.c)
        d_c = (
            self.pp.D * (cp.where(self.phi > 0.01, (phi_x * c_x + phi_y * c_y) / self.phi, 0))
            + self.pp.k_a * self.c * (1 - self.c) * (self.c - (self.pp.delta0 + self.pp.M * (self.get_area(self.c) - self.pp.c0)))
        )
        c_hat = cp.fft.fft2(self.c + self.pp.dt * d_c + cp.sqrt(self.pp.dt) * self.phi * (1 - self.phi) * self.noise)
        self.c = cp.fft.ifft2(c_hat / (1 + self.pp.D * self.k2 * self.pp.dt + self.pp.rho * self.pp.dt)).real

    def update_noise(self):
        self.noise = (
            self.noise
            - self.pp.dt * self.pp.k_n * self.noise
            + self.phi*cp.sqrt(self.pp.dt * self.pp.sigma) * cp.random.normal(size=(self.pp.N, self.pp.N))
        )

def run_trial(params: ExcitableCellParams, blind: int, simulation_runtime: int):
    excitable_cell = ExcitableCell2d(params)
    T = int(simulation_runtime // params.dt)
    phi_save = np.zeros((T // blind + 1, params.N, params.N))
    c_save = np.zeros((T // blind + 1, params.N, params.N))
    for i in trange(T):
        phi_x = excitable_cell.diffx(excitable_cell.phi)
        phi_y = excitable_cell.diffy(excitable_cell.phi)
        excitable_cell.update_phi(phi_x, phi_y)
        excitable_cell.update_c(phi_x, phi_y)
        excitable_cell.update_noise()
        if i % blind == 0:
            phi_save[i // blind] = excitable_cell.phi.get()
            c_save[i // blind] = excitable_cell.c.get()
            # excitable_cell.update_grid_location(params.N // 6, params.N // 6)
    return phi_save, c_save

for k_a in range(1, 6):
    cp.random.seed(0)
    params = ExcitableCellParams(k_a = k_a)
    blind = 100
    simulation_runtime = 50

    phi_save, c_save = run_trial(params, blind, simulation_runtime)


    plt.figure()
    plt.imshow(c_save[-1])
    plt.colorbar()
    print(c_save[-1].sum())

    plt.figure()
    plt.imshow(phi_save[-1])
    plt.colorbar()

    plt.figure()
    plt.plot(np.arange(0, simulation_runtime, params.dt*blind), params.dx**2 * c_save.sum(axis=(1, 2)))

    fps = 120
    fmt_str = f"k_a_{k_a}"
    nparray_to_video(f"../movies/phase_field{fmt_str}_tmp.mp4", phi_save, fps=fps)
    nparray_to_video(f"../movies/concentration_field{fmt_str}_tmp.mp4", c_save * phi_save, fps=fps)

    nparray_to_contour_video(f"../movies/phase_contour{fmt_str}.mp4", phi_save, background_color="#f2f2f2", fps=fps)

# %%


#%%
from dataclasses import dataclass
from util import nparray_to_video
from pathlib import Path

@dataclass
class ExcitableCellParams:
    gamma: float = 2
    eps: float = 2
    B_s: float = 10
    xi: float = 10
    alpha: float = 4
    k_a: float = 10
    K: float = 1
    b: float = 0.1
    A_t: float = 2
    d1: float = 1
    d2: float = 1
    c1: float = 1
    c2: float = 30
    tau: float = 10
    D_A: float = 0.5
    D_R: float = 0.5
    sigma: float = 0.1
    dt: float = 0.001
    dx: float = 0.2
    A0: float = 0.5
    N: int = 256
    r0: float = 6
    m_repulsion: float = 10
    initial_offset_x: int = 0
    initial_offset_y: int = 0


import cupy as cp
from cupyx.scipy.ndimage import convolve
from cupyx.scipy.signal import convolve2d
from cupyx.scipy.ndimage import laplace

# find the root of an equation using scipy
from scipy.optimize import fsolve

class MultiExcitableCell2d:
    def __init__(self, params: ExcitableCellParams, num_cells: int = 1):
        self.pp = params
        self.num_cells = num_cells

        self.phi = cp.zeros((num_cells, self.pp.N, self.pp.N))
        self.A = cp.zeros((num_cells, self.pp.N, self.pp.N))
        self.R = cp.zeros((num_cells, self.pp.N, self.pp.N))

        x = cp.linspace(-self.pp.N // 2, self.pp.N // 2, self.pp.N) * self.pp.dx
        X, Y = cp.meshgrid(x, x)
        self.laplacian_kernel = cp.tile(cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (num_cells, 1, 1))

        self.phi = cp.tile(
            1 / 2 * (1 - cp.tanh(3 * (cp.sqrt(X**2 + Y**2) - self.pp.r0) / self.pp.eps)),
            (num_cells, 1, 1)
        )
        self.S0 = cp.sum(self.phi[0]) * self.pp.dx**2

        A_equil, R_equil = self.initialize_fields()
        self.A[self.phi > 0.0025] = A_equil
        self.R[self.phi > 0.0025] = R_equil

        # self.grid_offset_x = params.initial_offset_x*np.arange(num_cells)
        # self.grid_offset_y = params.initial_offset_y*np.ones(num_cells)
        self.grid_offset_x = cp.arange(num_cells)*params.initial_offset_x
        self.grid_offset_y = cp.ones(num_cells)*params.initial_offset_y

        self.generator_R = cp.random.default_rng(seed=0)
        self.generator_A = cp.random.default_rng(seed=1)

        self.collision_force = cp.zeros((num_cells, self.pp.N, self.pp.N))

        self.phi_x = self.diffx(self.phi)
        self.phi_y = self.diffy(self.phi)
        self.phi_prev = self.phi
        self.A_prev = self.A

    def initialize_fields(self):
        A_nullcline = (
            lambda A: (self.pp.k_a * A**2 / (self.pp.K**2 + A**2) + self.pp.b)
            * (self.pp.A_t - A)
            - (self.pp.d1 + self.pp.d2 * (self.pp.c2 / (self.pp.c1 * self.pp.tau)) * A)
            * A
        )
        A_equil = fsolve(A_nullcline, 0)[0]
        R_equil = self.pp.c2 * A_equil / (self.pp.c1 * self.pp.tau)
        return A_equil, R_equil

    def get_center_of_mass_index(self):
        """Gets the unitless center of mass of the cell"""
        i_com = cp.sum(self.phi * cp.arange(self.pp.N)) / (
            self.get_area(self.phi) / self.pp.dx**2
        )
        j_com = cp.sum(self.phi * cp.arange(self.pp.N).reshape(-1, 1)) / (
            self.get_area(self.phi) / self.pp.dx**2
        )
        return i_com, j_com

    def get_distance_to_boundary(self, cell: int):
        """gets the distance of each point to the nearest boundary"""
        i, j = cp.where(self.phi[cell] > 0.5)
        # find the closest point to the boundary
        i_min = cp.min(i)
        i_max = cp.max(i)
        j_min = cp.min(j)
        j_max = cp.max(j)
        
        i_boundary = i_min if i_min < self.pp.N - i_max else i_max
        j_boundary = j_min if j_min < self.pp.N - j_max else j_max

        return i_boundary, j_boundary

    def move_phase_fields(self, field, axis, shift, cell) -> None:
        """Roll the phase fields in the direction specified"""
        if axis == 0 and shift > 0:
            field[cell, : self.pp.N - shift, :] = field[cell, shift:, :]
            field[cell, self.pp.N - shift :, :] = 0
        elif axis == 0 and shift < 0:
            field[cell, -shift:, :] = field[cell, : self.pp.N + shift, :]
            field[cell, :-shift, :] = 0
        elif axis == 1 and shift > 0:
            field[cell, :, : self.pp.N - shift] = field[cell, :, shift:]
            field[cell, :, self.pp.N - shift :] = 0
        elif axis == 1 and shift < 0:
            field[cell, :, -shift:] = field[cell, :, : self.pp.N + shift]
            field[cell, :, :-shift] = 0

    def update_grid_location(self, i_cutoff, j_cutoff, cell_index: int):
        i_com, j_com = self.get_distance_to_boundary(cell_index)
        shift = self.pp.N // 2 - i_cutoff
        if i_com <= i_cutoff:
            print(i_com, j_com)
            self.grid_offset_x[cell_index] -= shift
            self.move_phase_fields(self.phi, 0, -shift, cell_index)
            self.move_phase_fields(self.A, 0, -shift, cell_index)
            self.move_phase_fields(self.R, 0, -shift, cell_index)
        elif i_com >= self.pp.N - i_cutoff:
            print(i_com, j_com)
            self.grid_offset_x[cell_index] += shift
            self.move_phase_fields(self.phi, 0, shift, cell_index)
            self.move_phase_fields(self.A, 0, shift, cell_index)
            self.move_phase_fields(self.R, 0, shift, cell_index)

        shift = self.pp.N // 2 - j_cutoff
        if j_com <= j_cutoff:
            print(i_com, j_com)
            self.grid_offset_y[cell_index] -= shift
            self.move_phase_fields(self.phi, 1, -shift, cell_index)
            self.move_phase_fields(self.A, 1, -shift, cell_index)
            self.move_phase_fields(self.R, 1, -shift, cell_index)
        elif j_com >= self.pp.N - j_cutoff:
            print(i_com, j_com)
            self.grid_offset_y[cell_index] += shift
            self.move_phase_fields(self.phi, 1, shift, cell_index)
            self.move_phase_fields(self.A, 1, shift, cell_index)
            self.move_phase_fields(self.R, 1, shift, cell_index)

    def update_all_grid_locations(self, i_cutoff, j_cutoff):
        for i in range(self.num_cells):
            self.update_grid_location(i_cutoff, j_cutoff, i)

    def get_area(self, field):
        return cp.sum(field, axis=(1,2)) * self.pp.dx**2

        
    def laplacian(self, f):
        return laplace(f, mode='wrap')/self.pp.dx**2


    def diffx(self, field):
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=1)[
            1:-1, 1:-1, 1:-1
        ]

    def diffy(self, field):
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=2)[
            1:-1, 1:-1, 1:-1
        ]

    def gprime(self):
        # G = 18*phi**2*(1-phi)**2
        return 36 * self.phi * (1 - self.phi) ** 2 - 36 * self.phi**2 * (1 - self.phi)

    def update_phi(self):
        d_phi = (
            self.pp.gamma * self.laplacian(self.phi)
            - self.pp.gamma / self.pp.eps**2 * self.gprime()
            + (
                -self.pp.B_s * (self.get_area(self.phi).reshape(-1,1,1) - self.S0)
                + self.pp.alpha * self.A * self.phi
                - self.pp.m_repulsion * self.collision_force
            )
            * cp.sqrt((self.phi_x**2 + self.phi_y**2))
        )
        self.phi += self.pp.dt / self.pp.xi * d_phi

    def update_A(self, ):
        """Need to update phi before updating A"""
        A_x = self.diffx(self.A)
        A_y = self.diffy(self.A)
        d_A = cp.where(
            self.phi > 0.0025,
            (
                self.A
                + self.pp.dt
                * (
                    self.pp.D_A
                    * (self.phi * self.laplacian(self.A) + (self.phi_x * A_x + self.phi_y * A_y))
                    + self.phi
                    * (
                        (self.pp.A_t - self.A)
                        * (
                            self.pp.k_a * self.A**2 / (self.pp.K**2 + self.A**2)
                            + self.pp.b
                        )
                        - (self.pp.d1 + self.pp.d2 * self.R) * self.A
                        + cp.sqrt(self.pp.sigma / self.pp.dt)
                        * self.generator_A.standard_normal(
                            size=self.A.shape,
                        )
                    )
                    - self.A * (self.phi - self.phi_prev) / self.pp.dt
                )
                / self.phi
            ),
            0,
        )
        self.A = d_A

    def update_R(self):
        R_x = self.diffx(self.R)
        R_y = self.diffy(self.R)
        d_R = cp.where(
            self.phi > 0.0025,
            (
                self.R
                + self.pp.dt
                * (
                    self.pp.D_R
                    * (self.phi * self.laplacian(self.R) + (self.phi_x * R_x + self.phi_y * R_y))
                    + self.phi
                    * (
                        (self.pp.c2 * self.A - self.pp.c1 * self.R) / self.pp.tau
                        + cp.sqrt(self.pp.sigma / self.pp.dt)
                        * self.generator_R.standard_normal(
                            size=self.R.shape,
                        )
                    )
                    - self.R * (self.phi - self.phi_prev) / self.pp.dt
                )
                / self.phi
            ),
            0,
        )
        self.R = d_R

    def update_collision_force(self):
        self.collision_force = cp.zeros((self.num_cells, self.pp.N, self.pp.N))
        for i in range(self.num_cells):
            for j in range(i+1, self.num_cells):

                i_rel = self.grid_offset_x[j] - self.grid_offset_x[i]
                j_rel = self.grid_offset_y[j] - self.grid_offset_y[i]

                if abs( i_rel ) > self.pp.N or abs(j_rel) > self.pp.N:
                    continue

                s_y1 = cp.s_[i_rel:] if i_rel >= 0 else cp.s_[:i_rel]
                s_x1 = cp.s_[j_rel:] if j_rel >= 0 else cp.s_[:j_rel]
                s_y2 = cp.s_[:-i_rel] if i_rel > 0 else cp.s_[-i_rel:]
                s_x2 = cp.s_[:-j_rel] if j_rel > 0 else cp.s_[-j_rel:]

                self.collision_force[i, s_x1, s_y1] += self.phi[i, s_x1, s_y1]*self.phi[j, s_x2, s_y2]
                self.collision_force[j, s_x2, s_y2] += self.phi[i, s_x1, s_y1]*self.phi[j, s_x2, s_y2]






class ExcitableCell2d:
    def __init__(self, params: ExcitableCellParams):
        self.pp = params

        self.phi = cp.zeros((self.pp.N, self.pp.N))
        self.A = cp.zeros((self.pp.N, self.pp.N))
        self.R = cp.zeros((self.pp.N, self.pp.N))

        x = cp.linspace(-self.pp.N // 2, self.pp.N // 2, self.pp.N) * self.pp.dx
        X, Y = cp.meshgrid(x, x)

        self.phi = (
            1 / 2 * (1 - cp.tanh(3 * (cp.sqrt(X**2 + Y**2) - self.pp.r0) / self.pp.eps))
        )
        self.S0 = cp.sum(self.phi) * self.pp.dx**2

        A_equil, R_equil = self.initialize_fields()
        self.A[self.phi > 0.0025] = A_equil
        self.R[self.phi > 0.0025] = R_equil

        self.grid_offset_x = params.initial_offset_x
        self.grid_offset_y = params.initial_offset_y

        self.generator_R = cp.random.default_rng(seed=0)
        self.generator_A = cp.random.default_rng(seed=1)

        self.collision_force = cp.zeros((self.pp.N, self.pp.N))

    def initialize_fields(self):
        A_nullcline = (
            lambda A: (self.pp.k_a * A**2 / (self.pp.K**2 + A**2) + self.pp.b)
            * (self.pp.A_t - A)
            - (self.pp.d1 + self.pp.d2 * (self.pp.c2 / (self.pp.c1 * self.pp.tau)) * A)
            * A
        )
        A_equil = fsolve(A_nullcline, 0)[0]
        R_equil = self.pp.c2 * A_equil / (self.pp.c1 * self.pp.tau)
        return A_equil, R_equil

    def get_center_of_mass_index(self):
        """Gets the unitless center of mass of the cell"""
        i_com = cp.sum(self.phi * cp.arange(self.pp.N)) / (
            self.get_area(self.phi) / self.pp.dx**2
        )
        j_com = cp.sum(self.phi * cp.arange(self.pp.N).reshape(-1, 1)) / (
            self.get_area(self.phi) / self.pp.dx**2
        )
        return i_com, j_com

    def get_distance_to_boundary(self):
        """gets the distance of each point to the nearest boundary"""
        i, j = cp.where(self.phi > 0.5)
        # find the closest point to the boundary
        i_min = cp.min(i)
        i_max = cp.max(i)
        j_min = cp.min(j)
        j_max = cp.max(j)
        
        i_boundary = i_min if i_min < self.pp.N - i_max else i_max
        j_boundary = j_min if j_min < self.pp.N - j_max else j_max

        return i_boundary, j_boundary

    def move_phase_fields(self, field, axis, shift) -> None:
        """Roll the phase fields in the direction specified"""
        if axis == 0 and shift > 0:
            field[: self.pp.N - shift, :] = field[shift:, :]
            field[self.pp.N - shift :, :] = 0
        elif axis == 0 and shift < 0:
            field[-shift:, :] = field[: self.pp.N + shift, :]
            field[:-shift, :] = 0
        elif axis == 1 and shift > 0:
            field[:, : self.pp.N - shift] = field[:, shift:]
            field[:, self.pp.N - shift :] = 0
        elif axis == 1 and shift < 0:
            field[:, -shift:] = field[:, : self.pp.N + shift]
            field[:, :-shift] = 0

    def update_grid_location(self, i_cutoff, j_cutoff):
        i_com, j_com = self.get_distance_to_boundary()
        shift = self.pp.N // 2 - i_cutoff
        if i_com <= i_cutoff:
            print(i_com, j_com)
            self.grid_offset_x -= shift
            self.move_phase_fields(self.phi, 0, -shift)
            self.move_phase_fields(self.A, 0, -shift)
            self.move_phase_fields(self.R, 0, -shift)
        elif i_com >= self.pp.N - i_cutoff:
            print(i_com, j_com)
            self.grid_offset_x += shift
            self.move_phase_fields(self.phi, 0, shift)
            self.move_phase_fields(self.A, 0, shift)
            self.move_phase_fields(self.R, 0, shift)

        shift = self.pp.N // 2 - j_cutoff
        if j_com <= j_cutoff:
            print(i_com, j_com)
            self.grid_offset_y -= shift
            self.move_phase_fields(self.phi, 1, -shift)
            self.move_phase_fields(self.A, 1, -shift)
            self.move_phase_fields(self.R, 1, -shift)
        elif j_com >= self.pp.N - j_cutoff:
            print(i_com, j_com)
            self.grid_offset_y += shift
            self.move_phase_fields(self.phi, 1, shift)
            self.move_phase_fields(self.A, 1, shift)
            self.move_phase_fields(self.R, 1, shift)

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
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=0)[
            1:-1, 1:-1
        ]

    def diffy(self, field):
        return cp.gradient(cp.pad(field, 1, mode="wrap"), self.pp.dx, axis=1)[
            1:-1, 1:-1
        ]

    def gprime(self):
        # G = 18*phi**2*(1-phi)**2
        return 36 * self.phi * (1 - self.phi) ** 2 - 36 * self.phi**2 * (1 - self.phi)

    def update_phi(self, phi_x, phi_y):
        d_phi = (
            self.pp.gamma * self.laplacian(self.phi)
            - self.pp.gamma / self.pp.eps**2 * self.gprime()
            + (
                -self.pp.B_s * (self.get_area(self.phi) - self.S0)
                + self.pp.alpha * self.A * self.phi
                - self.pp.m_repulsion * self.collision_force
            )
            * cp.sqrt((phi_x**2 + phi_y**2))
        )
        self.phi += self.pp.dt / self.pp.xi * d_phi

    def update_A(self):
        """Need to update phi before updating A"""
        A_x = self.diffx(self.A)
        A_y = self.diffy(self.A)
        d_A = cp.where(
            self.phi > 0.0025,
            (
                self.A
                + self.pp.dt
                * (
                    self.pp.D_A
                    * (self.phi * self.laplacian(self.A) + (self.phi_x * A_x + self.phi_y * A_y))
                    + self.phi
                    * (
                        (self.pp.A_t - self.A)
                        * (
                            self.pp.k_a * self.A**2 / (self.pp.K**2 + self.A**2)
                            + self.pp.b
                        )
                        - (self.pp.d1 + self.pp.d2 * self.R) * self.A
                        + cp.sqrt(self.pp.sigma / self.pp.dt)
                        * self.generator_A.standard_normal(
                            size=(self.pp.N, self.pp.N),
                        )
                    )
                    - self.A * (self.phi -self.phi_prev) / self.pp.dt
                )
                / self.phi
            ),
            0,
        )
        self.A = d_A

    def update_R(self):
        R_x = self.diffx(self.R)
        R_y = self.diffy(self.R)
        d_R = cp.where(
            self.phi > 0.0025,
            (
                self.R
                + self.pp.dt
                * (
                    self.pp.D_R
                    * (self.phi * self.laplacian(self.R) + (self.phi_x * R_x + self.phi_y * R_y))
                    + self.phi
                    * (
                        (self.pp.c2 * self.A - self.pp.c1 * self.R) / self.pp.tau
                        + cp.sqrt(self.pp.sigma / self.pp.dt)
                        * self.generator_R.standard_normal(
                            size=(self.pp.N, self.pp.N),
                        )
                    )
                    - self.R * (self.phi - self.phi_prev) / self.pp.dt
                )
                / self.phi
            ),
            0,
        )
        self.R = d_R

def update_collision_force(cell_list: list[ExcitableCell2d], N: int):
    for cell in cell_list:
        cell.collision_force = cp.zeros((N, N))
    for i, cell in enumerate(cell_list):
        for j, other_cell in enumerate(cell_list[i+1:]):
            if i == j:
                continue
            i_rel = other_cell.grid_offset_x -cell.grid_offset_x 
            j_rel = other_cell.grid_offset_y - cell.grid_offset_y 
            if abs( i_rel ) > N or abs(j_rel) > N:
                continue

            s_y1 = cp.s_[i_rel:] if i_rel >= 0 else cp.s_[:i_rel]
            s_x1 = cp.s_[j_rel:] if j_rel >= 0 else cp.s_[:j_rel]
            s_y2 = cp.s_[:-i_rel] if i_rel > 0 else cp.s_[-i_rel:]
            s_x2 = cp.s_[:-j_rel] if j_rel > 0 else cp.s_[-j_rel:]

            cell.collision_force[s_x1, s_y1] += other_cell.phi[s_x2, s_y2]*cell.phi[s_x1, s_y1]
            other_cell.collision_force[s_x2, s_y2] += cell.phi[s_x1, s_y1]*other_cell.phi[s_x2, s_y2]


import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def run_trial(params: ExcitableCellParams, blind: int, simulation_runtime: int, gpu_id: int=0):
    with cp.cuda.Device(gpu_id):
        excitable_cell = ExcitableCell2d(params)
        T = int(simulation_runtime // params.dt)
        phi_save = np.zeros((T // blind + 1, params.N, params.N))
        A_save = np.zeros((T // blind + 1, params.N, params.N))
        R_save = np.zeros((T // blind + 1, params.N, params.N))
        for i in trange(T, position=gpu_id, desc=f"GPU {gpu_id}"):
            phi_x = excitable_cell.diffx(excitable_cell.phi)
            phi_y = excitable_cell.diffy(excitable_cell.phi)
            phi_prev = excitable_cell.phi
            A_prev = excitable_cell.A
            excitable_cell.update_phi(phi_x, phi_y)
            excitable_cell.update_A(phi_x, phi_y, phi_prev)
            excitable_cell.update_R(phi_x, phi_y, phi_prev, A_prev)

            if i % blind == 0:
                phi_save[i // blind] = excitable_cell.phi.get()
                A_save[i // blind] = excitable_cell.A.get()
                R_save[i // blind] = excitable_cell.R.get()
        return phi_save, A_save, R_save

from multiprocessing import Pool

def block_iterator(params_iterator, block_size):
    for i in range(0, len(params_iterator), block_size):
        yield params_iterator[i*block_size:min((i+1)*block_size, len(params_iterator))]


def gpu_dispatch(run_function, params_iterator, params_dicts, num_gpus, results_dir = Path('./results/')):
    """Dispatches the simulation to multiple GPUs
    
    Parameters
    ----------
    params_iterator : iterator
        an iterator that yields ExcitableCellParams
    num_gpus : int
        the number of GPUs to use, assumes that there are more GPUs than CPUs
    """

    for params_block, names_block in zip(block_iterator(params_iterator, num_gpus), block_iterator(params_dicts, num_gpus)):
        with Pool(num_gpus) as p:
            result = p.starmap(run_function, [(10, params, 100, 200, i) for i, params in enumerate(params_block)])
        
        for i in range(num_gpus):
            try:
                phi = result[i][0]
                A = result[i][1]
                R = result[i][2]
                format_string = "".join([f"{key}_{value}" for key, value in names_block[i].items()])
                np.savez(results_dir / f"results_{format_string}.npz", phi=phi, A=A, R=R)
            except:
                continue


def run_multi_cell_trial(num_cells: int, params: ExcitableCellParams, blind: int, simulation_runtime: int, gpu_id: int=0):
    with cp.cuda.Device(gpu_id):
        T = int(simulation_runtime // params.dt)
        cells = MultiExcitableCell2d(params, num_cells)
        phi = np.zeros((T//blind+1, num_cells, cells.pp.N, cells.pp.N))
        A = np.zeros((T//blind+1, num_cells, cells.pp.N, cells.pp.N))
        R = np.zeros((T//blind+1, num_cells, cells.pp.N, cells.pp.N))
        for t in trange(T, position=gpu_id, desc=f"GPU {gpu_id}", dynamic_ncols=True):
            cells.update_collision_force()
            cells.phi_x = cells.diffx(cells.phi)
            cells.phi_y = cells.diffy(cells.phi)
            cells.phi_prev = cells.phi
            cells.update_phi()
            cells.update_A()
            cells.update_R()
            if t % 100 == 0:
                cells.update_all_grid_locations(cells.pp.N//8, cells.pp.N//8)
                phi[t//blind] = cells.phi.get()
                A[t//blind] = cells.A.get()
                R[t//blind] = cells.R.get()

        return phi, A, R




from itertools import product

if __name__ == "__main__":
    # params_to_modify = [{'xi': xi, 'alpha': alpha, 'tau': tau} for xi, alpha, tau in product(range(2, 10, 2), np.arange(1, 5, 0.5), np.arange(1, 10, 1))]
    # params_iterator = [ExcitableCellParams(**params) for params in params_to_modify]
    # gpu_dispatch(params_iterator, 4, params_to_modify)
    num_cells = 10
    params_to_modify = [{'alpha': alpha, 'tau': tau} for alpha, tau in product([9, 11], [5, 10])]
    params_list = [ExcitableCellParams(xi=1, initial_offset_x=50, **params) for params in params_to_modify]
    print(params_to_modify)
    blind = 100
    simulation_time = 200

    # gpu_dispatch(run_multi_cell_trial, params_list, params_to_modify, 4)
    phi, A, R = run_multi_cell_trial(num_cells, params_list[2], blind, simulation_time)

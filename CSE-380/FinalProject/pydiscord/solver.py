import os
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional

import numpy as np
import scipy as sp

np.set_printoptions(linewidth=250)

def avg_flux(xi, xe, phi_i, mu, Q, sigma_t):
    dx = xe - xi
    tau = sigma_t * dx / mu

    avg_flux = (Q / sigma_t) + (phi_i - Q / sigma_t) * ((-1 * np.expm1(-tau)) / tau)

    return avg_flux

def list_split(input_list):
    assert len(input_list) % 2 == 0
    midpoint = len(input_list) // 2
    return input_list[:midpoint], input_list[midpoint:]


@dataclass
class Material:
    name: str
    total: float
    scatter: float
    Q: float
    bounds: Tuple[float, float]

    absorption: float = None

    def __post_init__(self):
        self.absorption: float = self.total - self.scatter

@dataclass
class Settings:
    # name: str
    # dst: str
    phiL: [float, str]
    phiR: [float, str]
    num_nodes: int
    sn: Optional[int] = None
    quadrature: Optional[dict] = None

class Matrix:
    def __init__(self, make:bool=False, solve:bool=False):
        self.matrix = None
        self.solved = None

        if make:
            self.make_matrix()

        if solve:
            self.solve_matrix()

    def make_matrix(self):
        self.matrix = None

    def solve_matrix(self):
        self.solved = None

    def solve_matrix(self, b:float=1):
        """
        form Ax=B
        """
        # scipy
        A = self.matrix.tocsr()
        b = np.ones(self.N) * b # todo make better
        x = sp.sparse.linalg.spsolve(
            A=A,
            b=b
        )
        self.solved = x

    def write_file(self, dst='./matrix.txt'):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        matrix_str= f"""
A:
{self.matrix}

Solved:
{self.solved}
"""
        with open(dst, 'w') as f:
            f.write(matrix_str)


class RightFlux(Matrix):
    """
    Right going flux
    """
    def __init__(self,
                 x: np.ndarray,
                 sigma_t: np.ndarray,
                 mu: float,
                 *args, **kwargs
                 ):

        self.x = x
        self.dx = x[1] - x[0]
        self.sigma_t = sigma_t
        self.mu = mu
        self.tau = self.sigma_t * self.dx / self.mu

        self.N = len(x)
        self.shape: Tuple = (self.N, self.N)

        super().__init__(*args, **kwargs)

    def make_matrix(self):
        diagonals = []
        diagonals.append(np.ones(self.N).tolist()) # D center

        assert len(self.tau) == self.N - 1
        diagonals.append((-1.0 * np.exp(-self.tau)).tolist()) # L1
        offset = [0, -1]

        self.matrix = sp.sparse.diags(
            diagonals=diagonals,
            shape=self.shape,
            offsets=offset,
        )


    def solve_matrix(self, phi0=0.0, Q_source: np.ndarray = None, Q_scatter: np.ndarray = None):
        """
        form Ax=B
        """
        if Q_source is None:
            Q_source = np.zeros(self.N - 1)
        else:
            assert len(Q_source) == self.N - 1

        if Q_scatter is None:
            Q_scatter = np.zeros(self.N - 1)
        else:
            assert len(Q_scatter) == self.N - 1

        b = (Q_source/2 + Q_scatter) / self.sigma_t * (-np.expm1(-self.tau)) # Q = (source + scatter)

        b = np.insert(b.copy(), 0, 0.0)
        b[0] += phi0

        A = self.matrix.tocsr()

        x = sp.sparse.linalg.spsolve(
            A=A,
            b=b,
        )
        self.solved = x.copy()


class LeftFlux(RightFlux):
    """
    Left going flux
    """
    def make_matrix(self):
        diagonals = []
        diagonals.append(np.ones(self.N).tolist())  # D center
        assert len(self.tau) == self.N - 1
        diagonals.append((-1.0 * np.exp(-1.0 * self.tau)).tolist())  # L1

        offset = [0, +1]

        self.matrix = sp.sparse.diags(
            diagonals=diagonals,
            shape=self.shape,
            offsets=offset,
        )

    def solve_matrix(self, phiN=0.0, Q_source: np.ndarray = None, Q_scatter: np.ndarray = None):
        """
        form Ax=B
        """
        if Q_source is None:
            Q_source = np.zeros(self.N - 1)
        else:
            assert len(Q_source) == self.N - 1

        if Q_scatter is None:
            Q_scatter = np.zeros(self.N - 1)
        else:
            assert len(Q_scatter) == self.N - 1

        b = (Q_source/2 + Q_scatter) / self.sigma_t * (1.0 - np.exp(-self.tau)) # Q = source + scatter

        b = np.insert(b.copy(), b.size, 0.0)
        b[-1] += phiN

        A = self.matrix.tocsr()

        x = sp.sparse.linalg.spsolve(
            A=A,
            b=b,
        )
        self.solved = x

def solve_flux(materials: List[Material], settings: Settings) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L_bound = +np.inf
    R_bound = -np.inf

    for idx in range(0, len(materials)):
        mat: Material = materials[idx]
        if mat.bounds[0] < L_bound:
            L_bound: float = mat.bounds[0]

        if mat.bounds[1] > R_bound:
            R_bound: float = mat.bounds[1]

    assert L_bound != +np.inf or not R_bound != -np.inf

    x_edges = np.linspace(L_bound, R_bound, num=settings.num_nodes + 1)
    dx = (R_bound - L_bound) / settings.num_nodes
    x_center = np.array([L_bound + i * dx for i in range(settings.num_nodes)]) + dx / 2

    nodal_mats = []
    nodal_abs = []
    nodal_D = []
    nodal_total = []
    nodal_scatter = []
    nodal_Q = []

    for idx, x_idx in enumerate(x_center):
        for jdx, mat_jdx in enumerate(materials):
            if mat_jdx.bounds[0] <= x_idx <= mat_jdx.bounds[1]:
                nodal_mats.append(mat_jdx)
                nodal_abs.append(mat_jdx.absorption)
                nodal_total.append(mat_jdx.total)
                nodal_scatter.append(mat_jdx.scatter)
                nodal_Q.append(mat_jdx.Q)
                break

    nodal_abs = np.array(nodal_abs)
    nodal_total = np.array(nodal_total)
    nodal_scatter = np.array(nodal_scatter)
    nodal_Q = np.array(nodal_Q)

    flux_error = 1.0
    max_step = 5e3

    Q_scatter = None
    angular_flux = None

    if settings.quadrature is None:
        assert settings.sn is not None
        mu, w = np.polynomial.legendre.leggauss(settings.sn)
    elif type(settings.quadrature) is dict:
        mu = np.array(settings.quadrature['mu'])
        w =  np.array(settings.quadrature['w'])
    else:
        raise ValueError
    left_mu, right_mu = list_split(mu)
    left_w, right_w = list_split(w)

    phiL0 = 0.0
    phiRN = 0.0

    total_flux = np.zeros(len(x_edges))
    total_avg_flux = np.zeros(len(x_edges)-1)

    Q_scatter = np.ones(len(x_edges)-1) * 1.0
    Q_source = np.ones(len(x_edges)-1) * nodal_Q

    for s in np.arange(max_step):
        angular_flux = np.zeros((len(mu), len(x_edges)))

        right_flux = np.zeros(len(x_edges))
        right_avg_flux = np.zeros(len(x_center))

        left_flux = np.zeros(len(x_edges))
        left_avg_flux = np.zeros(len(x_center))

        weights = np.zeros(len(mu))
        angles = np.zeros(len(mu))
        position_index = 0

        for idx in range(0, len(left_mu)):
            left_ordinance = left_mu[idx]
            left_weight = left_w[idx]
            lf = LeftFlux(  # left going
                x=x_edges,
                sigma_t=nodal_total,
                mu=-1.0 * left_ordinance,
                make=True,
            )
            # print(phiRN)
            if isinstance(settings.phiR, str):
                if settings.phiR.startswith('ref'):
                    phiN = phiRN
                elif settings.phiR.startswith('vac'):
                    phiN = 0.0
                else:
                    raise ValueError("phiR not recognized")
            else:
                phiN = settings.phiR

            lf.solve_matrix(phiN=phiN, Q_source=Q_source, Q_scatter=Q_scatter)  # starts at left


            angular_flux[position_index, :] = lf.solved # stacking
            weights[position_index] = left_weight
            angles[position_index] = left_ordinance
            left_flux += lf.solved * left_weight
            left_avg_flux += avg_flux(
                xi=x_edges[:-1],
                xe=x_edges[1:],
                phi_i=lf.solved.copy()[:-1],
                mu=left_ordinance,
                Q=Q_source/2 + Q_scatter,
                sigma_t=nodal_total,
            ) * left_weight

            position_index += 1

        for idx in range(0, len(right_mu)):
            right_ordinance = right_mu[idx]
            right_weight = right_w[idx]

            rf = RightFlux(  # right going
                        x=x_edges,
                        sigma_t=nodal_total,
                        mu=right_ordinance,
                        make=True,
                    )

            if isinstance(settings.phiL, str):
                if settings.phiL.startswith('ref'):
                    phi0 = phiL0
                elif settings.phiL.startswith('vac'):
                    phi0 = 0.0
                else:
                    raise ValueError("phiR not recognized")
            else:
                phi0 = settings.phiL

            rf.solve_matrix(phi0=phi0, Q_source=Q_source, Q_scatter=Q_scatter)  # starts at left

            angular_flux[position_index, :] = rf.solved.copy()
            weights[position_index] = right_weight
            angles[position_index] = right_ordinance
            right_flux += rf.solved.copy() * right_weight
            right_avg_flux += avg_flux(
                xi=x_edges[:-1],
                xe=x_edges[1:],
                phi_i=rf.solved.copy()[:-1],
                mu=right_ordinance,
                Q=Q_source/2 + Q_scatter,
                sigma_t=nodal_total,
            ) * right_weight

            position_index += 1

        old_flux = total_flux.copy()
        old_avg_flux = total_avg_flux.copy()

        total_flux = right_flux + left_flux
        total_avg_flux = right_avg_flux + left_avg_flux

        phiL0 = left_flux[0]
        phiRN = right_flux[-1]

        Q_new = nodal_scatter / 2 * total_avg_flux

        flux_error = np.linalg.norm((total_flux - old_flux)/(total_flux + 1e-16))

        Q_error = np.linalg.norm((Q_new - Q_scatter) / (Q_new + 1e-16))


        print(f'({int(s+1)}) dPhi/di = {flux_error * 100:.2E} % | dQ/di = {Q_error * 100:.2E} %')
        if flux_error < 1e-10:
            break

        Q_scatter = Q_new

    ##########################################################
    # end of Q update loop
    ##########################################################

    print(f'End of solution: flux error: {flux_error:.2E} \n')

    current = np.dot(w * mu, angular_flux)

    # returns a tuple:
    #   x_edges: edge positions
    #   total_flux: edge-wise scalar flux
    #   current: edge-wise current
    #   angular_flux: angular flux (num_angles x num_edges)
    #   mu: quadrature angles
    #   w: quadrature weights
    #   x_center: cell center positions
    return x_edges, total_flux, current, angular_flux, mu, w, x_center
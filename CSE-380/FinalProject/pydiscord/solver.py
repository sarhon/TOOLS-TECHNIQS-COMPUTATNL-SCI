import os
from dataclasses import dataclass
from pprint import pprint
from typing import Tuple, List, Union, Optional

import colorsys

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
colors = plt.cm.tab10.colors

np.set_printoptions(linewidth=250)

def avg_flux(xi, xe, phi_i, mu, Q, sigma_t):
    dx = xe - xi
    tau = sigma_t * dx / mu
    # un-simplified
    # avg_flux = (phi_i * term * mu / sigma_t + Q/sigma_t*(dx-mu/sigma_t*term))/dx

    # simplified
    avg_flux = (Q / sigma_t) + (phi_i - Q / sigma_t) * ((-1 * np.expm1(-tau)) / tau)

    # old and likely in correct
    # avg_flux = phi_i * term + (Q / sigma_t) * (1 - term)

    return avg_flux

def list_split(input_list):
    assert len(input_list) % 2 == 0
    midpoint = len(input_list) // 2
    return input_list[:midpoint], input_list[midpoint:]


def generate_colors(n, saturation=0.65, value=0.9):
    colors = []
    for i in range(n):
        # Evenly spaced hue (0-1)
        hue = i / n
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to hex string and append to the list
        colors.append('#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255)))
    return colors

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

        # pprint(self.matrix.todense())

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
                # phi_e=lf.solved[1:],
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
            # print(Q)

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
            # print('rf.solved', rf.solved)

            angular_flux[position_index, :] = rf.solved.copy()
            weights[position_index] = right_weight
            angles[position_index] = right_ordinance
            right_flux += rf.solved.copy() * right_weight
            # print('right_flux', right_flux)
            right_avg_flux += avg_flux(
                xi=x_edges[:-1],
                xe=x_edges[1:],
                phi_i=rf.solved.copy()[:-1],
                mu=right_ordinance,
                Q=Q_source/2 + Q_scatter,
                sigma_t=nodal_total,
            ) * right_weight
            # print('right_avg_flux', right_avg_flux)

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

    # print(current)

    # returns a tuple:
    #   x_edges: edge positions
    #   total_flux: edge-wise scalar flux
    #   current: edge-wise current
    #   angular_flux: angular flux (num_angles x num_edges)
    #   mu: quadrature angles
    #   w: quadrature weights
    #   x_center: cell center positions
    return x_edges, total_flux, current, angular_flux, mu, w, x_center
    # , angular_flux, current

    # x_edges
    # total_flux
    # nodal_flux = 0.5 * (total_flux[:-1] + total_flux[1:])
    # rr_abs = nodal_flux * nodal_abs
    # rr_total = nodal_flux * nodal_total
    #
    # print(f'Current(x=0)={current[0]:E}')
    # print(f'Current(x=N)={current[-1]:E}')
    #
    # results = {
    #     'x_edges': x_edges,
    #     'x_center': x_center,
    #     'total_flux': total_flux,
    #     'current': current,
    #     'nodal_flux': nodal_flux,
    #     'rr_abs': rr_abs,
    #     'rr_total': rr_total,
    # }

    # fig, ax = plt.subplots()
    # palette = generate_colors(len(angular_flux))
    #
    # for idx in range(0, len(angular_flux)):
    #     angle = angles[idx]
    #     af = angular_flux[idx]
    #     color = palette[idx]
    #     ax.plot(x_edges, af,
    #             '.-', label=r'$\mu$='+f'{angle:.2F}',
    #             color=color)
    #
    # ax.plot(x_edges, total_flux, '.-', color='black', label='Total Flux')
    #
    #
    # for idx, material in enumerate(materials):
    #     xmin, xmax = material.bounds
    #     color = colors[idx % len(colors)]
    #     ax.axvspan(xmin, xmax, color=color, alpha=0.3, label=material.name)
    #
    # if len(angular_flux) <= 8:
    #     ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    #
    # # plt.ylim([0, 1.5])
    # plt.xlabel(r'Position $[cm]$')
    # plt.ylabel(r'Fluence $[\frac{n}{cm^2}]$')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(settings.dst + f'/flux.png', dpi=600)
    # plt.close()
    #
    # fig, ax = plt.subplots()
    # max_value = -1
    # for idx in range(0, len(angular_flux)):
    #     angle = angles[idx]
    #     af = angular_flux[idx]
    #     color = palette[idx]
    #     ax.scatter(angle, np.linalg.norm(af),
    #                label=r'$\mu$=' + f'{angle:.2F}',
    #                color=color)
    #     max_value = max(max_value, np.linalg.norm(af))
    #
    # assert max_value > 0.0
    # if len(angular_flux) <= 16:
    #     ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    #
    # plt.ylabel(r'$||{\Phi_k}||$')
    # plt.xlabel(r'$\mu$')
    # plt.ylim([0.0, max_value*1.1])
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(settings.dst + f'/mu.png', dpi=600)
    # plt.close()
    #
    # # current = -1.0 * nodal_D * dflux_dx
    #
    #
    #
    #
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(x_edges, current, '.-', label='Current')
    # plt.xlabel('x $[cm]$')
    # plt.ylabel(r'Current $[\frac{n}{cm^2 s}]$')
    #
    # for idx, material in enumerate(materials):
    #     xmin, xmax = material.bounds
    #     color = colors[idx % len(colors)]
    #     ax.axvspan(xmin, xmax, color=color, alpha=0.3, label=material.name)
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(settings.dst + f'/current.png', dpi=600)
    # plt.close()
    #
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(x_center, rr_abs, '.-', label='Reaction Rate')
    # plt.xlabel('x $[cm]$')
    # plt.ylabel(r'Absorption Rate $[\frac{abs}{cm^2 s}]$')
    #
    # for idx, material in enumerate(materials):
    #     xmin, xmax = material.bounds
    #     color = colors[idx % len(colors)]
    #     ax.axvspan(xmin, xmax, color=color, alpha=0.3, label=material.name)
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(settings.dst + f'/rr_abs.png', dpi=600)
    # plt.close()
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(x_center, rr_total, '.-', label='Reaction Rate')
    # plt.xlabel('x $[cm]$')
    # plt.ylabel(r'Total Reaction Rate $[\frac{abs}{cm^2 s}]$')
    #
    # for idx, material in enumerate(materials):
    #     xmin, xmax = material.bounds
    #     color = colors[idx % len(colors)]
    #     ax.axvspan(xmin, xmax, color=color, alpha=0.3, label=material.name)
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(settings.dst + f'/rr_total.png', dpi=600)
    # plt.close()


# outdated, but is still right for a single infinite material
# def inf_flux(material: Material):
#     return (1/2/material.total) * (1 + material.scatter/(material.total - material.scatter + 1e-16)) * material.Q

# def main():
#
#
#     L = 100.0
#     nums = int(L) + 1
#
#     sns = [
#         # 2, 4, 8,
#         # 16, 32,
#         # 64,
#         100
#         ]
#
#
#     materials_a = [
#             Material(
#                 name='absorber',
#                 total=10.0,
#                 scatter=2.0,
#                 Q=0.0,
#                 bounds=(0.0, 4.0)
#             ),
#             Material(
#                 name='isotropic',
#                 total=1.0,
#                 scatter=0.9,
#                 Q=1.0,
#                 bounds=(4.0, 10.0),
#             ),
#             Material(
#                 name='absorber',
#                 total=10.0,
#                 scatter=2.0,
#                 Q=0.0,
#                 bounds=(10.0, 14.0)
#             )
#         ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_a = Settings(
#             name=f'{sn}',
#             phiL=0.0,
#             phiR=0.0,
#             num_nodes= 100,
#             sn = sn,
#             dst = f'./plots/a/{sn}/'
#         )
#
#         solve_flux(materials=materials_a, settings=settings_a)
#
#     # B)
#
#     materials_b = [
#             Material(
#                 name='absorber',
#                 total=10.0,
#                 scatter=2.0,
#                 Q=0.0,
#                 bounds=(0.0, 0.5)
#             ),
#             Material(
#                 name='scatter',
#                 total=2.0,
#                 scatter=1.99,
#                 Q=0.0,
#                 bounds=(0.5, 6.5),
#             ),
#             Material(
#                 name='isotropic',
#                 total=0.1,
#                 scatter=0.0,
#                 Q=1.0,
#                 bounds=(6.5, 36.5)
#             )
#         ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_b = Settings(
#             name=f'{sn}',
#             phiL=0.0,
#             phiR='ref',
#             num_nodes=100,
#             sn=sn,
#             dst=f'./plots/b/{sn}/'
#         )
#
#         solve_flux(materials=materials_b, settings=settings_b)
#
#     materials_c = [
#             Material(
#                 name='isotropic',
#                 total=0.1,
#                 scatter=0.0,
#                 Q=1.0,
#                 bounds=(0.0, 3.0)
#             ),
#             Material(
#                 name='air',
#                 total=0.01,
#                 scatter=0.006,
#                 Q=0.0,
#                 bounds=(3.0, 13.0)
#             ),
#             Material(
#                 name='reflector',
#                 total=      2.0,
#                 scatter= 1.8,
#                 Q= 0.0,
#                 bounds=(13.0, 18.0)
#             ),
#             Material(
#                 name='isotropic',
#                 total=0.1,
#                 scatter=0.0,
#                 Q=1.0,
#                 bounds=(18.0, 28.0)
#             ),
#             Material(
#                 name='scatter',
#                 total=2.0,
#                 scatter=1.99,
#                 Q=0.0,
#                 bounds=(28.0, 33.0)
#             ),
#             Material(
#                 name='absorber',
#                 total=10.0,
#                 scatter=2.0,
#                 Q=0.0,
#                 bounds=(33.0, 53.0)
#             )
#         ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_c = Settings(
#             name=f'{sn}',
#             phiL='ref',
#             phiR='ref',
#             num_nodes=100,
#             sn=sn,
#             dst=f'./plots/c/{sn}/'
#         )
#
#         solve_flux(materials=materials_c, settings=settings_c)
#
#         for sn in sns:
#             print(f'sn = {sn}')
#             settings_b = Settings(
#                 name=f'{sn}',
#                 phiL=0.0,
#                 phiR='ref',
#                 num_nodes=100,
#                 sn=sn,
#                 dst=f'./plots/b/{sn}/'
#             )
#
#             solve_flux(materials=materials_b, settings=settings_b)
#
#     materials_q1 = [
#         Material(
#             name='hw3-mat',
#             total=1.0,
#             scatter=0.9,
#             Q=1.0,
#             bounds=(0.0, 100.0)
#         )
#     ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_q1 = Settings(
#             name=f'{sn}',
#             phiL='ref',
#             phiR='ref',
#             num_nodes=100,
#             sn=sn,
#             dst=f'./plots/q1/{sn}/'
#         )
#
#         solve_flux(materials=materials_q1, settings=settings_q1)
#
#     materials_q2 = [
#         Material(
#             name='hw3-mat',
#             total=1.0,
#             scatter=0.9,
#             Q=0.0,
#             bounds=(0.0, 100.0)
#         )
#     ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_q2 = Settings(
#             name=f'{sn}',
#             phiL=1.0,
#             phiR='vac',
#             num_nodes=100,
#             sn=sn,
#             dst=f'./plots/q2/{sn}/'
#         )
#
#         solve_flux(materials=materials_q2, settings=settings_q2)
#
#     materials_q3 = [
#         Material(
#             name='hw3-mat',
#             total=1.0,
#             scatter=0.9,
#             Q=1.0,
#             bounds=(0.0, 100.0)
#         )
#     ]
#
#     for sn in sns:
#         print(f'sn = {sn}')
#         settings_q3 = Settings(
#             name=f'{sn}',
#             phiL='vac',
#             phiR='vac',
#             num_nodes=100,
#             sn=sn,
#             dst=f'./plots/q3/{sn}/'
#         )
#
#         solve_flux(materials=materials_q3, settings=settings_q3)
#
#             # print(f'expected angular flux: {inf_flux(materials)}\n')
#
#     pass
#
#
#
# if __name__ == '__main__':
#     main()
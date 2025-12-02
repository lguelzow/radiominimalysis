import numpy as np
from scipy import integrate
import copy


def calculate_geomagnetic_energy(r, f, param=None, wrapper=None, x=None):

    if wrapper is not None:
        y = wrapper(f(r, **param))
    elif param is None:
        y = f(r)
    else:
        y = f(r, **param)

    if x is not None:
        r = x

    sort = np.argsort(r)
    r = copy.deepcopy(r)[sort]
    y = copy.deepcopy(y)[sort]

    E_geo = integrate.trapezoid(y=y * r, x=r) * 2 * np.pi

    return E_geo


def calculate_geomagnetic_energy_quad(ldf_geo, rmin, rmax, params):
    # not tested or validated

    def integral_ldf_geo(r, **kwargs):
        return 2 * np.pi * r * ldf_geo(r, **kwargs)

    e_geo = integrate.quad(integral_ldf_geo, rmin, rmax, args=params)

    return e_geo


def calculate_geo_magnetic_energy_fluence(f_vxB, phi, alpha, a):
    return f_vxB / (1 + np.cos(phi) / np.abs(np.sin(alpha)) * np.sqrt(a)) ** 2.

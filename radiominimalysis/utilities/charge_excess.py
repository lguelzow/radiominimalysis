import numpy as np


def f_geo_on_vxB(f_pos, f_neg):
    return 1 / 4. * np.square(np.sqrt(f_pos) + np.sqrt(f_neg))


def f_ce_on_vxB(f_pos, f_neg):
    return 1 / 4. * np.square(np.sqrt(f_pos) - np.sqrt(f_neg))


def ce_fraction_param_xmax_combined(xdata, A=1.05448757e-04, B=4.20729929e+00, C=8.09690461e-04, C2=0.00333554):
    zenith, distance, Xmax = xdata
    return A * distance * np.cos(zenith) ** B * np.exp(C * distance) * np.exp(C2 * Xmax)


# ce fractions needs to be corrected with xmax before (used from freddy)
def ce_fraction_param_xmax_reduced(xdata, A, B, C):
    zenith, distance = xdata
    return A * distance * np.cos(zenith) ** B * np.exp(C * distance)


# Parametrisierung des ce-Anteils (Gesamtenergie)
def A_xmax_fit(input_data, A_2, B_2, C_2):
    zenith, xmax = input_data
    return A_2 * np.cos(zenith) ** B_2 * np.exp(C_2 * xmax)


def A_rho_fit(rho_max, p0, p1, p2=0):
    rho_avg = 0.4  # in kg/m^3, needed for correction of radio estimator
    return p0 * np.exp(p1 * (rho_max - rho_avg)) + p2


def A_rho_fit_mod(rho_max, p0, p1, pp0, pp1):
    rho_avg = 0.4151646429189798  # in kg/m^3, needed for correction of radio estimator
    A = p0 * np.exp(p1 * (rho_max -  rho_avg)) + pp0 * rho_max ** pp1
    return A


def charge_excess_fraction_free_density_correction(xdata, ce0, ce1, ce2):
    r, d, rho = xdata
    # rho_avg = 0.4
    # return charge_excess_fraction_shaped(xdata, ce0, ce1, ce2, ce3)
    # ce2 = [0.03523276  7.85129472 - 0.18438715]
    return ce0 * r / d * np.exp(ce1 * r / 1000) * ce2


def charge_excess_fraction_new_density_correction_const(xdata, ce0, ce1, p1, p2, p3):
    r, d, rho = xdata
    return ce0 * r / d * np.exp(ce1 * r / 1000) * (p1 * rho ** p2 + p3)


def charge_excess_fraction_density_scaling_icrc21(rho, p1=1.66965694e+01, p2=3.31954904e+00, p3=-5.73577715e-03):
    return p1 * rho ** p2 + p3
charge_excess_fraction_density_scaling_icrc21.latex = r"$p_1 \rho\mathrm{max^{p_2} + p_3"
charge_excess_fraction_density_scaling_icrc21.latex_with_num = r"$\left(\rho_\mathrm{max} / 0.428\,\mathrm{kg}\,\mathrm{m}^{-3}\right)^{3.32} - 0.0057$"


def charge_excess_fraction_icrc21(xdata, ce01, ce02, ce1, p1, p2, p3):
    r, d, rho = xdata
    ce0 = ce01 * d + ce02
    return ce0 * r / d * np.exp(ce1 * r / 1000) * charge_excess_fraction_density_scaling_icrc21(rho, p1, p2, p3)


def charge_excess_fraction_icrc19(xdata, A, B, p0, p1):
    r, d, rho = xdata
    rho_avg = 0.4
    return A * r / d * np.exp(B * r / 1000) * (p0 + np.exp(p1 * (rho - rho_avg)))


def get_density_correction(rho, p0, p1):
    rho_avg = 0.4
    return (p0 + np.exp(p1 * (rho - rho_avg))) ** 2

# Lukas' parametrisation
# first change: use r_cherenkov instead of off-axis angle
def GRAND_charge_excess(xdata, par1, d1, r1, rho1, exp1, par2):
    r, d, rho, r_che = xdata
    # print(r / r1)
    return (par1 - d * d1) * r / r_che * np.exp(r / r_che * r1) * (rho1 * rho ** exp1 - par2)
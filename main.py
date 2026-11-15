"""
PHYS-213: Beta decay Energy/momentum graph makers
A program that creates a graph for the energy and momentum distributions of a
particle in a beta plur or minus decay nuclear reaction

author Diego Guerrero
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

m_e = 0.511  # MeV/c^2
c = 1.0      # Natural units (MeV, MeV/c)

print("--- Beta Decay Energy/momentum graph ---")
Z_parent = float(input("Enter atomic charge of parent Z: ") or 29)
m_parent_u = float(input("Enter atomic mass of parent X [u]: ") or 63.929766)
m_daughter_u = float(input("Enter atomic mass of daughter X_Z+1 [u]: ") or 63.929145)

Z_daughter = Z_parent + 1

Q = (m_parent_u - m_daughter_u) * 931.494  # MeV


def F(T_e):
    spectrum = (
        np.sqrt(T_e**2 + 2 * m_e * T_e)
        * ((Q - T_e) ** 2)
        * (T_e + m_e)
    )
    return np.where(T_e <= Q, spectrum, 0)


p_max = np.sqrt((Q + m_e)**2 - m_e**2)

TArray = np.arange(0.0001, Q, 0.0001)
NTArray = F(TArray)

I = simpson(TArray * NTArray, x=TArray)
NTArray = NTArray / I
C = (c * c) / I

def N_p(p):
    spectrum = (
        (C / c**2)
        * (p**2)
        * (Q - np.sqrt(p**2 + m_e**2) + m_e) ** 2
    )
    T_e = np.sqrt(p**2 + m_e**2) - m_e
    return np.where(T_e <= Q, spectrum, 0)

pArray = np.arange(0.0001, p_max, 0.0001)
NPArray = N_p(pArray)

fig, (ax1, ax3) = plt.subplots(2, 1, constrained_layout=True)

ax1.plot(pArray, NPArray)
ax1.set_ylabel("$N(p)$")
ax1.set_xlabel("$p$ (MeV/c)")
ax1.set_xlim(xmin=0)
ax1.set_ylim(ymin=0)
ax1.set_title("Distributions of B- decay in Copper-63")




print(f"Q-value: {Q:.6f} MeV")


mu_pure = simpson(TArray * NTArray, x=TArray) / simpson(NTArray, x=TArray)
print(f"Pure Phase Space Mean <T_e>: {mu_pure:.6f} MeV")

# --- Fermi Function ---
alpha = 1.0 / 137.036
def Fermi(Z_prime, p):
    E_e = np.sqrt(p**2 + m_e**2)
    v_over_c = p / E_e
    eta = (Z_prime * alpha) / v_over_c
    two_pi_eta = 2.0 * np.pi * eta
    return two_pi_eta / (1.0 - np.exp(-two_pi_eta))


p_T = np.sqrt(TArray**2 + 2 * m_e * TArray)
NTArray_fermi = Fermi(Z_daughter, p_T) * F(TArray)

I = simpson(TArray * NTArray_fermi, x=TArray)
NTArray_fermi = NTArray_fermi / I
mu_fermi = simpson(TArray * NTArray_fermi, x=TArray) / simpson(NTArray_fermi, x=TArray)
print(f"Fermi Theory Mean <T_e>: {mu_fermi:.6f} MeV")

ax3.plot(TArray, NTArray_fermi)
ax3.set_ylabel("$N(T_e)$")
ax3.set_xlabel("$T_e$ (MeV)")
ax3.set_xlim(xmin=0)
ax3.set_ylim(ymin=0)
plt.show()



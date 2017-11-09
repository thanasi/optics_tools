#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:46:10 2017

@author: thanasi
"""

import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()
Q = ureg.Quantity

# %%

# pulse energy
e0 = Q(250, "mJ")

# pulse duration
tau = Q(4, "ns")

# peak power
# conservative estimate, assuming gaussian temporal profile
peak_pow = 2 * (e0 / tau).to("W")

# beam diameter
d0 = Q(7, "mm")

# beam area
A = (np.pi * d0**2 / 4).to("cm^2")

energy_density = (e0 / A).to("J/cm^2")
peak_pow_dens = (peak_pow / A).to("MW/cm^2")

print("-"*30)
print("E-Density: {:0.3f~})".format(energy_density))
print("P-Density: {:0.1f~}".format(peak_pow_dens))
print("-"*30)

# %% 
# check necessary extinction for photodetector
# Thorlabs DET025A

# power damage threshold
p_dam_thresh = Q(18, "mW")

# detector active area
det_area = Q(250, "um")**2

# maximum allowable power density
max_det_pow_dens = p_dam_thresh / det_area

# reduction factor needed from laser beam
red = (peak_pow_dens / max_det_pow_dens).to("").magnitude
red_OD = np.ceil(np.log10(red))

print("-"*30)
print("Power Reduction Needed: {:1.1g}".format(red))
print("OD Needed: {:2.0f}".format(red_OD))
print("-"*30)






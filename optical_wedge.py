import numpy as np
import pandas as pd

#%% optical properties

# index of refraction @ 532nm
n = 1.460678

# power reflection/transmission coeff
# theoretical
#R = ((n-1)/(n+1))**2
#T = 1-R

# empirical
T = 0.93730864 
R = 1-T

#%% set up the geometry

# wedge angle
A = np.deg2rad(5)

# incidence angle
B = np.deg2rad(2)

#%% calculate transmission & deviation

# note: 
# odd orders reflect back towards incident beam
# even orders transmit through the wedge


m = np.arange(-1,5)
even = (1-(m%2)).astype(bool)
odd = (m%2).astype(bool)

df = pd.DataFrame(index=m, columns=["L","Dir","EnergyFrac","Deviation_deg","Angle_deg"])

df.loc[odd, "Dir"] = "Back"
df.loc[even, "Dir"] = "Thru"

L = (1+m)/2 * odd + (m/2+1) * even
df["L"] = L
#df.loc[:0,"L"] = np.nan

F = lambda L: B + (2*L-1)*n*A
H = lambda L: B + 2*L*n*A


df["Angle_deg"] = np.rad2deg(F(L) * odd + H(L) * even)

df["Deviation_deg"] = np.rad2deg((B + H(L))*odd + (F(L)-B-A) * even)
df.loc[-1,"Deviation_deg"] = np.rad2deg(2*B)

df["EnergyFrac"] = T**2 * R**m
df.loc[-1,"EnergyFrac"] = R

df
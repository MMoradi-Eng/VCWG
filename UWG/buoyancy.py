import numpy
import math

"""
This Class is used to calculate buoyancy term for the TKE equation. (eq. 5.2, term IX, Krayenhoff, PhD thesis)
Developed by Scott Krayenhoff
University of Guelph, Guelph, Canada
Last update: March 2017
"""

class Buoyancy:
    def __init__(self, nz, dz, th, Km, th0, prandtl, g):
        self.nz = nz            # Number of grid points in vertical column
        self.dz = dz            # Grid resolution [m]
        self.th = th            # Potential temperature [K]
        self.Km = Km            # Turbulent diffusion coefficient [m^2 s^-1]
        self.th0 = th0          # Reference potential temperature [K]
        self.prandtl = prandtl  # Turbulent Prandtl number
        self.g = g              # Gravitational acceleration [m s^-2]

    def BuoProd(self):
        # Set a minimum value for diffusion coefficient
        cdmin = 0.01
        # Define buoyancy term [m^2 s^-3]
        bu = numpy.zeros(self.nz)
        # Set the buoyancy value at street level to zero
        bu[0] = 0
        # Calculate buoyancy using (eq. 5.2, term IX, Krayenhoff, PhD thesis)
        # buoyant production [m^2 s^-3] = (g/th0)*(Km/prandtl)*(dth/dz)
        for i in range(1,self.nz-1):
            # Discretize potential temperature gradient
            dthdz1 = (self.th[i]-self.th[i-1])/self.dz
            dthdz2 = (self.th[i+1]-self.th[i])/self.dz
            cdm = max(0.5*(self.Km[i]+self.Km[i+1])/self.prandtl,cdmin)

            dthmdz = 0.5*(dthdz1+dthdz2)

            bu[i] = -self.g*cdm*dthmdz/self.th0[i]

        bu[self.nz-1] = 0

        return bu
import numpy
import math

"""
Shear production calculation
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Alberto Martilli, Scott Krayenhoff, and Negin Nazarian
"""

# This Class is used to calculate shear term for the TKE equation. (eq. 5.2, term II, Krayenhoff, PhD thesis)
class Shear:
    def __init__(self,nz,dz,vx,vy,km):
        self.nz = nz    # Number of grid points in vertical column
        self.dz = dz    # Grid resolution [m]
        self.vx = vx    # x component of horizontal wind speed [m s^-1]
        self.vy = vy    # y component of horizontal wind speed [m s^-1]
        self.km = km    # Turbulent diffusion coefficient [m^2 s^-1]

    def ShearProd(self):
        # Define shear production term [m^2 s^-3]
        sh = numpy.zeros(self.nz)
        # Set the shear production value at street level to zero
        sh[0] = 0
        # Set a minimum value for diffusion coefficient
        cdmin = 0.01
        # Calculate shear production (eq. 5.2, term II, Krayenhoff, PhD thesis)
        # shear production [m^2 s^-3] = Km*[(du/dz)^2+(dv/dz)^2]
        for i in range(1,self.nz-1):
            # Discretize gradient of x and y components of velocities
            dudz1 = (self.vx[i]-self.vx[i-1])/self.dz
            dvdz1 = (self.vy[i]-self.vy[i-1])/self.dz
            dudz2 = (self.vx[i+1]-self.vx[i])/self.dz
            dvdz2 = (self.vy[i+1]-self.vy[i])/self.dz

            cdm = max(0.5*(self.km[i]+self.km[i+1]),cdmin)

            dumdz = 0.5*((dudz1**2+dvdz1**2)+(dudz2**2+dvdz2**2))

            sh[i] = cdm*dumdz

        sh[self.nz-1] = 0

        return sh
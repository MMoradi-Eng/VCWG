import numpy
import math

"""
Calculate the turbulent diffusion coefficient
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Scott Krayenhoff
"""

class CdTurb:
    def __init__(self,nz,Ck,tke,dlk,it,Ri_b,var_sens):
        self.nz = nz               # Number of grid points in vertical column
        self.Ck = Ck               # Coefficient used in the equation of diffusion coefficient (kappa)
        self.tke = tke             # Turbulent kinetic energy [m^2 s^-2]
        self.dlk = dlk             # Mixing length [m]
        self.it = it               # time step in column (1-D) model
        self.Ri_b = Ri_b           # Bulk Richardson number
        self.var_sens = var_sens   # any variable for sensitivity analysis @@@@@ sensitivity analysis @@@@@

    def TurbCoeff(self):
        # set critical bulk Richardson number. It is a measure to distinguish between stable and unstable conditions
        Ri_b_cr = 0.01

        # Define turbulent diffusion coefficient [m^2 s^-1]
        Km = numpy.zeros(self.nz+1)

        # Km should be zero at street level
        if self.Ri_b > Ri_b_cr:
            Km[0] = 0
        else:
            Km[0] = 0

        # Calculate turbulent diffusion coefficient [m^2 s^-1] (eq. 4.8, Krayenhoff, PhD thesis)
        # Km = Ck*lk*(TKE)^0.5
        for i in range(1,self.nz-1):
            # Discretize TKE and length scale (vertical resolution (dz) is kept constant)
            tke_m = (self.tke[i-1]+self.tke[i])/2
            dlk_m = (self.dlk[i-1]+self.dlk[i])/2

            # Depending on how bulk Richardson number is calculated (option1: based on surface temperature or
            # option2: based on air temperature), Ck in stable and unstable conditions are given values
            '''
            #Option 1: Rib based on surface temperature
            if self.Ri_b < Ri_b_cr:
                Ck = 1.2
                Km[i] = dlk_m * (math.sqrt(tke_m)) * Ck
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]
                #print(Ck)
            else:
                Ck = 9
                Km[i] = dlk_m * (math.sqrt(tke_m)) * Ck
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]
                #print(Ck)
            '''

            #Option 2: Rib based on air temperature

            if self.Ri_b > Ri_b_cr:

                # Constant coefficient for stable condition
                Ck = 8

                # Calculate turbulent diffusion coefficient [m^2 s^-1]
                Km[i] = dlk_m * (math.sqrt(tke_m)) * Ck

                # It is assumed that there is no mixing at the top of the domain. Thus, vertical gradient of Km is zero.
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]
            else:

                # Constant coefficient for unstable condition
                Ck = 8

                # Calculate turbulent diffusion coefficient [m^2 s^-1]
                Km[i] = dlk_m * (math.sqrt(tke_m)) * Ck

                # It is assumed that there is no mixing at the top of the domain. Thus, vertical gradient of Km is zero.
                Km[self.nz - 1] = Km[self.nz - 2]
                Km[self.nz] = Km[self.nz - 2]

        return Km
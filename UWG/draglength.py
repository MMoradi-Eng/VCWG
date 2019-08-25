import numpy
import math

"""
Calculate mixing length scale for the 1-D column model in the urban area
Developed by Scott Krayenhoff
University of Guelph, Guelph, Canada
Last update: March 2017
"""

class Drag_Length:

    def __init__(self, nz,nz_u, z, lambdap,lambdaf,bldHeight,Ceps,Ck,Cmu,pb):
        self.nz = nz                # number of points
        self.nz_u = nz_u            # number of canopy levels in the vertical
        self.z = z                  # vertical grid
        self.lambdap = lambdap      # Plan area density
        self.lambdaf = lambdaf      # Frontal area density
        self.bldHeight = bldHeight  # Average building height [m]
        self.Ceps = Ceps            # Coefficient for the destruction of turbulent dissipation rate
        self.Ck = Ck                # Coefficient used in the equation of diffusion coefficient
        self.Cmu = Cmu              # Coefficient which will be used to determine length scales
        self.pb = pb                # Probability that a building has a height greater or equal to z

    # Calculate sectional drag coefficient (Cdrag) for buildings w/o trees(eq. 4.17, Krayenhoff, PhD thesis)
    def Drag_Coef(self):
        Cdrag = numpy.zeros(self.nz)
        for i in range(0,self.nz):
            if (self.lambdaf*self.pb[i+1]) <= 0.33:
                Cdrag[i] = 7.3*((self.lambdaf*self.pb[i+1])**(0.62))
            else:
                Cdrag[i] = 3.67
        return Cdrag

   # Calculate turbulent and dissipation length scales (eq. 4.15 and 4.18, Krayenhoff, PhD thesis)
    def Length_Scale(self):
        a1 = 1.95
        a2 = 1.07
        # Calculate displacement height (eq. 4.19, Krayenhoff 2014, PhD thesis)
        disp = self.bldHeight * (self.lambdap ** (0.15))
        # Dissipation length scale [m]
        dls = numpy.zeros(self.nz)
        # Turbulent length scale [m]
        dlk = numpy.zeros(self.nz)
        for i in range(0,self.nz):
            zc = (self.z[i]+self.z[i+1])/2

            if self.bldHeight == 0:
                dls[i] = self.Ceps*a2*zc[i]
            elif (zc/self.bldHeight) <= 1:
                dls[i] = self.Ceps*a1*(self.bldHeight-disp)
            elif (zc/self.bldHeight) > 1 and (zc/self.bldHeight) <= 1.5:
                dls[i] = self.Ceps*a1*(zc-disp)
            elif (zc/self.bldHeight) > 1.5:
                d2 = (1-a1/a2)*1.5*self.bldHeight+(a1/a2)*disp
                dls[i] = self.Ceps*a2*(zc-d2)
            dlk[i] = self.Cmu*dls[i]/(self.Ceps*self.Ck)
        return dlk,dls

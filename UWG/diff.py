import numpy
import math

"""
Formulate linear system of equations to solve; make matrix of coefficients A and the right hand side RHS vector
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Alberto Martilli, Scott Krayenhoff
"""

# This Class is used to calculate the diffusion in 1D (by A. Martilli)
## 'iz1' and 'izf' should be calculated with the knowledge of starting from zero
## Calculation of  'fc' and 'df' are skipped
class Diff:
    def __init__(self,nz,dt,sf,vol,dz,rho):
        self.nz = nz
        self.dt = dt
        self.sf = sf
        self.vol = vol
        self.dz = dz
        self.rho = rho

    #Solver for applying constant value (2) at the bottom of domain and zero gradient (1) on the top of domain
    def Solver21(self,iz1,izf,var,srim,srex,k):

        #Make a vector and assign the turbulent diffusion coefficient for each element of the 1D domain
        kdz = numpy.zeros(self.nz+1)
        kdz[0] = self.rho[0]*self.sf[0]*k[0]/self.dz
        for i in range(1,self.nz):
            kdz[i] = self.rho[i]*self.sf[i]*k[i]/(2*self.dz)/2
        if izf > 1:
            kdz[self.nz] = self.sf[i]*self.rho[self.nz-1]*k[self.nz]/self.dz
        else:
            kdz[self.nz] = 0

        A = numpy.zeros((self.nz,3))
        RHS = numpy.zeros(self.nz)

        #For first element we need constant value
        A[0][0] = 0
        A[0][1] = 1
        A[0][2] = 0
        RHS[0] = var[0]

        #For interior cells i should vary from 1 to nz-2
        for i in range(1,self.nz-1):
            dzv = self.vol[i]*self.dz
            A[i][0] = -(1/self.rho[i])*kdz[i]*self.dt/dzv
            A[i][1] = (1/self.rho[i])*self.dt*(1/dzv)*(kdz[i]+kdz[i+1])+1-srim[i]*self.dt
            A[i][2] = -(1/self.rho[i])*kdz[i+1]*self.dt/dzv
            RHS[i] = var[i]+srex[i]*self.dt

        #On top of the domain apply zero gradient condition
        dzv = self.vol[self.nz - 1] * self.dz
        A[self.nz - 1][0] = -kdz[self.nz - 1] * self.dt / dzv
        A[self.nz - 1][1] = 1 + (self.dt * kdz[self.nz - 1] / dzv) - srim[self.nz - 1] * self.dt
        A[self.nz - 1][2] = 0
        RHS[self.nz - 1] = var[self.nz - 1] + srex[self.nz - 1] * self.dt

        return A,RHS

    # Solver for applying zero gradient (1) at the bottom of domain and constant value (2) on the top of domain
    def Solver12(self, iz1, izf, var, srim, srex, k):

        # Make a vector and assign the turbulent diffusion coefficient for each element of the 1D domain
        kdz = numpy.zeros(self.nz + 1)
        kdz[0] = self.rho[0] * self.sf[0] * k[0] / self.dz
        for i in range(1, self.nz):
            kdz[i] = self.rho[i] * self.sf[i] * k[i] / (2 * self.dz) / 2
        if izf > 1:
            kdz[self.nz] = self.sf[i] * self.rho[self.nz - 1] * k[self.nz] / self.dz
        else:
            kdz[self.nz] = 0

        A = numpy.zeros((self.nz, 3))
        RHS = numpy.zeros(self.nz)

        #On bottom of the domain apply zero gradient condition
        dzv = self.vol[0] * self.dz
        A[0][0] = -kdz[0] * self.dt / dzv
        A[0][1] = 1 + (self.dt * kdz[0] / dzv) - srim[0] * self.dt
        A[0][2] = 0
        RHS[0] = var[0] + srex[0] * self.dt

        #For interior cells i should vary from 1 to nz-2
        for i in range(1,self.nz-1):
            dzv = self.vol[i]*self.dz
            A[i][0] = -(1/self.rho[i])*kdz[i]*self.dt/dzv
            A[i][1] = (1/self.rho[i])*self.dt*(1/dzv)*(kdz[i]+kdz[i+1])+1-srim[i]*self.dt
            A[i][2] = -(1/self.rho[i])*kdz[i+1]*self.dt/dzv
            RHS[i] = var[i]+srex[i]*self.dt

        # For last element we need constant value
        A[self.nz - 1][0] = 0
        A[self.nz - 1][1] = 1
        A[self.nz - 1][2] = 0
        RHS[self.nz - 1] = var[self.nz - 1]

        return A, RHS
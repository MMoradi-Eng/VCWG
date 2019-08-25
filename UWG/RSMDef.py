import os
import numpy
import math
from pprint import pprint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Vertical Diffusion Model (VDM) for rural temperature profiles
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Bruno Bueno
"""

"""
% Rural Site & Vertical Diffusion Model (VDM)
% Calculates the vertical profiles of air temperature above the weather
% station per 'The UWG' (2012) Eq. 4, 5, 6.

properties
    lat;           % latitude [deg]
    lon;           % longitude [deg]
    GMT;           % GMT hour correction
    height         % average obstacle height [m]
    z0r;           % rural roughness length [m]
    disp;          % rural displacement length [m]
    z;             % vertical height [m]
    dz;            % vertical discretization [m]
    tempProf;      % potential temperature profile at the rural site [K]
    presProf;      % pressure profile at the rural site [Pa]
    tempRealProf;  % real temperature profile at the rural site [K]
    densityProfC;  % density profile at the center of layers [kg m^-3]
    densityProfS;  % density profile at the sides of layers [kg m^-3]
    windProf;      % wind profile at the rural site [m s^-1]
    ublPres;       % Average pressure at UBL [Pa]
end
"""

ppr = pprint

class RSMDef(object):

    Z_MESO_FILE_NAME = "z_meso.txt"

    def __init__(self,lat,lon,GMT,height,T_init,P_init,parameter,z,dz,nz):

        self.lat = lat                # latitude [deg]
        self.lon = lon                # longitude [deg]
        self.GMT = GMT                # GMT hour correction
        self.height = height          # rural average obstacle height [m]
        self.z0r = 0.1 * height       # rural roughness length [m] (Raupach,1991)
        self.disp = 0.5 * height      # rural displacement length [m]
        self.z = z                    # Vertical grid in the rural area (the same as urban area)
        self.dz = dz                  # Grid resolution [m] (the same as urban area)(as opposed to UWG, self.dz is kept constant)
        self.nz = nz                  # Number of grid points in vertical column


        # Initialize potential temperature profile in the rural area [K]
        self.tempProf = [T_init for x in range(self.nz)]
        # Initialize pressure profile in the rural area [Pa]
        self.presProf = [P_init for x in range(self.nz)]
        # Initialize real temperature profile in the rural area [K]
        self.tempRealProf = [T_init for x in range(self.nz)]
        # Initialize density profile at the center of layers in the rural area [kg m^-3]
        self.densityProfC = [None for x in range(self.nz)]
        # Initialize wind speed profile in the rural area [m s^-1]
        self.windProf = [1 for x in range(self.nz)]

        # Calculate pressure profile
        for iz in xrange(1,self.nz):
            self.presProf[iz] = (self.presProf[iz-1]**(parameter.r/parameter.cp) -\
               parameter.g/parameter.cp * (P_init**(parameter.r/parameter.cp)) * (1./self.tempProf[iz] +\
               1./self.tempProf[iz-1]) * 0.5 * self.dz)**(1./(parameter.r/parameter.cp))

        # Calculate real temperature profile [K]
        for iz in xrange(self.nz):
           self.tempRealProf[iz] = self.tempProf[iz] * (self.presProf[iz] / P_init)**(parameter.r/parameter.cp)

        # Calculate density profiles [kg m^-3]
        for iz in xrange(self.nz):
           self.densityProfC[iz] = self.presProf[iz] / parameter.r / self.tempRealProf[iz]

        # Calculate density profile at the sides of layers [kg m^-3]
        self.densityProfS = [self.densityProfC[0] for x in range(self.nz+1)]
        for iz in xrange(1,self.nz):
           self.densityProfS[iz] = (self.densityProfC[iz] * self.dz +\
               self.densityProfC[iz-1] * self.dz) / (self.dz+self.dz)
        self.densityProfS[self.nz] = self.densityProfC[self.nz-1]

    def __repr__(self):
        return "RSM: obstacle ht={a}".format(
            a=self.height
            )

    def is_near_zero(self,num,eps=1e-16):
        return abs(float(num)) < eps

    def VDM(self,forc,rural,parameter,simTime,SolarRad,h_ublavg,var_sens):

        # Boundary condition for potential temperature near the ground [K]
        self.tempProf[0] = forc.temp

        # Calculate pressure profile [Pa]
        for iz in reversed(range(self.nz)[1:]):
           self.presProf[iz-1] = (math.pow(self.presProf[iz],parameter.r/parameter.cp) + \
               parameter.g/parameter.cp*(math.pow(forc.pres,parameter.r/parameter.cp)) * \
               (1./self.tempProf[iz] + 1./self.tempProf[iz-1]) * \
               0.5 * self.dz)**(1./(parameter.r/parameter.cp))

        # Calculate the real temperature profile [K]
        for iz in xrange(self.nz):
            self.tempRealProf[iz]= self.tempProf[iz] * \
            (self.presProf[iz]/forc.pres)**(parameter.r/parameter.cp)

        # Calculate the density profile [kg m^-3]
        for iz in xrange(self.nz):
           self.densityProfC[iz] = self.presProf[iz]/parameter.r/self.tempRealProf[iz]
        self.densityProfS[0] = self.densityProfC[0]

        # Calculate the density profile [kg m^-3]
        for iz in xrange(1,self.nz):
           self.densityProfS[iz] = (self.densityProfC[iz] * self.dz + \
               self.densityProfC[iz-1] * self.dz)/(self.dz + self.dz)
        self.densityProfS[self.nz] = self.densityProfC[self.nz-1]

        # Calculate diffusion coefficient (Kt) and friction velocity (ustarRur)
        self.Kt, ustarRur,self.lm_1,self.lm_2 = self.DiffusionCoefficient(self.dz, self.nz, forc.wind,parameter,self.windProf,SolarRad,var_sens)

        # Calculate heat source (gamma) in the rural area
        # Constant coefficient
        C_gamma = 10
        self.gamma = C_gamma*rural.flux/(parameter.cp*self.densityProfS[0]*h_ublavg)

        # Initialize diffusion coefficient
        Kt_initial = 0.01
        if int(simTime.secDay*simTime.timeSim*3600/simTime.timeMax) == simTime.dt:
            self.Kt = [Kt_initial for x in range(0,len(self.Kt))]

        # Solve vertical diffusion equation for potential temperature in the rural area
        self.tempProf = self.TransportEquationSimp(rural.T_ext, self.gamma, self.Kt)
        # Modify temperature based on calibration equation
        self.tempProf = [0.8865*self.tempProf[x]+33.1956 for x in range(0,len(self.tempProf))]
        # Calculate vapour pressure and saturation pressure
        # Calculate density at 2 m height
        rho_0 = forc.pres/(287*forc.temp)
        z_ref = 2
        # Calculate density profile using a constant density lapse rate of - 0.00133 [kg m-3 m-1]
        for ii in range(0,self.nz-1):
            rho = rho_0-0.000133*(self.z[self.nz-1]-z_ref)
            Treal = (self.tempProf[ii]*(rho*287/forc.pres)**0.286)**(1/0.714)

            C8 = -5.8002206e3
            C9 = 1.3914993
            C10 = -4.8640239e-2
            C11 = 4.1764768e-5
            C12 = -1.4452093e-8
            C13 = 6.5459673
            PWS = numpy.exp(C8 / Treal + C9 + C10 * Treal + C11 * pow(Treal, 2) + C12 * pow(Treal, 3) + C13 * numpy.log(Treal))
            # Vapor pressure
            PW = forc.rHum * PWS / 100.0
            # Specific humidity [gr kg-1]
            W = 0.62198 * PW / (forc.pres - PW)
            # Air pressure [Pa] = density [kg m-3] * gas constant [J kg-1 K-1] * temperature [K]
            P_air = rho*287*Treal
            # Vapor pressure [kPa]
            self.Pv = W*P_air/0.622/1000
            # Saturation pressure [kPa]
            self.Psat = 6.1094*numpy.exp(17.625*(Treal-273.15)/((Treal-273.15)+243.04))

            # Print warning alert if vapour pressure exceeds saturation pressure
            #if self.Pv > self.Psat:
                #print('WARNING: P_v is greater than P_sat')


        # Calculate wind speed profile in the rural area
        for iz in xrange(self.nz):
            self.windProf[iz] = ustarRur/parameter.vk*\
                math.log((2*(iz+1)-self.disp)/self.z0r)

        # Calculate average pressure
        self.ublPres = 0.
        for iz in xrange(self.nz):
            self.ublPres = self.ublPres + \
                self.presProf[iz]*self.dz/(self.z[self.nz-1]+self.dz/2.)

    # Function to calculate diffusion coefficient (Kt) and friction velocity (ustar)
    def DiffusionCoefficient(self,dz,nz,uref,parameter,windProf,SolarRad,var_sens):

        # Initialize diffusion coefficient
        Kt = [0 for x in xrange(nz+1)]

        # Calculate friction velocity (Aliabadi et al, 2018)
        ustar = 0.07 * uref + 0.12

        # Mixing length Model
        # Initialize mixing length
        lm = [0 for x in xrange(nz)]
        lm_term1 = [0 for x in xrange(nz)]
        lm_term2 = [0 for x in xrange(nz)]
        # Correction factor for friction velocity under unstable condition
        Custarunstable = 50
        # Scaling correction factor under unstable atmospheric condition
        Clunstable = 1    # 2
        # Correction factor for friction velocity under stable condition
        Custarstable = 50
        # Scaling correction factor under stable atmospheric condition
        Clstable = 0.9    # 1.5

        # Calculate mixing length (lm) for stable and unstable conditions based on direct solar radiation in the rural area
        # Unstable conditions
        if SolarRad > 0.01:

            for iz in xrange(nz):
                lm[iz] = Clunstable * parameter.vk * (dz*iz) * Custarunstable * ustar / (parameter.vk * (dz*iz) + Custarunstable * ustar)
                lm_term1[iz] = (Clunstable)*((parameter.vk * (dz*iz)))
                lm_term2[iz] = (Clunstable)*((Custarunstable * ustar))

        # Stable and neutral conditions
        else:

            for iz in xrange(nz):
                lm[iz] = Clstable * parameter.vk * (dz*iz) * Custarstable * ustar / (parameter.vk * (dz*iz) + Custarstable * ustar)
                lm_term1[iz] = (Clstable)*((parameter.vk * (dz*iz)))
                lm_term2[iz] = (Clstable)*((Custarstable * ustar))

        # Calculate diffusion coefficient
        # Kt = (velocity gradient)*(mixing length)^2
        for iz in xrange(nz-1):
            dSdz = (windProf[iz+1]-windProf[iz])/dz
            Kt[iz] = lm[iz]**2 * dSdz
        Kt[nz] = Kt[nz - 1]

        return Kt, ustar, lm_term1, lm_term2

    def TransportEquationSimp(self,Ts,gamma,Kt):

        # Define under-relaxation factor
        alpha = 0.1
        # Define maximum iteration number
        MaxIter = 100
        # Define relative error
        Err = 0.0001

        # Define and initialize temperature [K]
        Tinitial = 300
        Tmean = numpy.zeros((self.nz,1))
        Tmean[:] = Tinitial

        # Define unknown vector X, coefficient matrix A, and vector B, in AX=B
        x = numpy.zeros((self.nz, 1))
        b = numpy.zeros((self.nz, 1))
        a = numpy.zeros((self.nz, self.nz))

        # Initialize solution vector X
        # This is a short syntax for for loop
        x[0:self.nz] = Tmean[0:self.nz]

        for iter in range(1, MaxIter):

            # Heat equations
            # i=0
            a[0][0] = 1
            b[0] = Ts
            # i=1 to N-1
            for i in range(1, self.nz-1):
                # Calculate derivatives by finite differences for the current i index
                # Remember to shift indices by N+1 if needed

                k0 = Kt[i]
                k1 = (Kt[i + 1] - Kt[i - 1]) / (2 * self.dz)
                # Set constants necessary to build the coefficient matrix
                e1 = k1
                e2 = k0
                eb = -(-gamma)

                # Set the coefficient matrix and the B vector
                a[i][i - 1] = -e1 / (2 * self.dz) + e2 / (self.dz ** 2)
                a[i][i] = -2 * e2 / (self.dz ** 2)
                a[i][i + 1] = e1 / (2 * self.dz) + e2 / (self.dz ** 2)
                b[i] = eb

            # i=N
            a[self.nz-1][self.nz-2] = 1
            a[self.nz-1][self.nz-1] = -1
            b[self.nz-1] = 0

            xnew = numpy.linalg.solve(a, b)

            # Calculate maximum norm errors for all solutions
            ErrTmean = numpy.max(numpy.abs(numpy.divide(xnew[1:self.nz] - x[1:self.nz], x[1:self.nz])))

            if ErrTmean < Err :
                #print('Solutions converged at iteration: ', iter)
                # Exit the loop
                break

            # Update solution
            x[:] = x[:] + alpha * (xnew[:] - x[:])

        # Assign the X vector to the original T vector
        Tmean[0:self.nz] = x[0:self.nz]

        return Tmean









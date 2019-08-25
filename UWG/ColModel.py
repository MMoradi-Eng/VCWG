import numpy
import math
import matplotlib.pyplot as plt

from buildingCol import BuildingCol
from cdturb import CdTurb
#from cdturbWF import CdTurbWF
from shear import Shear
from buoyancy import Buoyancy
from diff import Diff
from invert import Invert

"""
Column Model for momentum, turbulent kinetic energy, temperature, and specific humidity in the urban environment
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Alberto Martilli, Scott Krayenhoff, and Negin Nazarian
"""

class ColModel:

    def __init__(self,WallTemp,RoofTemp,RoadTemp,ForcTemp,ForcHum,ForcWind,ForcWindDir,vx,vy,tke,th,qn,nz,Ck,dlk,nz_u,dz,dt,vol,
                 VegCoverage,lambdap,lambdaf,hmean,Cp,th0,Cdrag,pb,ss,prandtl,schmidt,g,Ceps,dls,sf,rho,h_LAD,f_LAD,L_abs,S_t,L_t,it,var_sens,SensHt_HVAC,theta_can,HVAC_street_frac,HVAC_atm_frac):
        self.WallTemp = WallTemp         # Wall temperature [K]
        self.RoofTemp = RoofTemp         # Roof temperature [K]
        self.RoadTemp = RoadTemp         # Road Temperature [K]
        self.ForcTemp = ForcTemp         # Forced temperature [K] at the top of the domain
                                         # (~ five times of building height) and will be used as boundary condition
        self.ForcWind = ForcWind         # Forced horizontal wind speed [m s^-1] at the top of the domain
                                         # (~ five times of building height) and will be used as boundary condition
        self.ForcWindDir = ForcWindDir   # Forced wind direction [deg] at the top of the domain
                                         # (~ five times of building height) and will be used as boundary condition
                                         # It is the angle between wind direction blowing from rural area and
                                         # geographical north, in the counter direction
        self.vx = vx                     # x component of horizontal wind speed [m s^-1]
        self.vy = vy                     # y component of horizontal wind speed [m s^-1]
        self.tke = tke                   # Turbulent kinetic energy [m^2 s^-2]
        self.th = th                     # Potential temperature [K]
        self.qn = qn                     # Specific humidity [kgv kga^-1]
        self.nz = nz                     # Number of grid points in vertical column
        self.Ck = Ck                     # Coefficient used in the equation of diffusion coefficient (kappa)
        self.dlk = dlk                   # Mixing length [m]
        self.nz_u = nz_u                 # Number of grid points within the canyon
        self.dz = dz                     # Grid resolution [m]
        self.dt = dt                     # Time step [s]
        self.vol = vol                   # Fraction of air in each cell
        self.VegCoverage = VegCoverage   # Fraction of the urban ground covered in grass/shrubs
        self.lambdap = lambdap           # Plan area fraction of buildings
        self.lambdaf = lambdaf           # Ratio of wall area facing ambient wind to plan area
        self.hmean = hmean               # Average building height [m]
        self.Cp = Cp                     # Heat capacity of dry air [J kg^-1 K^-1]
        self.th0 = th0                   # Reference potential temperature [K]
        self.Cdrag = Cdrag               # Drag coefficient due to buildings (sectional drag coefficient)
        self.pb = pb                     # Probability distribution of building height (assumed to be one
                                         # within the canyon and zero above the canyon)
        self.ss = ss                     # Probability that a building has a height equal to z (assumed to be
                                         # one at average building height and zero the other heights)
        self.prandtl = prandtl           # Turbulent Prandtl number
        self.schmidt = schmidt           # Turbulent Schmidt number
        self.g = g                       # Gravitational acceleration [m s^-2]
        self.Ceps = Ceps                 # Coefficient for the destruction of turbulent dissipation rate
        self.dls = dls                   # Length scale for turbulent dissipation [m]
        self.sf = sf                     # fraction of air at the interface between cells
        self.rho = rho                   # density profile [kg m^-3]
        self.ForcHum = ForcHum           # Forced specific humidity [kgv kga^-1] at the top of the domain
                                         # (~ five times of building height) and will be used as boundary condition
        self.h_LAD = h_LAD               # Leaf area density (LAD) function (z location) [m]
        self.f_LAD = f_LAD               # Leaf area density (LAD) function
        self.L_abs = L_abs               # Net long wave radiation [W m^-2] (for later use)
        self.S_t = S_t                   # Net short wave radiation absorbed by trees [W m^-2]
        self.L_t = L_t                   # Net long wave radiation absorbed by trees [W m^-2]
        self.it = it                     # time step in column (1-D) model
        self.var_sens = var_sens         # any variable for sensitivity analysis @@@@@ sensitivity analysis @@@@@
        self.SensHt_HVAC = SensHt_HVAC   # HVAC waste heat flux per unit building footprint area [W m^-2]
        self.theta_can = theta_can
        self.HVAC_street_frac = HVAC_street_frac       # Fraction of Sensible waste heat from building released into the atmosphere at street level
        self.HVAC_atm_frac = HVAC_atm_frac             # Fraction of sensible waste heat from building released into the atmosphere

    def ColumnModelCal(self):

        # Calculate angle between wind direction and canyon orientation (theta_S) [deg]
        theta_S = (360+abs(self.theta_can-self.ForcWindDir))%90



        # Road roughness
        z0g = 0.05
        # Roof roughness
        z0r = 0.15
        # gas constant dry air [J kg^-1 K^-1]
        r = 287.04
        rcp = r / self.Cp
        # Define explicit and implicit parts of source and sink terms
        srex_vx = numpy.zeros(self.nz)       # Explicit part of x component of horizontal wind speed [m s^-2]
        srim_vx = numpy.zeros(self.nz)       # Implicit part of x component of horizontal wind speed [s^-1]
        srex_vy = numpy.zeros(self.nz)       # Explicit part of y component of horizontal wind speed [m s^-2]
        srim_vy = numpy.zeros(self.nz)       # Implicit part of y component of horizontal wind speed [s^-1]
        srex_tke = numpy.zeros(self.nz)      # Explicit part of turbulent kinetic energy [m^2 s^-3]
        srim_tke = numpy.zeros(self.nz)      # Implicit part of turbulent kinetic energy [s^-1]
        srex_th = numpy.zeros(self.nz)       # Explicit part of potential temperature [K s^-1]
        srim_th = numpy.zeros(self.nz)       # Implicit part of potential temperature [s^-1]
        srex_qn = numpy.zeros(self.nz)       # Explicit part of specific humidity [K s^-1] ?????
        srim_qn = numpy.zeros(self.nz)       # Implicit part of specific humidity [s^-1] ?????

        srex_th_veg = numpy.zeros(self.nz)   # Explicit part of potential temperature caused by vegetation
        srex_qn_veg = numpy.zeros(self.nz)   # Explicit part of specific humidity caused by vegetation
        Tveg = numpy.zeros(self.nz)

        # Apply boundary conditions at the top of the domain using vertical diffusion model(VDM) outputs
        self.th[self.nz-1] = self.ForcTemp
        self.qn[self.nz-1] = self.ForcHum
        self.vx[0] = 0.001
        self.vy[0] = 0.001
        self.tke[0] = 0.00001

        # Calculate bulk Richardson number (Ri_b):
        # Ri_b = (g*H/((Uroof - Ustreet)^2+(Vroof - Vstreet)^2))*(Troof - Tstreet)/Tavg (equation 6, Aliabadi et al., 2018)
        delU = ((self.vx[0]-self.vx[self.nz_u+1])**2+(self.vy[0]-self.vy[self.nz_u+1])**2)
        # Denominator of the fraction must not be zero. So, a minimum value for denominator is considered
        delU = max(delU,0.1)
        #For calculation of Rib, option 1: surface temperature difference
        #delT = self.RoofTemp-self.RoadTemp
        #For calculation of Rib, option 2: air temperature difference
        delT = self.th[self.nz_u+1]-self.th[1]
        Ri_b =  ((self.g*self.hmean)/delU)*(delT/numpy.mean(self.th[0:self.nz_u]))
        # Calculate turbulent diffusion coefficient (Km) [m^2 s^-1]
        TurbDiff = CdTurb(self.nz, self.Ck, self.tke, self.dlk,self.it,Ri_b,self.var_sens)
        Km = TurbDiff.TurbCoeff()

        # Road surface temperature [K]
        ptg = self.RoadTemp
        # Wall surface temperature [K]
        ptw = self.WallTemp
        # Roof surface temperature [K]
        ptr = self.RoofTemp

        # Call "BuildingCol" to calculate sink and source terms in momentum, temperature and turbulent kinetic energy (TKE)
        #  equations which are caused by building
        BuildingCoef = BuildingCol(self.nz, self.dz, self.dt, self.vol, (1-self.VegCoverage), self.lambdap, self.lambdaf,
                                   self.hmean, self.Ck, self.Cp, self.th0, self.vx, self.vy, self.th, self.Cdrag,
                                   ptg,ptr, ptw, self.rho,self.nz_u, self.pb, self.ss,self.g,z0g,z0r,self.SensHt_HVAC,self.HVAC_street_frac,self.HVAC_atm_frac)

        # Calculate shear production [m^2 s^-3] in TKE equation. (Term II of equation 5.2, Krayenhoff 2014, PhD thesis)
        Shear_Source = Shear(self.nz, self.dz, self.vx, self.vy, Km)
        sh = Shear_Source.ShearProd()

        # Calculate buoyant production [m^2 s^-3] in TKE equation. (Term IX of equation 5.2, Krayenhoff 2014, PhD thesis)
        Buoyancy_Source = Buoyancy(self.nz, self.dz, self.th, Km, self.th0, self.prandtl, self.g)
        bu = Buoyancy_Source.BuoProd()

        # Calculate dissipation (td) [s^-1] in TKE equation. (Term VI of equation 5.2, Krayenhoff 2014, PhD thesis)
        # parameterization of dissipation is based on Nazarian's code. (https://github.com/nenazarian/MLUCM/blob/master/Column_Model/column_lkPro.f90)
        td = numpy.zeros(self.nz)
        for i in range(0, self.nz):
            if self.dls[i] != 0:
                td[i] = -self.Ceps*(math.sqrt(self.tke[i]))/self.dls[i]
            else:
                td[i] = 0
            sh[i] = sh[i]*self.sf[i]
            bu[i] = bu[i]*self.sf[i]

        # Return sink and source terms caused by buildings
        srex_vx_h, srex_vy_h, srex_tke_h, srex_th_h, srim_vx_v, srim_vy_v, srex_tke_v, srim_th_v, srex_th_v, sff,swf, ustarCol = BuildingCoef.BuildingDrag()

        # Friction velocity (Aliabadi et al, 2018)
        ustar = 0.07 * self.ForcWind + 0.12
        # Calculate pressure gradient
        C_dpdx = 5.4
        dpdx = C_dpdx*self.rho[self.nz-1]*(ustar**2)*math.cos(math.radians(theta_S))/(self.dz*self.nz)
        dpdy = C_dpdx*self.rho[self.nz-1]*(ustar**2)*math.sin(math.radians(theta_S))/(self.dz*self.nz)

        # Latent heat of vaporization [J kg^-1]
        latent = 2.45e+06
        # Latent heat of vaporization [J mol^-1](Campbell and Norman,1998)
        latent2 = 44100
        # The average surface and boundary-layer conductance for humidity for the whole leaf
        gvs = 0.330
        # Set leaf dimension of trees
        leaf_width = 0.05
        leaf_dim = 0.72 * leaf_width
        # Air pressure [Pa]
        pr = 101300
        # Total neighbourhood foliage clumping [non dimensional]
        omega = 1
        # Molar heat capacity [J mol^-1 K^-1](Campbell and Norman, 1998)
        cp_mol = 29.3
        # Drag coefficient for vegetation foliage
        cdv = 0.2
        omega_drag = 0.34

        # Calculate source and sink terms caused by trees and then calculate total source and sink terms
        for i in range(0, self.nz):

            # source/sink terms of specific humidity
            wind = numpy.sqrt(self.vx[i] ** 2 + self.vy[i] ** 2)
            # Boundary-layer conductance for vapor (p. 101 Campbell and Norman, 1998)
            gva = 1.4 * 0.147 * numpy.sqrt(wind / leaf_dim)
            # Overall vapour conductance for leaves [mol m^-2 s^-1] (equation 14.2, Campbell and Norman, 1998):
            gv = gvs * gva / (gvs + gva)
            # Conductance for heat [mol m^-2 s^-1]
            gHa = 1.4 * 0.135 * numpy.sqrt(wind / leaf_dim)
            # Since a leaf has two sides in parallel, gHa should be multiplied by 2
            gHa = gHa * 2
            # Convert potential air temperature to real temperature [K]
            # potential temperature = real temperature * (P0/P)^(R/cp)
            tair = self.th[i] / (pr / 1.e+5) ** (-rcp)
            # Convert absolute humidity to vapour pressure [Pa]
            eair = self.qn[i] * pr / 0.622
            # Saturation vapor pressure [Pa] (equation 7.5.2d, Stull 1988)
            es = 611.2 * numpy.exp(17.67 * (tair - 273.16) / (tair - 29.66))
            D = es - eair
            desdT = 0.622 * latent * es / r / (tair) ** 2
            s = desdT / pr
            # Calculate terms in transport equations caused by trees. "wt" is term in temperature equation adn "wt_drag"
            # is term in TKE and momentum equations. It is assumed there is no vegetation above average building height
            if self.dz * i > max(self.h_LAD):
                wt = 0         # [m^2 m^-3]
                wt_drag = 0    # [m^2 m^-3]
            else:
                wt = self.f_LAD(self.dz * i) * omega * (1 - self.lambdap) / self.vol[i]             # [m^2 m^-3]
                wt_drag = self.f_LAD(self.dz * i) * omega_drag * (1. - self.lambdap) / self.vol[i]  # [m^2 m^-3]

            # Stefan-Boltzmann constant [W m^-2 K^-4]
            sigma = 5.67e-8
            gam = 6.66e-4
            # Emissivity of leaves surface
            emveg = 0.95
            # Total fraction scattered by leaves: reflected & transmitted
            albv_u = 0.5
            fact = 1
            # Total radiation absorbed by leaves [W m^-2]
            Rabs = (1-albv_u)*self.S_t+self.L_t*emveg
            gr = 4 * emveg * sigma * tair ** 3 / cp_mol
            gr = gr * 2. * omega * fact
            sides = 2. * omega * fact
            gHr = gHa + gr
            gamst = gam * gHr / gv
            # Calculate temperature of vegetation [K]
            tveg_tmp = tair+gamst/(s+gamst)*((Rabs-sides*emveg*sigma*(tair**4))/gHr/cp_mol-D/pr/gamst)
            Tveg[i] = tveg_tmp

            # Calculate terms in temperature and humidity equations caused by trees.
            if self.dz * i > max(self.h_LAD):
                srex_th_veg[i] = 0
                srex_qn_veg[i] = 0
            else:
                srex_th_veg[i] = cp_mol*gHa*tveg_tmp*wt/self.Cp/self.rho[i]
                srex_qn_veg[i] = (latent2*gv*(s*(tveg_tmp-tair)+es/pr))*wt/self.rho[i]/latent

            # Calculate total explicit terms
            # Explicit term in x momentum equation [m s^-2] = fluxes from horizontal surfaces + pressure gradient
            # pressure gradient is zero, because boundary conditions are forced by vertical diffusion model
            srex_vx[i] = srex_vx_h[i]+dpdx

            # Explicit term in y momentum equation [m s^-2] = fluxes from horizontal surfaces + pressure gradient
            # pressure gradient is zero, because boundary conditions are forced by vertical diffusion model
            srex_vy[i] = srex_vy_h[i]+dpdy

            # Explicit term in TKE equation [m^2 s^-3] = terms from urban horizontal surfaces [??????] +
            # terms from walls [m^2 s^-3] + shear production [m^2 s^-3] + buoyant production [m^2 s^-3] +
            # term caused by vegetation [m^2 s^-3]
            srex_tke[i] = srex_tke_h[i] + srex_tke_v[i] + sh[i] + bu[i] + cdv*wind**3.*wt_drag

            # Explicit term in temperature equation [K s^-1] = term from urban horizontal surfaces [K s^-1] +
            # term from walls [K s^-1] + term caused by vegetation [K s^-1]
            srex_th[i] = srex_th_h[i] + srex_th_v[i] + srex_th_veg[i] #+ 4*rho_abs*kbs*(1-self.lambdap)*self.L_abs/self.rho/self.Cp/self.vol[i]

            # Explicit term in humidity equation [K s^-1] = term caused by latent heat from vegetation [K s^-1]
            srex_qn[i] = srex_qn[i] + srex_qn_veg[i]

            # Calculate total Implicit terms
            # Implicit term in x momentum equation [s^-1] = term from walls [s^-1] - term caused by vegetation [s^-1]
            srim_vx[i] = srim_vx_v[i]-cdv*wind*wt_drag

            # Implicit term in y momentum equation [s^-1] = term from walls [s^-1] - term caused by vegetation [s^-1]
            srim_vy[i] = srim_vy_v[i]-cdv*wind*wt_drag

            # Implicit term in TKE equation [s^-1] = dissipation [s^-1] - term caused by vegetation [s^-1]
            srim_tke[i] = td[i]-6.5*cdv*wind*wt_drag

            # Implicit term in temperature equation [s^-1] = term from wall [s^-1] - term caused by vegetation [s^-1]
            srim_th[i] = srim_th_v[i]-cp_mol*gHa*wt/self.Cp/self.rho[i]

            # Implicit term in humidity equation [s^-1] = term caused by latent heat from vegetation [s^-1]
            srim_qn[i] = srim_qn[i]-latent2*gv*(pr/0.622)/pr*wt/self.rho[i]/latent


        # Solve transport equations
        # Set type of boundary conditions (B.Cs):
        # Neumann boundary condition (Flux): iz = 1
        # Dirichlet boundary condition (Constant value): iz = 2
        # Sol.Solver(B.C. at the bottom of domain)
        Sol = Diff(self.nz, self.dt, self.sf, self.vol, self.dz, self.rho)
        # Solve x component of momentum equation
        A_vx = Sol.Solver21(2, 1, self.vx, srim_vx, srex_vx,Km)[0]
        RHS_vx = Sol.Solver21(2, 1, self.vx, srim_vx, srex_vx,Km)[1]
        Inv_vx = Invert(self.nz, A_vx, RHS_vx)
        self.vx = Inv_vx.Output()
        # Solve y component of momentum equation
        A_vy = Sol.Solver21(2, 1, self.vy, srim_vy, srex_vy,Km)[0]
        RHS_vy = Sol.Solver21(2, 1, self.vy, srim_vy, srex_vy,Km)[1]
        Inv_vy = Invert(self.nz, A_vy, RHS_vy)
        self.vy = Inv_vy.Output()
        # Solve TKE equation
        A_tke = Sol.Solver21(2, 1, self.tke, srim_tke, srex_tke,Km)[0]
        RHS_tke = Sol.Solver21(2, 1, self.tke, srim_tke, srex_tke,Km)[1]
        Inv_tke = Invert(self.nz, A_tke, RHS_tke)
        self.tke = Inv_tke.Output()
        # Solve temperature equation
        A_th = Sol.Solver12(1, 2, self.th, srim_th, srex_th,Km/self.prandtl)[0]
        RHS_th = Sol.Solver12(1, 2, self.th, srim_th, srex_th,Km/self.prandtl)[1]
        Inv_th = Invert(self.nz, A_th, RHS_th)
        self.th = Inv_th.Output()
        # Solve specific humidity equation
        A_qn = Sol.Solver12(1, 2, self.qn, srim_qn, srex_qn,Km/self.schmidt)[0]
        RHS_qn = Sol.Solver12(1, 2, self.qn, srim_qn, srex_qn,Km/self.schmidt)[1]
        Inv_qn = Invert(self.nz, A_qn, RHS_qn)
        self.qn = Inv_qn.Output()

        # Set a minimum value for kinetic energy which avoid trapping of heat at street level
        for i in range(0, self.nz):
            if self.tke[i] < 1e-3:
               self.tke[i] = 1e-3

        return self.vx,self.vy,self.tke,self.th,self.qn, ustarCol,Km,tveg_tmp,Ri_b,Tveg
import numpy
import math

"""
Calculate sink and source terms associated with the presence of buildings in the 1D model for momentum, heat, and TKE  
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Alberto Martilli, Scott Krayenhoff, and Negin Nazarian
"""

# explicit and implicit terms for the building
class BuildingCol:

    # number of street direction (assumed to be one)
    nd = 1

    def __init__(self,nz,dz,dt,vol,roadfrac,lambdap,lambdaf,hmean,Ck,Cp,th0,vx,vy,th,Cdrag,ptg,ptr,ptw,rho,nz_u,pb,ss,g,z0g,z0r,SensHt_HVAC,HVAC_street_frac,HVAC_atm_frac):
        self.nz = nz                     # Number of grid points in vertical column
        self.dz = dz                     # Grid resolution [m]
        self.dt = dt                     # Time step [s]
        self.vol = vol                   # Fraction of air to total volume in each urban unit cell
        self.roadfrac = roadfrac         # It is not given in Scott's code (assumption: roadfrac = 1)
        self.lambdap = lambdap           # Plan area fraction of buildings
        self.lambdaf = lambdaf           # Ratio of wall area facing ambient wind to plan area
        self.hmean = hmean               # Average building height [m]
        self.Ck = Ck                     # Coefficient used in the equation of diffusion coefficient (kappa)
        self.Cp = Cp                     # Heat capacity of dry air [J kg^-1 K^-1]
        self.th0 = th0                   # Reference potential temperature [K]
        self.vx = vx                     # x component of horizontal wind speed [m s^-1]
        self.vy = vy                     # y component of horizontal wind speed [m s^-1]
        self.th = th                     # Potential temperature [K]
        self.Cdrag = Cdrag               # Drag coefficient due to buildings (sectional drag coefficient)
        self.ptg = ptg                   # Road surface temperature (ptg) [K].
        self.ptr = ptr                   # Roof surface temperature (ptg) [K]
        self.ptw = ptw                   # Wall surface temperature (ptg) [K]
        self.rho = rho                   # density profile [kg m^-3]
        self.nz_u = nz_u                 # Number of grid points within the canyon
        self.pb = pb                     # Probability distribution of building height (assumed to be one
                                         # within the canyon and zero above the canyon)
        self.ss = ss                     # Probability that a building has a height equal to z (assumed to be
                                         # one at average building height and zero the other heights)
        self.g = g                       # Gravitational acceleration [m s^-2]
        self.z0g = z0g                   # Road roughness [m]
        self.z0r = z0r                   # Roof roughness [m]
        self.SensHt_HVAC = SensHt_HVAC   # Sensible waste heat from building
        self.HVAC_street_frac = HVAC_street_frac       # Fraction of Sensible waste heat from building released into the atmosphere at street level
        self.HVAC_atm_frac = HVAC_atm_frac             # Fraction of sensible waste heat from building released into the atmosphere

    def Flux_Flat(self,z0,vx,vy,th,th0,pts):

        Utot = (vx**2+vy**2)**0.5
        zz = self.dz/2

        Utot = max(Utot,0.01)

        # Compute bulk Richardson number
        Ri = 2 * self.g * zz * (th - pts) / ((th + pts) * (Utot ** 2))
        # Calculation from Louis, 1979 (eq. 11 and 12)
        b = 9.4
        cm = 7.4
        ch = 5.3
        R = 0.74
        a = self.Ck/math.log(zz/z0)

        if Ri > 0:
            fm = 1/((1+0.5*b*Ri)**2)
            fh = fm
        else:
            c = b*cm*a*a*(zz/z0)**0.5
            fm = 1-b*Ri/(1+c*(-Ri)**0.5)
            c = c*ch/cm
            fh = 1-b*Ri/(1+c*(-Ri)**0.5)

        fbuw = -(a**2)*(Utot**2)*fm
        fbpt = -(a**2)*Utot*(th-pts)*fh/R

        ustar = (-fbuw)**0.5
        tstar = -fbpt/ustar
        ## 'a1', 'buu' and 'muu' are skipped (L. 255-257,building.f90)

        # x component momentum flux from horizontal surfaces [m^2 s^-2]
        uhb = -(ustar**2)*vx/Utot
        # y component momentum flux from horizontal surfaces [m^2 s^-2]
        vhb = -(ustar**2)*vy/Utot
        # Heat flux from horizontal surfaces [K m s^-1]
        thb = -ustar*tstar
        # Turbulent flux of TKE from horizontal surfaces [m^2 s^-3]
        ehb = -(self.g/th0)*ustar*tstar

        return uhb,vhb,ehb,thb,ustar

    def Flux_Wall(self,vx,vy,th,Cdrag,ptw,rho):

        vett = (vx**2+vy**2)**0.5
        # Implicit term of x component momentum flux from vertical surfaces [m s^-1]
        uva = -Cdrag*vett
        # Implicit term of y component momentum flux from vertical surfaces [m s^-1]
        vva = -Cdrag*vett
        # Explicit term of x component momentum flux from vertical surfaces [????]
        uvb = 0
        # Explicit term of y component momentum flux from vertical surfaces [????]
        vvb = 0

        # Calculation for S_theta_wall in eq. 5.5 (Krayenhoff, PhD thesis)
        # Convective heat transfer coefficient [W K^-1 m^-2]
        hc = 5.678*(1.09+0.23*(vett/0.3048))
        # Using energy balance for a control volume inside the urban unit, the convective heat transfer coefficient should be limited
        # hc must be less than (rho * cp / dt) * [(1-lambdap) * Hmean / (4 * lambdaf * dz)] -> Check Mohsen Moradi's notebook Dec. 21, 2018
        # In practice [(1-lambdap) * Hmean / (4 * lambdaf * dz)] is in the order of 1 so Scott and Negin should not feel too bac!
        if hc > ((rho*self.Cp/self.dt)*((1-self.lambdap)*self.hmean)/(4*self.lambdaf*self.dz)):
            hc = (rho*self.Cp/self.dt)*((1-self.lambdap)*self.hmean)/(4*self.lambdaf*self.dz)
        # Term in energy equation [K m s^-1]
        tvb = (hc/(rho*self.Cp))*(ptw-th)
        tva = 0

        evb = Cdrag*(abs(vett)**3)

        return uva,vva, uvb, vvb, tva, tvb, evb

    def BuildingDrag(self):
        # Define momentum and heat fluxes from horizontal surfaces
        uhb = numpy.zeros(self.nz_u+1)    # Term in momentum equation [m^2 s^-2]
        vhb = numpy.zeros(self.nz_u+1)    # Term in momentum equation [m^2 s^-2]
        ehb = numpy.zeros(self.nz_u+1)    # Term in turbulent kinetic energy equation [m^2 s^-3]
        thb = numpy.zeros(self.nz_u+1)    # Term in energy equation [K m s^-1]
        # Define momentum and heat fluxes on vertical surfaces
        uva = numpy.zeros(self.nz_u)      # Term in momentum equation [m s^-1]
        vva = numpy.zeros(self.nz_u)      # Term in momentum equation [m s^-1]
        uvb = numpy.zeros(self.nz_u)      # Term in momentum equation [s^-1]
        vvb = numpy.zeros(self.nz_u)      # Term in momentum equation [s^-1]
        tva = numpy.zeros(self.nz_u)      # Term in energy equation [m s^-1]
        tvb = numpy.zeros(self.nz_u)      # Term in energy equation [K m s^-1]
        evb = numpy.zeros(self.nz_u)      # Term in energy equation [m^3 s^-3]

        # Define friction velocity [m s^-1](for record keeping only)
        ustarCol = numpy.zeros(self.nz_u+1)

        # Define explicit and implicit parts of source and sink terms due to building
        srex_vx_h = numpy.zeros(self.nz)  # Term in momentum equation
        srex_vy_h = numpy.zeros(self.nz)  # Term in momentum equation
        srex_tke_h = numpy.zeros(self.nz) # Term in turbulent kinetic energy equation
        srex_th_h = numpy.zeros(self.nz)  # Term in energy equation
        srim_vx_v = numpy.zeros(self.nz)  # Term in momentum equation
        srim_vy_v = numpy.zeros(self.nz)  # Term in momentum equation
        srex_tke_v = numpy.zeros(self.nz) # Term in turbulent kinetic energy equation
        srim_th_v = numpy.zeros(self.nz)  # Term in energy equation
        srex_th_v = numpy.zeros(self.nz)  # Term in energy equation

        # Define surface heat fluxes [W m^-2](for record keeping only)
        sfr = numpy.zeros(self.nz_u+1)
        sfw = numpy.zeros(self.nz)

        Flux_CI = BuildingCol(self.nz,self.dz,self.dt,self.vol,self.roadfrac,self.lambdap,self.lambdaf,self.hmean,self.Ck,self.Cp,self.th0,self.vx,self.vy,self.th,self.Cdrag,self.ptg,self.ptr,self.ptw,self.rho,self.nz_u,self.pb,self.ss,self.g,self.z0g,self.z0r,self.SensHt_HVAC,self.HVAC_street_frac,self.HVAC_atm_frac)

        # Calculation of ground flux (for simulation w/o probability, this part should be the same as next part???)

        FluxFlatG = Flux_CI.Flux_Flat(self.z0g,self.vx[0],self.vy[0],self.th[0],self.th0[0],self.ptg)
        uhb[0] = FluxFlatG[0]
        vhb[0] = FluxFlatG[1]
        ehb[0] = FluxFlatG[2]
        thb[0] = FluxFlatG[3]
        ustarCol[0] = FluxFlatG[4]

        # Term in momentum equation [m s^-2]
        srex_vx_h[0] = (uhb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in momentum equation [m s^-2]
        srex_vy_h[0] = (vhb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in turbulent kinetic energy equation [m s^-3] (???? should it be [m^2 s^-3]?)
        srex_tke_h[0] = (ehb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Term in energy equation [K s^-1]
        # Option a: do not consider HVAC waste heat as source/sink term
        srex_th_h[0] = (thb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac
        # Option b: consider HVAC waste heat as source/sink term at the street
        # Kinematic heat flux of the HVAC waste heat should be scaled according to building footprint area and urban area in each urban unit
        # HVAC waste heat flux building x Building footprint area = HVAC waste heat flux urban x urban area
        # HVAC waste heat flux urban = (Building footprint area / urban area) x HVAC waste heat flux building
        # HVAC waste heat flux urban = (lambdap / (1-lambdap)) x HVAC waste heat flux building
        srex_th_h[0] = (thb[0]/self.nd)/self.dz*(1-self.lambdap)/self.vol[0]*self.roadfrac + \
                       self.HVAC_atm_frac*self.HVAC_street_frac*(self.SensHt_HVAC/(self.rho[0]*self.Cp)/self.dz)*self.lambdap/(1-self.lambdap)

        # Calculation of fluxes of other points
        for i in range(0,self.nz_u+1):
            # At roof level for simple and non-probabilistic canyon
            if self.ss[i] > 0:
                FluxFlatR = Flux_CI.Flux_Flat(self.z0r, self.vx[i],self.vy[i], self.th[i], self.th0[i],self.ptr)
                uhb[i] = FluxFlatR[0]
                vhb[i] = FluxFlatR[1]
                ehb[i] = FluxFlatR[2]
                thb[i] = FluxFlatR[3]
                ustarCol[i] = FluxFlatR[4]
            else:
                # On walls for simple and non-probabilistic canyon
                uhb[i] = 0
                vhb[i] = 0
                ehb[i] = 0
                thb[i] = 0

            # At roof level for simple and non-probabilistic canyon
            # Term in momentum equation [m s^-2]
            srex_vx_h[i] += (uhb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in momentum equation [m s^-2]
            srex_vy_h[i] += (vhb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in turbulent kinetic energy equation [m s^-3] (???? should it be [m^2 s^-3]?)
            srex_tke_h[i] += (ehb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Term in energy equation [K s^-1]
            # Option a: do not consider HVAC waste heat as source/sink term
            srex_th_h[i] += (thb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz)
            # Option b: consider HVAC waste heat as source/sink term at the street
            # Kinematic heat flux of the HVAC waste heat should be scaled according to building footprint area and urban area in each urban unit
            # HVAC waste heat flux building x Building footprint area = HVAC waste heat flux urban x urban area
            # HVAC waste heat flux urban = (Building footprint area / urban area) x HVAC waste heat flux building
            # HVAC waste heat flux urban = (lambdap / (1-lambdap)) x HVAC waste heat flux building
            srex_th_h[i] = (thb[i]/self.nd)*(self.ss[i]*self.lambdap/self.vol[i]/self.dz) + \
                           self.HVAC_atm_frac*(1-self.HVAC_street_frac)*(self.SensHt_HVAC /(self.rho[i]*self.Cp)/self.dz)*self.lambdap/(1 - self.lambdap)

        for i in range(0,self.nz_u):
            ## Calculate wall fluxes
            FluxWall = Flux_CI.Flux_Wall(self.vx[i],self.vy[i],self.th[i],self.Cdrag[i],self.ptw,self.rho[i])
            uva[i] = FluxWall[0]
            vva[i] = FluxWall[1]
            uvb[i] = FluxWall[2]
            vvb[i] = FluxWall[3]
            tva[i] = FluxWall[4]
            tvb[i] = FluxWall[5]
            evb[i] = FluxWall[6]

            # Within the canyon for simple and non-probabilistic canyon
            # Term in momentum equation [s^-1]
            srim_vx_v[i] = uva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in momentum equation [s^-1]
            srim_vy_v[i] = vva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in turbulent kinetic energy equation [m^2 s^-3]
            srex_tke_v[i] = evb[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in energy equation [s^-1]
            srim_th_v[i] = tva[i]*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Term in energy equation [K s^-1]
            # Option a: do not consider HVAC waste heat as source/sink term
            srex_th_v[i] = tvb[i]*4*self.lambdaf*self.pb[i]/max(1e-6,self.hmean)/self.vol[i]/self.nd
            # Option b: consider HVAC waste heat as source/sink term at the street
            # Kinematic heat flux of the HVAC waste heat should be scaled according to building footprint area and urban area in each urban unit
            # HVAC waste heat flux building x Building footprint area = HVAC waste heat flux urban x urban area
            # HVAC waste heat flux urban = (Building footprint area / urban area) x HVAC waste heat flux building
            # HVAC waste heat flux urban = (lambdap / (1-lambdap)) x HVAC waste heat flux building
            #srex_th_v[i] = tvb[i] * 4 * self.lambdaf * self.pb[i] / max(1e-6, self.hmean) / self.vol[i] / self.nd+\
            #               self.HVAC_frac*(self.SensHt_HVAC/(self.rho[i]*self.Cp))*(1/max(1e-6,self.hmean))*self.lambdap/(1 - self.lambdap)**2


        for i in range(0, self.nz_u + 1):
            sfr[i] = -self.rho[i]*self.Cp*thb[i]

        for i in range(0,self.nz_u):
            sfw[i] = self.rho[i]*self.Cp*(tvb[i]+tva[i]*self.th[i])


        return srex_vx_h,srex_vy_h,srex_tke_h,srex_th_h,srim_vx_v,srim_vy_v,srex_tke_v,srim_th_v,srex_th_v,sfr,sfw, ustarCol







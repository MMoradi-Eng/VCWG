import numpy
import math
from scipy.interpolate import interp1d

"""
This class is a radiation model aimed to determine solar radiation received by urban elements
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Shortwave radiations are calculated based on (Redon et al., 2017)
Longwave radiations are calculated based on (Lee and Park, 2008)
"""

class SolarModel2:
    def __init__(self,forc,solar,UCM,BEM,road,nz,nz_u,dz,bldHeight,wy,tveg_tmp,f_LAD,h_LAD,h_tk,albVeg,Ncloud,LAI,var_sens,emisVeg,albWall):
        self.forc = forc
        self.solar = solar
        self.UCM = UCM
        self.BEM = BEM
        self.road = road
        self.nz = nz                    # Number of grid points in vertical column
        self.nz_u = nz_u                # Number of grid points within the canyon
        self.dz = dz                    # Grid resolution [m]
        self.bldHeight = bldHeight      # Average building height [m]
        self.wy = wy                    # distance between buildings at street level in the y direction [m]
        self.tveg_tmp = tveg_tmp        # vegetation temperature [K]
        self.f_LAD = f_LAD              # Leaf area density (LAD) function [m^2 m^-3]
        self.h_LAD = h_LAD              # Leaf area density (LAD) function (z location) [m]
        self.h_tk = h_tk                # Height of trunk [m]
        self.albVeg = albVeg            # Vegetation albedo
        self.Ncloud = Ncloud            # Fraction of sky covered by cloud
        self.LAI = LAI                  # Leaf area index (LAI) [m^2 m^-2]
        self.var_sens = var_sens        # any variable for sensitivity analysis @@@@@ sensitivity analysis @@@@@
        self.emisVeg = emisVeg
        self.albWall = albWall

        # Avoid zenith angle to approach 90 deg.
        if self.solar.zenith * 180/numpy.pi > 88:
            self.solar.zenith = 88*numpy.pi/180

    def SolCalRedon(self):

        # Calculate azimuth angle:
        # Azimuth Angle = Pi - acos[(sin(lat)*cos(zenith)-sin(Dec))/(cos(lat)*sin(zenith))]
        # where "radlat" is latitude (rad), "zenith" is Angle between normal to earth's surface and sun position
        # "decsol" is solar declination angle
        azimuth = math.acos((math.sin(self.solar.radlat)*math.cos(self.solar.zenith)-math.sin(self.solar.decsol))/
                            (math.cos(self.solar.radlat) * math.sin(self.solar.zenith)))


        # Extinction coefficient
        k = 2.5

        # Canyon direction [rad] (degrees from geographical north, in the counter trigonometric direction)
        theta_can = numpy.pi*135/180

        # Canyon and vegetation parameters
        hw = self.bldHeight/self.wy       # Canyon  aspect ratio
        h_t = max(self.h_LAD)             # Height of tree canopy [m]
        h_cw = (h_t+self.h_tk)/2          # Mid-height of tree's crown [m]
        delta_g = self.road.vegCoverage   # cover fractions of garden
        delta_r = 1-delta_g               # Cover fractions of road
        Gamma = 1                         # fraction of vegetation in "delta_g"
        delta_t = 0.5                     # High-vegetation cover fraction
        delta_nv = 0.5                    # Proportion of bare soil in gardens
        delta_lv = 1 - delta_nv           # Proportion of low vegetation in gardens
        n_cw = int(h_t/self.dz)           # Number grid points up to the height of tree

        # Calculate view factors between urban elements (Appendix A, Redon et al., 2017)
        vf_rs = numpy.sqrt((hw) ** 2 + 1) - (hw)                     # View factor between road and sky
        vf_sr = vf_rs                                                # View factor between sky and road
        vf_wr = 0.5 * ((hw) + 1 - numpy.sqrt((hw) ** 2 + 1)) / (hw)  # View factor between wall and road
        vf_ws = vf_wr                                                # View factor between wall and sky
        vf_sw = 1 - vf_sr                                            # View factor between sky and wall
        vf_ww = 1 - 2 * vf_ws                                        # View factor between walls
        vf_rw = 1 - vf_rs                                            # View factor between road and wall
        vf_st = numpy.sqrt((hw * (self.bldHeight - h_cw) / self.bldHeight) ** 2 + 1) - (hw * (self.bldHeight - h_cw) / self.bldHeight) # View factor between sky and tree
        vf_rt = numpy.sqrt((hw * h_cw / self.bldHeight) ** 2 + 1) - (hw * h_cw / self.bldHeight)                               # View factor between road and tree
        vf_wt = 1 - 0.5 * (vf_st + vf_rt)                            # View factor between wall and tree

        # Calculate transmissivity between urban elements (Appendix B, Redon et al., 2017)
        # Calculate transmissivity between road and sky (and vice versa)
        intg_rs = 0
        for i in range(0, n_cw):
            intg_rs = intg_rs + self.f_LAD(i*self.dz) * self.dz
        tau_rs = 1 - delta_t * (1 - numpy.exp(-k * intg_rs))
        tau_sr = tau_rs

        # Calculate transmissivity between road and wall (and vice versa)
        intg_rw = 0
        for i in range(0, int(n_cw/2)+1):
            intg_rw = intg_rw + self.f_LAD(i*self.dz) * self.dz
        tau_rw = 1 - delta_t * (1 - numpy.exp(-k * intg_rw))
        tau_wr = tau_rw

        # Calculate transmissivity between sky and wall (and vice versa)
        intg_ws = 0
        for i in range(int(n_cw/2),n_cw):
            intg_ws = intg_ws + self.f_LAD(i*self.dz) * self.dz
        tau_ws = 1 - delta_t * (1 - numpy.exp(-k * intg_ws))
        tau_sw = tau_ws

        # Calculate transmissivity between walls
        intg_ww = 0
        for i in range(0,n_cw):
            intg_ww = intg_ww + self.f_LAD(i*self.dz) * self.dz
        tau_ww = 1 - delta_t * (1 - numpy.exp(-k * intg_ww))

        alfa_r = self.road.albedo                # Albedo of road
        alfa_g = self.albVeg                     # Albedo of garden
        alfa_w = self.albWall                    # Albedo of wall
        alfa_t = self.albVeg                     # Albedo of tree

        eps_w = self.BEM[0].wall.emissivity      # Emissivity for the wall
        eps_r = self.road.emissivity             # Emissivity for the road
        eps_roof = self.BEM[0].roof.emissivity   # Emissivity for the roof
        eps_t = self.emisVeg                     # Emissivity for the tree
        sigma = 5.67e-8                          # Stefan-Boltzmann constant [W m^-2 K^-4]
        sigma_t = 0.9                            # Tree radiation property from [Lee and Park 2008]
        # sigma_t = self.var_sens #@@@@ sensitivity analysis @@@@@#

        # Calculate direct solar radiation (Section 4.1, Redon et al., 2017)
        # Calculate direct solar radiation reaching the top of trees (equation 1) [W m^-2]
        # which is composed of radiation transmitted through the foliage ("SDir_tf"), reflected ("SRef") and absorbed("SRes")
        SDir_t = self.forc.dir*max(0,1-hw*(self.bldHeight-h_t)*math.tan(self.solar.zenith)*math.sin(abs(azimuth-theta_can))/self.bldHeight)

        # The proportion of direct solar radiation transmitted through the foliage (equation 3) [W m^-2]
        SDir_tf = SDir_t * numpy.exp(-k * self.LAI)
        # The reflected radiation part of direct solar radiation reaching the top of trees (equation 4) [W m^-2]
        SRef = alfa_t * SDir_t * (1 - numpy.exp(-k * self.LAI))
        # The radiation absorbed by tress (equation 5) [W m^-2]
        SRes = (1 - alfa_t) * SDir_t * (1 - numpy.exp(-k * self.LAI))

        # The direct solar radiation received by the ground [W m^-2] (equation 6)
        SDir_r = (self.forc.dir-alfa_t*(SRef+SRes))*max(0,1-hw* math.tan(self.solar.zenith)*math.sin(abs(azimuth-theta_can)))

        # The direct solar radiation received by the sunlit wall [W m^-2] (equation 7)
        SDir_wA = (self.forc.dir - SDir_r - alfa_t * (SRef + SRes)) * (1 / hw)
        # The direct solar radiation received by the shaded wall [W m^-2] (equation 8)
        SDir_wB = 0

        # The direct solar radiation received by the garden [W m^-2]
        SDir_g = SDir_r

        # Calculate diffusive solar radiation [W m^-2] (Section 4.2, Redon et al., 2017)
        # Diffuse solar radiation received by road [W m^-2] (equation 9)
        SDif_r = self.forc.dif * vf_rs * tau_rs
        # Diffuse solar radiation received by garden [W m^-2]
        SDif_g = SDif_r
        # Diffuse solar radiation received by wall [W m^-2] (equation 10)
        SDif_w = self.forc.dif * vf_ws * tau_ws
        # Diffuse solar radiation received by trees [W m^-2] (equation 11)
        SDif_t = (self.forc.dif - (delta_r * SDif_r + delta_g * SDif_g + 2 * hw * SDif_w)) / delta_t

        # Calculate total shortwave radiation absorption by solving infinite reflection (Appendix C, Redon et al., 2017)
        # Calculate the first reflection ("0") on each element [W m^-2]
        # road (R), sunlit wall (A), shaded wall (B), garden (G), tree (T)
        R0 = alfa_r * (SDir_r + SDif_r)
        A0 = alfa_w * (SDir_wA + SDif_w)
        B0 = alfa_w * (SDir_wB + SDif_w)
        G0 = alfa_g * (SDir_g + SDif_g)
        W0 = (A0+B0)/2  # ???????????????
        # The specific coefficients associated with the view factors
        crt = 0
        cwt = (1 - vf_st) / (2 - vf_st - vf_rt)
        T0 = alfa_t * (SDir_t - SDir_tf + SDif_t)

        # Geometric and reflective factors
        Frw = vf_rw * tau_rw * alfa_r
        Frt = crt * vf_rt * delta_t * alfa_r
        Fgw = vf_rw * tau_rw * alfa_g
        Fgt = crt * vf_rt * delta_t * alfa_g
        Fwr = vf_wr * tau_wr * delta_r * alfa_w
        Fwg = vf_wr * tau_wr * delta_g * alfa_w
        Fww = vf_ww * tau_ww * alfa_w
        Fwt = cwt * vf_wt * delta_t * alfa_w
        Ftw = (vf_sw * (1 - tau_sw) + vf_rw * (1 - tau_rw) + vf_ww * (1 - tau_ww)) * (1 / delta_t) * alfa_t
        Ftr = (vf_sr * (1 - tau_sr) + vf_wr * (1 - tau_wr)) * (delta_r / delta_t) * alfa_t
        Ftg = (vf_sr * (1 - tau_sr) + vf_wr * (1 - tau_wr)) * (delta_g / delta_t) * alfa_t

        # Calculate multiple reflections ("inf")
        D = (1 - Fwr * Frw - Fwg * Fgw - Fww) * (1 - Ftr * Frt - Ftg * Fgt) - (Fwr * Frt + Fwg * Fgt + Fwt) * (
                Ftr * Frw + Ftg * Fgw + Ftw)

        Term1_Rinf = Frw * (Frt * Fwr + Fgt * Fwg + Fwt) + Frt * (1 - Fwr * Frw - Fwg * Fgw - Fww)
        Term2_Rinf = Frw * (Fwr * (1 - Ftg * Fgt) + Ftr * (Fwg * Fgt + Fwt)) + Frt * (
                Ftr * (1 - Fwg * Fgw) + Fwr * (Ftg * Fgw + Ftw))
        Term3_Rinf = Frw * (Fwg * (1 - Ftr * Frt) + Ftg * (Fwr * Frt + Fwt)) + Frt * (
                Ftg * (1 - Fwr * Frw - Fww) + Fwg * (Ftr * Frw + Ftw))
        Term4_Rinf = Frt * (Ftr * Frw + Ftg * Fgw + Ftw) + Frw * (1 - Ftr * Frt - Ftg * Fgt)
        Rinf = (Term1_Rinf / D) * T0 + (1 + Term2_Rinf / D) * R0 + (Term3_Rinf / D) * G0 + (Term4_Rinf / D) * W0

        Term1_Ginf = Fgw * (Frt * Fwr + Fgt * Fwg + Fwt) + Fgt * (1 - Fwr * Frw - Fwg * Fgw - Fww)
        Term2_Ginf = Fgw * (Fwr * (1 - Ftg * Fgt) + Ftr * (Fwg * Fgt + Fwt)) + Fgt * (
                Frt * (1 - Fwg * Fgw) + Fwr * (Ftg * Fgw + Ftw))
        Term3_Ginf = Fgw * (Fwg * (1 - Ftr * Frt) + Ftg * (Fwr * Frt + Fwt)) + Fgt * (
                Ftg * (1 - Fwr * Frw - Fww) + Fwg * (Ftr * Frw + Ftw))
        Term4_Ginf = Fgt * (Ftr * Frw + Ftg * Fgw + Ftw) + Fgw * (1 - Ftr * Frt - Ftg * Fgt)
        Ginf = (Term1_Ginf / D) * T0 + (Term2_Ginf / D) * R0 + (1 + Term3_Ginf / D) * G0 + (Term4_Ginf / D) * W0

        Term1_Tinf = 1 - Fwr * Frw - Fwg * Fgw - Fww
        Term2_Tinf = Ftr * (1 - Fwg * Fgw - Fww) + Fwr * (Ftg * Fgw + Ftw)
        Term3_Tinf = Ftg * (1 - Fwr * Frw - Fww) + Fwg * (Ftr * Frw + Ftw)
        Term4_Tinf = Ftr * Frw + Ftg * Fgw + Ftw
        Tinf = (Term1_Tinf / D) * T0 + (Term2_Tinf / D) * R0 + (Term3_Tinf / D) * G0 + (Term4_Tinf / D) * W0

        Term1_Winf = Fwr * Frt + Fwg * Fgt + Fwt
        Term2_Winf = Fwr * (1 - Ftg * Fgt) + Ftr * (Fwg * Fgt + Fwt)
        Term3_Winf = Fwg * (1 - Ftr * Frt) + Ftg * (Fwr * Frt + Fwt)
        Term4_Winf = 1 - Frt * Ftr - Ftg * Fgt
        Winf = (Term1_Winf / D) * T0 + (Term2_Winf / D) * R0 + (Term3_Winf / D) * G0 + (Term4_Winf / D) * W0

        # First absorption of total shortwave radiation by each element
        S_r0 = (1 - alfa_r) * (SDir_r + SDif_r)
        S_g0 = (1 - alfa_g) * (SDir_g + SDif_g)
        S_wA0 = (1 - alfa_w) * (SDir_wA + SDif_w)  # ??????????
        S_wB0 = (1 - alfa_w) * (SDir_wB + SDif_w)  # ??????????
        S_t0 = (1 - alfa_t) * (SDir_t - SDir_tf + SDif_t)

        # Sum of total shortwave radiation absorbed by each element
        # Total shortwave radiation absorbed by sky [W m^-2]
        self.S_sinf = vf_sr * tau_sr * (delta_r * Rinf + delta_g * Ginf) + vf_sw * tau_sw * Winf + vf_st * delta_t * Tinf

        # Total shortwave radiation absorbed by road [W m^-2]
        self.S_rinf = S_r0 + (1 - alfa_r) * (vf_rw * tau_rw * Winf + crt * vf_rt * delta_t * Tinf)

        # Total shortwave radiation absorbed by garden [W m^-2]
        self.S_ginf = S_g0 + (1 - alfa_g) * (vf_rw * tau_rw * Winf + crt * vf_rt * delta_t * Tinf)

        # Total shortwave radiation absorbed by sunlit wall [W m^-2]
        self.S_wAinf = S_wA0 + (1 - alfa_w) * (vf_wr * tau_wr * (delta_r * Rinf + delta_g * Ginf) + vf_ww * tau_ww * (
                Winf / 2) + cwt * vf_wt * delta_t * Tinf)  # ????

        # Total shortwave radiation absorbed by shaded wall [W m^-2]
        self.S_wBinf = S_wB0 + (1 - alfa_w) * (vf_wr * tau_wr * (delta_r * Rinf + delta_g * Ginf) + vf_ww * tau_ww * (
                Winf / 2) + cwt * vf_wt * delta_t * Tinf)  # ????

        # Total shortwave radiation absorbed by sunlit and shaded walls [W m^-2]
        SumWall = S_wA0 + S_wB0 + (1 - alfa_w)*(2*vf_wr * tau_wr * (delta_r * Rinf + delta_g * Ginf) + 2*vf_ww * tau_ww * Winf + 2*cwt * vf_wt * delta_t * Tinf)

        # Total shortwave radiation absorbed by trees [W m^-2]
        self.S_tinf = (1 / delta_t) * (
                (self.forc.dir + self.forc.dif) - (self.S_sinf + delta_r * self.S_rinf + delta_g * self.S_ginf + hw * SumWall))

        # Total shortwave radiation received by road can not be greater than sum of direct and diffusive radiation at road
        if (SDir_r + SDif_r)-self.S_rinf < 0:
            self.S_rinf = 0

        # Calculate longwave radiation (Lee and Park, 2008)
        L_atm_emt = (9.4e-6)*sigma*(self.forc.temp**6)+60*self.Ncloud # Longwave radiation from sky [W m^-2] (equation 65 in Lee and Park 2008)
        L_r_emt = eps_r*sigma*self.UCM.roadTemp**4               # Longwave radiation from road [W m^-2]
        L_w_emt = eps_w*sigma*self.UCM.wallTemp**4               # Longwave radiation from wall [W m^-2]
        L_roof_emt = eps_roof*sigma*self.UCM.roofTemp**4
        L_t_emt = eps_t*sigma*self.tveg_tmp**4                   # Longwave radiation from tree [W m^-2]
        L_tt = L_t_emt*delta_g*Gamma*(1-self.LAI)                     # Absorbed longwave radiation fluxes emitted from trees [W m^-2]
        L_tr = 0.5*sigma_t*(L_t_emt-L_tt)*vf_st                  # Absorbed longwave radiation fluxes emitted from road [W m^-2]
        L_tw = (0.25/hw)*sigma_t*(L_t_emt-L_tt)*(2-2*vf_st)      # Absorbed longwave radiation fluxes emitted from wall [W m^-2]

        # Calulate longwave radiation absorbed by wall [W m^-2]
        self.L_w_abs = tau_sw*eps_w*vf_ws*L_atm_emt + tau_ww*(eps_w**2)*(1-2*vf_ws)*sigma*(self.UCM.wallTemp**4) + tau_rw*eps_w*vf_ws*L_r_emt + \
                   tau_sr*tau_rw*(1-eps_r)*vf_ws*vf_rs*L_atm_emt + tau_sw*tau_ww*(1-eps_w)*vf_ws*(1-2*vf_ws)*L_atm_emt +\
                   tau_wr*tau_rw*(1-eps_r)*eps_w*vf_ws*(1-vf_rs)*sigma*self.UCM.wallTemp**4 + (tau_ww**2)*eps_w*(1-eps_w)*((1-2*vf_ws)**2)*sigma*(self.UCM.wallTemp)**4 + \
                   tau_rw*tau_ww*(1-eps_w)*vf_ws*(1-2*vf_ws)*L_r_emt + L_tw

        # Calulate longwave radiation absorbed by road [W m^-2]
        self.L_r_abs = tau_sr*eps_r*vf_rs*L_atm_emt + tau_wr*eps_w*eps_r*(1-vf_rs)*sigma*self.UCM.wallTemp**4 +\
                   tau_sw*tau_wr*(1-eps_w)*vf_ws*(1-vf_rs)*1*L_atm_emt +\
                   tau_ww*tau_wr*eps_w*(1-eps_w)*(1-vf_rs)*(1-2*vf_ws)*1*sigma*self.UCM.wallTemp**4 +\
                   tau_rw*tau_wr*(1-eps_w)*vf_ws*(1-vf_rs)*L_r_emt + 1*delta_r*L_tr

        # Calculate longwave radiation absorbed by trees [W m^-2]
        L_atm_rec = eps_t*(1-vf_st)*L_t_emt + eps_r*(1-vf_sr)*L_r_emt + eps_w*(1-vf_sw)*L_w_emt + eps_roof*L_roof_emt
        self.L_t_abs = (1/sigma_t)*(L_atm_emt + 2*hw*L_w_emt + L_r_emt + sigma_t*L_tt - 2*hw*self.L_w_abs - self.L_r_abs - 0.6*L_atm_rec)

        # Calculate net longwave radiation which is equal to absorbed longwave radiation - emitted longwave radiation
        # Calculate net longwave radiation of the wall [W m^-2] (equation 39)
        self.L_w = self.L_w_abs - eps_w*sigma*(self.UCM.wallTemp**4)

        # Calculate net longwave radiation of the road [W m^-2] (equation 40)
        self.L_r = self.L_r_abs - eps_r*sigma*(self.UCM.roadTemp**4)

        # Calculate net longwave radiation of the trees [W m^-2] (equation 42)
        self.L_t = self.L_t_abs - L_t_emt

        # Calculate total net longwave radiation within the canyon [W m^-2]
        self.L_tot = self.L_w+self.L_r+self.L_t

        return azimuth,self.solar.zenith,vf_rs,vf_wr,vf_sw,vf_ww,vf_rw,vf_st,vf_rt,vf_wt,tau_sr,tau_wr,tau_sw,tau_ww,\
               SDir_t,SDir_tf,SRef,SRes,SDir_wA,SDir_wB,SDir_r,SDif_r,SDif_w,SDif_t,self.S_sinf,self.S_rinf,self.S_ginf,\
               self.S_wAinf,self.S_wBinf,self.S_tinf,L_atm_emt,L_r_emt,L_w_emt,L_roof_emt,L_t_emt,L_tt,L_tr,L_tw,self.L_w_abs,\
               self.L_r_abs,self.L_t_abs,self.L_w,self.L_r,self.L_t,self.L_tot



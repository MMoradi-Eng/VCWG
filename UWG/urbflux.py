from infracalcs import infracalcs
from math import log
import numpy

"""
Calculate the surface heat fluxes in the urban area
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Bruno Bueno
"""

def urbflux(UCM, BEM, forc, parameter, simTime, RSM, L_w, L_r, var_sens,WindUrban):

    T_can = UCM.canTemp     # Temperature within the canyon [K]

    Cp = parameter.cp       # Heat capacity of dry air [J kg^-1 K^-1]
    UCM.Q_roof = 0.         # Sensible heat flux from building roof (convective) [W m^-2]
    sigma = 5.67e-8         # Stephan-Boltzman constant [W m^-2 K^-4]
    UCM.roofTemp = 0.       # Average urban roof temperature [K]
    UCM.wallTemp = 0.       # Average urban wall temperature [K]

    for j in xrange(len(BEM)):
        # Building energy model
        BEM[j].building.BEMCalc(UCM, BEM[j], forc, parameter, simTime)
        # Electricity consumption of urban area [W]
        BEM[j].ElecTotal = BEM[j].building.ElecTotal * BEM[j].fl_area

        # Update net longwave radiation of the roof
        # Emissivity of the outer surface of the roof
        e_roof = BEM[j].roof.emissivity
        # Roof temperature exposed to outdoor environment [K]
        T_roof = BEM[j].roof.layerTemp[0]
        # Net longwave radiation of the roof [W m^-2]
        BEM[j].roof.infra = e_roof * (forc.infra - sigma * T_roof**4.)

        # Update net longwave radiation of the wall
        # There are two radiation models to calculate net longwave radiation:
        # Option 1: Using UWG radiation model:
        # Emissivity of the outer surface of the wall
        # e_wall = BEM[j].wall.emissivity
        # Wall temperature exposed to outdoor environment [K]
        # T_wall = BEM[j].wall.layerTemp[0]
        # calculates the infrared radiation for wall, taking into account radiation exchange from road [W m^-2]
        #_infra_road_, BEM[j].wall.infra = infracalcs(UCM, forc, UCM.road.emissivity, e_wall, UCM.roadTemp, T_wall)
        # Option 2: Using new radiation model developed by (Lee and Park, 2008)
        # Net longwave radiation of the wall [W m^-2]
        BEM[j].wall.infra = L_w

        # Update element temperatures [K]
        # Calculate temperature of the floor (mass) [K]
        BEM[j].mass.layerTemp = BEM[j].mass.Conduction(simTime.dt, BEM[j].building.fluxMass,1.,0.,BEM[j].building.fluxMass)
        # Calculate surface temperatures of the roof [K]
        BEM[j].roof.SurfFlux(forc,parameter,simTime,UCM.canHum,T_can,max(forc.wind,UCM.canWind),1.,BEM[j].building.fluxRoof,'roof',var_sens)
        # Calculate surface temperatures of the wall [K]
        BEM[j].wall.SurfFlux(forc,parameter,simTime,UCM.canHum,T_can,UCM.canWind,1.,BEM[j].building.fluxWall,'wall',var_sens)

        # Calculate average wall & roof temperatures [K]
        # Depending on how many building types we have, surface temperature are determined by summing up the
        # fraction of building stock * outdoor temperature
        UCM.wallTemp = UCM.wallTemp + BEM[j].frac*BEM[j].wall.layerTemp[0]
        UCM.roofTemp = UCM.roofTemp + BEM[j].frac*BEM[j].roof.layerTemp[0]

    # Update net longwave radiation of the road
    # There are two radiation models to calculate net longwave radiation:
    # Option 1: Using UWG radiation model:
    # assume walls have similar emissivity, so use the last one
    # UCM.road.infra, _wall_infra = infracalcs(UCM,forc,UCM.road.emissivity,e_wall,UCM.roadTemp,UCM.wallTemp)
    # Option 2: Using new radiation model developed by (Lee and Park, 2008)
    UCM.road.infra = L_r
    # Calculate surface temperature of the road [K]
    UCM.road.SurfFlux(forc,parameter,simTime,UCM.canHum,T_can,UCM.canWind,2.,0.,'road',var_sens)
    # Update surface temperature of the road exposed to the outdoor environment [K]
    UCM.roadTemp = UCM.road.layerTemp[0]

    # Calculate latent heat flux within the canopy [W m^-2]
    if UCM.latHeat != None:
        UCM.latHeat = UCM.latHeat + UCM.latAnthrop + UCM.treeLatHeat + UCM.road.lat*(1.-UCM.bldDensity)
    '''
    # !!!!! UBL IS REMOVED !!!!!
    # ---------------------------------------------------------------------
    # Advective heat flux to UBL from VDM
    #
    # Note: UWG_Matlab code here is modified to compensate for rounding errors
    # that occur when recursively adding forDens, intAdv1, and intAdv2.
    # This causes issues in the UBL.advHeat calculatiuon when large (1e5)
    # numbers are subtracted to produce small numbers (1e-10) that can
    # differ from equivalent matlab calculations by a factor of 2.
    # Values this small are ~ 0, but for consistency's sake Kahan Summation
    # algorithm is applied to keep margin of difference from UWG_Matlab low.
    # ---------------------------------------------------------------------
    
    forDens = 0.0
    intAdv1 = 0.0
    intAdv2 = 0.0

    # c1 & c2 stores values truncated by floating point rounding for values < 10^-16
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    # for iz in xrange(RSM.nzfor):
    for iz in xrange(RSM.nz):
        # At c loss of precision at at low order of magnitude, that we need in UBL.advHeat calc
        # Algebraically t is 0, but with floating pt numbers c will accumulate truncated values
        #y = RSM.densityProfC[iz]*RSM.dz[iz]/(RSM.z[RSM.nzfor-1] + RSM.dz[RSM.nzfor-1]/2.)
        y = RSM.densityProfC[iz]*2/(RSM.z[RSM.nz-1] + 2/2.)
        t = forDens + y
        c1 += (t - forDens) - y
        forDens = t

        # y = RSM.windProf[iz]*RSM.tempProf[iz]*RSM.dz[iz]
        y = RSM.windProf[iz] * RSM.tempProf[iz] * 2
        t = intAdv1 + y
        c2 += (t - intAdv1) - y
        intAdv1 = t

        #y = RSM.windProf[iz]*RSM.dz[iz]
        y = RSM.windProf[iz] * 2
        t = intAdv2 + y
        c3 += (t - intAdv2) - y
        intAdv2 = t

    # Add the truncated values back
    forDens -= c1
    intAdv1 -= c2
    intAdv2 -= c3
    UBL.advHeat = UBL.paralLength*Cp*forDens*(intAdv1-(UBL.ublTemp*intAdv2))/UBL.urbArea
    '''

    # Blending height [m] (approximately the top of the roughness sublayer)
    zrUrb = 2*UCM.bldHeight
    # Reference height [m]
    zref = RSM.z[RSM.nz-1]

    # There are two ways to calculate friction velocity (ustar):
    # Option1: Using logarithmic wind speed profile (Appendix A, Bueno et al.,2014)
    # Reference wind speed [m s^-1]
    # windUrb = forc.wind*log(zref/RSM.z0r)/log(parameter.windHeight/RSM.z0r)*log(zrUrb/UCM.z0u)/log(zref/UCM.z0u)
    # Friction velocity [m s^-1]
    # UCM.ustar = parameter.vk*windUrb/log((zrUrb-UCM.l_disp)/UCM.z0u)
    # Option2: Using wind speed profile determined from column (1-D) model.
    # "WindUrban" represent mean wind speed within the canyon obtained from column (1-D) model

    # Calculate canyon air density [kg m^-3]
    dens = forc.pres/(1000*0.287042*T_can*(1.+1.607858*UCM.canHum))
    UCM.rhoCan = dens

    # Friction velocity [m s^-1] (equation 8, Appendix A, Bueno et al.,2014)
    # ustar evaluated at twice building height given WindUrban from the 1D column model and urban aerodynamic roughness length z0u
    UCM.ustar = parameter.vk * WindUrban / log((zrUrb - UCM.l_disp) / UCM.z0u)
    # Convective scaling velocity [m s^-1] (equation 10, Appendix A, Bueno et al.,2014)
    wstar = (parameter.g*max(UCM.sensHeat,0.0)*zref/dens/Cp/T_can)**(1/3.)
    # Modified friction velocity [m s^-3]
    # For thermal convection dominated condition wstar may be more significant than ustar, take the larger of the two
    UCM.ustarMod = max(UCM.ustar,wstar)

    # parameter.exCoeff = var_sens  #@@@@@ sensitivity analysis @@@@@

    # Calculate exchange velocity [m s^-1]
    UCM.uExch = parameter.exCoeff*UCM.ustarMod

    # Canyon turbulent velocity components (Bueno et al. 2014); not used here
    # UCM.turbU = 2.4*UCM.ustarMod
    # UCM.turbV = 1.9*UCM.ustarMod
    # UCM.turbW = 1.3*UCM.ustarMod

    # Urban wind profile
    # logarithmic profile is no longer used. It has been replaced by wind profile from column model
    #for iz in xrange(RSM.nzref):
    #    UCM.windProf.append(UCM.ustar/parameter.vk*\
    #        log((RSM.z[iz]+UCM.bldHeight-UCM.l_disp)/UCM.z0u))

    return UCM,BEM

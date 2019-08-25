import math
import logging

"""
Caculate properties of urban and rural elements 
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Bruno Bueno
"""

class Element(object):
    """
    UWG Element: These are urban elements such as road, roof, wall, ... for which properties are defined

    # Note: In matlab not all instance variables are instantiated. They are assumed to be a 0-by-0 empty matrix
    # https://www.mathworks.com/help/matlab/matlab_oop/specifying-properties.html

    Attributes:
        albedo;          % outer surface albedo
        emissivity;      % outer surface emissivity
        layerThickness;  % vector of layer thicknesses [m]
        layerThermalCond;% vector of layer thermal conductivities [W m^-1 K^-1]
        layerVolHeat;    % vector of layer volumetric heat [J m^-3 K^-1]
        vegCoverage;     % surface vegetation coverage
        layerTemp;       % vector of layer temperatures [K]
        waterStorage;    % thickness of water film [m] (only for horizontal surfaces)
        horizontal;      % 1-horizontal, 0-vertical
        solRec;          % solar radiation received [W m^-2]
        infra;           % net longwave radiation [W m^-2]
        lat;             % surface latent heat flux [W m^-2]
        sens;            % surface sensible heat flux [W m^-2]
        solAbs;          % solar radiation absorbed [W m^-2]
        aeroCond;        % convective heat transfer [W m^-2 K^-1]
        T_ext;           % external surface temperature [K]
        T_int;           % internal surface temperature [K]
        flux;            % net heat flux [W m^-2]
    """

    THICKNESSLST_EQ_MATERIALLST_MSG = \
    "-----------------------------------------\n" +\
    "ERROR: the number of layer thickness must\n" +\
    "match the number of layer materials\n"
    "-----------------------------------------"
    CONDUCTION_INPUT_MSG = 'ERROR: check input parameters in the Conduction routine'

    def __init__(self, alb, emis, thicknessLst, materialLst, vegCoverage, T_init, horizontal,name=None):

        if len(thicknessLst) != len(materialLst):
            raise Exception(self.THICKNESSLST_EQ_MATERIALLST_MSG)
        else:
            self._name = name                                       # purely for internal process
            self.albedo = alb                                       # outer surface albedo
            self.emissivity = emis                                  # outer surface emissivity
            self.layerThickness = thicknessLst                      # vector of layer thicknesses [m]
            self.layerThermalCond = map(lambda z: 0, materialLst)   # vector of layer thermal conductivity [W m^-1 K^-1]
            self.layerVolHeat = map(lambda z: 0, materialLst)       # vector of layer volumetric heat [J m^-3 K^-1]

            # Create list of layer k and (Cp*density) from materialLst properties
            for i in xrange(len(materialLst)):
              self.layerThermalCond[i] = materialLst[i].thermalCond
              self.layerVolHeat[i] = materialLst[i].volHeat

            self.vegCoverage = vegCoverage                          # surface vegetation coverage
            self.layerTemp = [T_init] * len(thicknessLst)           # vector of layer temperatures [K]
            self.waterStorage = 0.                                  # thickness of water film [m] for horizontal surfaces only
            self.infra = 0.                                         # net longwave radiation [W m^-2]
            self.horizontal = horizontal                            # 1-horizontal, 0-vertical
            self.sens = 0.                                          # surface sensible heat flux [W m^-2]

            # B/c we have to explicity define this in python. Set as None
            self.solRec = None                                       # solar radiation received [W m^-2]
            self.lat = None                                          # surface latent heat flux [W m^-2]
            self.solAbs = None                                       # solar radiation absorbed [W m^-2]
            self.aeroCond = None                                     # convective heat transfer coefficient [W m^-2 K^-1]
            self.T_ext = None                                        # external surface temperature [K]
            self.T_int = None                                        # internal surface temperature [K]
            self.flux = None                                         # net heat flux [W m^-2]


    def __repr__(self):
        # Returns some representative wall properties
        s1 = "Element: {a}\n\tlayerNum={b}, totaldepth={c}\n\t".format(
            a=self._name,
            b=len(self.layerThickness),
            c=sum(self.layerThickness)
            )
        s2 = "e={d}, a={e}\n\tk_avg={f}, Cp*dens_avg={g}\n\tlayerTemp: {h}".format(
            d=self.emissivity,
            e=self.albedo,
            f=round(sum(self.layerThermalCond),2),
            g=round(sum(self.layerVolHeat),2),
            h=self.layerTemp
            )
        return s1 + s2

    def is_near_zero(self,num,eps=1e-10):
        return abs(float(num)) < eps

    def SurfFlux(self,forc,parameter,simTime,humRef,tempRef,windRef,boundCond,intFlux,ZONE,var_sens):
        # Calculate net heat flux, and update element layer temperatures
        # Calculated per unit area (m^2)
        # Calculate air density [kg m^-3]
        dens = forc.pres/(1000*0.287042*tempRef*(1.+1.607858*humRef))

        # parameterization for convective heat transfer coeff can be significantly altered ( Palyvos, 2008)
        # UWG used (equation 12, Bueno et al., 2013)
        self.aeroCond = 5.8 + 10* windRef

        # Calculate latent, sensible and net heat fluxes
        if (self.horizontal):
            # Evaporation [m s^-1], Film water & soil latent heat
            if not self.is_near_zero(self.waterStorage) and self.waterStorage > 0.0:
                # N.B In the current UWG code, latent heat from evapotranspiration, stagnant water,
                # or anthropogenic sources is not modelled due to the difficulty of validation, and
                # lack of reliability of precipitation data from EPW files.Therefore this condition
                # is never run because all elements have had their waterStorage hardcoded to 0.
                qtsat = self.qsat([self.layerTemp[0]],[forc.pres],parameter)[0]
                eg = self.aeroCond*parameter.colburn*dens*(qtsat-humRef)/parameter.waterDens/parameter.cp
                self.waterStorage = min(self.waterStorage + simTime.dt*(forc.prec-eg),parameter.wgmax)
                self.waterStorage = max(self.waterStorage,0.) # [m]
            else:
                eg = 0.

            # Calculate latent heat flux from soil [W m^-2]
            soilLat = eg*parameter.waterDens*parameter.lv

            # Calculate sensible heat flux from elements including wall, roof, road and rural area on a semester basis
            # Winter: no vegetation
            # Solar radiation absorbed by roof and road are calculated in the new version of radiation model (Redon et al., 2017)
            # Herein, we only calculate solar radiation absorbed by rural surface area
            if simTime.month < parameter.vegStart and simTime.month > parameter.vegEnd:
                if ZONE == 'rural':
                    # Calculate solar radiation absorbed by surface rural area [W m^-2]
                    self.solAbs = (1.-self.albedo)*self.solRec

                # It is assumed that there is no vegetation during winter. So, latent and sensible heat fluxes are zero.
                vegLat = 0.
                vegSens = 0.

            # Summer: effect of vegetation is considered
            # Herein, we only calculate solar radiation absorbed by rural surface area
            else:
                if ZONE == 'rural':
                    # Calculate solar radiation absorbed by surface rural area [W m^-2]
                    self.solAbs = ((1.-self.vegCoverage)*(1.-self.albedo)+self.vegCoverage*(1.-parameter.vegAlbedo))*self.solRec

                # Calculate latent heat flux caused by vegetation [W m^-2]
                vegLat = self.vegCoverage*parameter.grassFLat*(1.-parameter.vegAlbedo)*self.solRec
                # Calculate sensible heat flux caused by vegetation [W m^-2]
                vegSens = self.vegCoverage*(1.-parameter.grassFLat)*(1.-parameter.vegAlbedo)*self.solRec

            # Calculate total latent heat flux [W m^-2]
            self.lat = soilLat + vegLat

            # Calculate total sensible heat flux [W m^-2]
            self.sens = vegSens + self.aeroCond*(self.layerTemp[0]-tempRef)
            # Calculate net heat flux [W m^-2]
            self.flux = -self.sens + self.solAbs + self.infra - self.lat

        # Calculate latent, sensible and net heat fluxes for wall
        else:
            # This "if" statement is redundant in the new version of UWG, which is integrated into the column model
            if ZONE == 'rural':
                self.solAbs = (1.-self.albedo)*self.solRec

            # Calculate total latent heat flux [W m^-2]
            self.lat = 0.

            # Calculate total sensible heat flux [W m^-2]
            self.sens = self.aeroCond*(self.layerTemp[0]-tempRef)
            # Calculate net heat flux [W m^-2]
            self.flux = -self.sens + self.solAbs + self.infra - self.lat

        # Solve conduction problem to determine temperature of layers
        self.layerTemp = self.Conduction(simTime.dt, self.flux, boundCond, forc.deepTemp, intFlux)
        # external surface temperature [K]
        self.T_ext = self.layerTemp[0]
        # internal surface temperature [K]
        self.T_int = self.layerTemp[-1]

    def SurfFluxTest(self,forc,parameter,simTime,humRef,tempRef,windRef,boundCond,intFlux):
        """ Calculate net heat flux, and update element layer temperatures
        """

        # Calculated per unit area (m^2)
        dens = forc.pres/(1000*0.287042*tempRef*(1.+1.607858*humRef)) # air density (kgd m-3)
        self.aeroCond = 5.8 + 8 * windRef         # Convection coef (ref: UWG, eq. 12))

        if (self.horizontal):
            # Evaporation (m s-1), Film water & soil latent heat
            if self.waterStorage > 0.:
                qtsat = self.qsat([self.layerTemp[0]],[forc.pres],parameter)[0]
                eg = self.aeroCond*parameter.colburn*dens*(qtsat-humRef)/parameter.waterDens/parameter.cp
                self.waterStorage = min(self.waterStorage + simTime.dt*(forc.prec-eg),parameter.wgmax)
                self.waterStorage = max(self.waterStorage,0.) # (m)
            else:
                eg = 0.
            soilLat = eg*parameter.waterDens*parameter.lv

            # Winter, no veg
            if simTime.month < parameter.vegStart and simTime.month > parameter.vegEnd:
                #self.solAbs = (1.-self.albedo)*self.solRec # (W m-2)
                # ---------------------------------------------------------------------
                # Redon Model
                # ---------------------------------------------------------------------
                self.solAbs = self.solRec
                vegLat = 0.
                vegSens = 0.
            else:    # Summer, veg
                #self.solAbs = ((1.-self.vegCoverage)*(1.-self.albedo)+self.vegCoverage*(1.-parameter.vegAlbedo))*self.solRec
                # ---------------------------------------------------------------------
                # Redon Model
                # ---------------------------------------------------------------------
                self.solAbs = self.solRec
                vegLat = self.vegCoverage*parameter.grassFLat*(1.-parameter.vegAlbedo)*self.solRec
                vegSens = self.vegCoverage*(1.-parameter.grassFLat)*(1.-parameter.vegAlbedo)*self.solRec

            self.lat = soilLat + vegLat

            # Sensible & net heat flux
            self.sens = vegSens + self.aeroCond*(self.layerTemp[0]-tempRef)
            self.flux = -self.sens + self.solAbs + self.infra - self.lat # (W m-2)

        else:               # For vertical surfaces (wall)
            #self.solAbs = (1.-self.albedo)*self.solRec
            # ---------------------------------------------------------------------
            # Redon Model
            # ---------------------------------------------------------------------
            self.solAbs = self.solRec
            self.lat = 0.

            # Sensible & net heat flux
            self.sens = self.aeroCond*(self.layerTemp[0]-tempRef)
            self.flux = -self.sens + self.solAbs + self.infra - self.lat # (W m-2)

        self.layerTemp = self.Conduction(simTime.dt, self.flux, boundCond, forc.deepTemp, intFlux)
        self.T_ext = self.layerTemp[0]
        self.T_int = self.layerTemp[-1]

    """
    Solve the conductance of heat based on of the element layers.
    arg:
        flx1  : net heat flux on surface
        bc    : boundary condition parameter (1 or 2)
        temp2 : deep soil temperature (ave of air temperature)
        flx2  : surface flux (sum of absorbed, emitted, etc.)

    key prop:
        za = [[ x00, x01, x02 ... x0w ]
              [ x10, x11, x12 ... x1w ]
                        ...
              [ xh0, xh1, xh2 ... xhw ]]

        where h = matrix row index    = element layer number
              w = matrix column index = 3

    """
    def Conduction(self, dt, flx1, bc, temp2, flx2):
        t = self.layerTemp          # vector of layer temperatures (K)
        hc = self.layerVolHeat      # vector of layer volumetric heat (J m-3 K-1)
        tc = self.layerThermalCond  # vector of layer thermal conductivities (W m-1 K-1)
        d = self.layerThickness     # vector of layer thicknesses (m)

        # flx1                      : net heat flux on surface
        # bc                        : boundary condition parameter (1 or 2)
        # temp2                     : deep soil temperature (avg of air temperature)
        # flx2                      : surface flux (sum of absorbed, emitted, etc.)

        fimp = 0.5                  # implicit coefficient
        fexp = 0.5                  # explicit coefficient
        num = len(t)                # number of layers

        # Mean thermal conductivity over distance between 2 layers (W/mK)
        tcp = [0 for x in xrange(num)]
        # Thermal capacity times layer depth (J/m2K)
        hcp = [0 for x in xrange(num)]
        # lower, main, and upper diagonals
        za = [[0 for y in xrange(3)] for x in xrange(num)]
        # RHS
        zy = [0 for x in xrange(num)]

        #--------------------------------------------------------------------------
        # Define the column vectors for heat capacity and conductivity
        hcp[0] = hc[0] * d[0]
        for j in xrange(1,num):
            tcp[j] = 2. / (d[j-1] / tc[j-1] + d[j] / tc[j])
            hcp[j] = hc[j] * d[j]

        #--------------------------------------------------------------------------
        # Define the first row of za matrix, and RHS column vector
        za[0][0] = 0.
        za[0][1] = hcp[0]/dt + fimp*tcp[1]
        za[0][2] = -fimp*tcp[1]
        zy[0] = hcp[0]/dt*t[0] - fexp*tcp[1]*(t[0]-t[1]) + flx1

        #--------------------------------------------------------------------------
        # Define other rows
        for j in xrange(1,num-1):
          za[j][0] = fimp*(-tcp[j])
          za[j][1] = hcp[j]/dt + fimp*(tcp[j]+tcp[j+1])
          za[j][2] = fimp*(-tcp[j+1])
          zy[j] = hcp[j]/dt * t[j] + fexp * \
            (tcp[j]*t[j-1] - tcp[j]*t[j] - tcp[j+1]*t[j] + tcp[j+1]*t[j+1])

        #--------------------------------------------------------------------------
        # Boundary conditions
        # heat flux
        if self.is_near_zero(bc-1.):
            za[num-1][0] = fimp * (-tcp[num-1])
            za[num-1][1] = hcp[num-1]/dt + fimp*tcp[num-1]
            za[num-1][2] = 0.
            zy[num-1] = hcp[num-1]/dt*t[num-1] + fexp*tcp[num-1]*(t[num-2]-t[num-1]) + flx2
        # deep-temperature
        elif self.is_near_zero(bc-2.):
            za[num-1][0] = 0.
            za[num-1][1] = 1.
            za[num-1][2] = 0.
            zy[num-1] = temp2
        else:
            raise Exception(self.CONDUCTION_INPUT_MSG)

        #--------------------------------------------------------------------------

        zx = self.invert(num,za,zy)
        # return zx as 1d vector of temperature layers
        return zx

    """
    Calculate (qsat_lst) vector of saturation humidity from:
    temp = vector of element layer temperatures
    pres = pressure (at current timestep).
    """
    def qsat(self,temp,pres,parameter):

        gamw = (parameter.cl - parameter.cpv) / parameter.rv
        betaw = (parameter.lvtt/parameter.rv) + (gamw * parameter.tt)
        alpw = math.log(parameter.estt) + (betaw /parameter.tt) + (gamw * math.log(parameter.tt))
        work2 = parameter.r/parameter.rv
        foes_lst = [0 for i in xrange(len(temp))]
        work1_lst = [0 for i in xrange(len(temp))]
        qsat_lst = [0 for i in xrange(len(temp))]

        for i in xrange(len(temp)):
          # saturation vapor pressure
          foes_lst[i] = math.exp( alpw - betaw/temp[i] - gamw*math.log(temp[i])  )
          work1_lst[i] = foes_lst[i]/pres[i]
          # saturation humidity
          qsat_lst[i] = work2*work1_lst[i] / (1. + (work2-1.) * work1_lst[i])

        return qsat_lst

    """
    Inversion and resolution of a tridiagonal matrix
            A X = C
    Input:
    nz number of layers
    a(*,1) lower diagonal (Ai,i-1)
    a(*,2) principal diagonal (Ai,i)
    a(*,3) upper diagonal (Ai,i+1)
    c
    Output
    x     results
    """

    def invert(self,nz,A,C):
        X = [0 for i in xrange(nz)]

        for i in reversed(xrange(nz-1)):
            C[i] = C[i] - A[i][2] * C[i+1]/A[i+1][1]
            A[i][1] = A[i][1] - A[i][2] * A[i+1][0]/A[i+1][1]

        for i in  xrange(1,nz,1):
            C[i] = C[i] - A[i][0] * C[i-1]/A[i-1][1]

        for i in xrange(nz):
            X[i] = C[i]/A[i][1]

        return X

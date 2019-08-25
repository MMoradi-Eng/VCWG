from math import sqrt, pow
import copy

"""
Urban Canopy Model; zero-dimensional energy balance model coupled to the 1D column model
Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed by Bruno Bueno
"""

"""
 Definition of Urban Canopy - Building Energy Model Class

 properties
     road;          % Road element class (moved from BEM)

     % Urban Canyon Parameters
     bldHeight;     % average building height [m]
     bldDensity;    % horizontal building density (footprint)
     verToHor;      % vertical-to-horizontal urban area ratio (facade area/urban area)
     treeCoverage;  % horizontal tree density (footprint)
     sensAnthrop;   % sensible anthropogenic heat (other than from buildings) [W m^-2]
     latAnthrop;    % latent anthropogenic heat (other than from buildings) [W m^-2]
     z0u;           % urban roughness length [m]
     l_disp;        % urban displacement length [m]
     roadShad;      % shadowing of roads
     canWidth;      % canyon width [m]
     bldWidth;      % bld width [m]
     canAspect;     % canyon aspect ratio
     roadConf;      % road-sky configuration factors (sky view factor)
     alb_wall;      % average wall albedo
     wallConf;      % wall-sky configuration factors (sky view factor)
     VFwallroad;    % wall-road view factor
     VFroadwall;    % road-wall view factor
     facArea;       % facade area [m^2]
     roadArea;      % road area [m^2]
     roofArea;      % roof area [m^2] (also building area)
     facAbsor;      % average facade absorptivity
     roadAbsor;     % average road absorptivity
     h_mix;         % fraction of building HVAC waste heat set to the street canyon [as opposed to the roof]

     % Urban Canyon Variables
     canTemp;       % canyon air temperature (dry-bulb temperature) [K]
     Tdp;           % dew point temperature
     Twb;           % wetbulb temperature
     canHum;        % canyon specific humidity [kgv kga^-1]
     canRHum;       % canyon relative humidity (%)
     canWind;       % urban canyon wind velocity [m s^-1]
     turbU;         % canyon turbulent velocities [m s^-1]
     turbV;         % canyon turbulent velocities [m s^-1]
     turbW;         % canyon turbulent velocities [m s^-1]
     ublTemp;       % urban boundary layer temperature [K]
     ublTempdx;     % urban boundary layer temperature discretization
     ublWind;       % urban boundary layer wind velocity [m s^-1]
     ustar;         % friction velocity [m s^-1]
     ustarMod;      % modified friction velocity [m s^-1]
     uExch;         % exchange velocity [m s^-1]
     treeLatHeat;   % latent heat from trees [W m^-2]
     treeSensHeat;  % sensible heat from trees [W m^-2]
     sensHeat;      % urban sensible heat [W m^-2]
     latHeat;       % urban latent heat [W m^-2]
     windProf;      % urban wind profile
     Q_roof;        % sensible heat flux from building roof (convective)
     Q_wall;        % sensible heat flux from building wall (convective)
     Q_window;      % sensible heat flux from building window (via U-factor)
     Q_road;        % sensible heat flux from road (convective)
     Q_hvac;        % sensible heat flux from HVAC waste
     Q_traffic;     % sensible heat flux from traffic (net)
     Q_ubl;         % Convective heat exchange with UBL layer
     Q_vent;        % Convective heat exchange from ventilation/infiltration
     SolRecWall;    % Solar received by wall [W m^-2]
     SolRecRoof;    % Solar received by roof [W m^-2]
     SolRecRoad;    % Solar received by road [W m^-2]
     roadTemp;      % average road temperature [K]
     roofTemp;      % average roof temperature [K]
     wallTemp;      % average wall temperature [K]
     ElecTotal;     % Total Electricity consumption of urban area [MW]
     GasTotal;      % Total Gas consumption of the urban area [MW]
 """

class UCMDef(object):

    CANYON_TEMP_BOUND_ERROR = "Something obviously went wrong (UCMDef.py)... "

    def __init__(self,bldHeight,bldDensity,verToHor,treeCoverage,sensAnthrop,latAnthrop,
        initialTemp,initialHum,initialWind,parameter,r_glaze,SHGC,alb_wall,road):
        self.road = road                                                   # Road element class (moved from BEM)
        self.bldHeight = bldHeight                                         # average building height [m]
        self.verToHor = verToHor                                           # ratio of vertical building height to total horizontal distance in an urban unit
                                                                           # UWG assumes a 2D infinite canyon,
                                                                           # while 1D transport model assumes symmetric 3D canyon (square building and urban blocks)
        self.bldDensity = bldDensity                                       # horizontal building density W/(w+W) where W is building width and w is canyon width
        self.treeCoverage = treeCoverage                                   # horizontal tree density (footprint)
        self.sensAnthrop = sensAnthrop                                     # sensible anthropogenic heat (other than from buildings) [W m^-2]
        self.latAnthrop = latAnthrop                                       # latent anthropogenic heat (other than from buildings) [W m^-2]
        self.roadShad = min(treeCoverage/(1-bldDensity),1)                 # fraction of road not building shadowed
        # Key to understanding next few formulas is that UWG_Matlab assumes bld_area = square, so sqrt(bld_area) = side length
        self.bldWidth = bldHeight*bldDensity/verToHor                      # bld width (W or side length) derived from bldDensity, building height, and verToHor [m]
        d = self.bldWidth/(bldDensity)                                     # urban unit width W/(W/(W+w)) [m]
        self.canWidth = d - self.bldWidth                                  # canyon width w = urban unit width - building width = W + w - W [m]
        self.canAspect = bldHeight/self.canWidth                           # canyon aspect ratio H / w
        self.roadConf = pow(pow(self.canAspect,2)+1,0.5) - self.canAspect  # road-sky configuration factor (sky view factor SVF)
        self.wallConf = 0.5 * (self.canAspect + 1 -
            pow(pow(self.canAspect,2)+1,0.5)) / (self.canAspect)           # wall-sky configuration factor (sky view factor SVF)
        self.facArea = 2*bldHeight                                         # Facade area per unit depth of an infinite 2D canyon  [m^2 m^-1]
        self.roadArea = self.canWidth                                      # road area per unit depth of an infinite 2D canyon  [m^2 m^-1]
        self.roofArea = self.bldWidth                                      # roof area per unit depth of an infinite 2D canyon  [m^2 m^-1]
        self.canTemp = initialTemp                                         # canyon air temperature (dry-bulb temperature) [K]
        self.roadTemp = initialTemp                                        # average road temperature [K]
        self.canHum = initialHum                                           # canyon specific humidity (kgv kga^-1)
        self.ublWind = max(initialWind,parameter.windMin)                  # urban boundary layer wind velocity [m s^-1]
        self.canWind = initialWind                                         # urban canyon wind velocity [m s^-1]
        self.ustar = 0.1*initialWind                                       # friction velocity [m s^-1]
        self.ustarMod = 0.1*initialWind                                    # modified friction velocity [m s^-1]

        # Calculate z0u = urban roughness length [m]
        # density of just street facing facade
        frontDens = verToHor
        # If buildings are very sparse with respect to cross canyon flow, i.e. lambdaf is small,
        # the actual urban aerodynamic roughness length is less than the Raupach (1991) prediction z0 ~ 0.15 H
        if frontDens < 0.15:
          self.z0u = frontDens * self.bldHeight
        else:
          # Otherwise use Raupach 1991 prediction z0 ~ 0.15 H
          self.z0u = 0.15 * self.bldHeight

        # Calculate l_dsp = urban displacement length (m)
        if frontDens < 0.05:
          self.l_disp = 3 * frontDens * self.bldHeight
        elif frontDens < 0.15:
          self.l_disp = (0.15+5.5*(frontDens-0.05))*self.bldHeight
        elif frontDens < 1:
          self.l_disp = (0.7+0.35*(frontDens-0.15))*self.bldHeight
        else:
          self.l_disp = 0.5*self.bldHeight

        # average wall albedo
        self.alb_wall = alb_wall
        # avg facade absorptivity == wall_mat_fraction * absorption + window_frac * non_solar_heat_gain
        self.facAbsor = (1-r_glaze)*(1-alb_wall) + r_glaze*(1-0.75*SHGC)
        # average road absorptivity
        self.roadAbsor = (1-road.vegCoverage)*(1-road.albedo)
        # urban sensible heat [W m-2]
        self.sensHeat = 0.0

        # Variables set in urbflux()
        # urban latent heat [W m-2]
        self.latHeat = None
        # wind profile
        self.windProf = []
        # canyon relative humidity(%)
        self.canRHum = None
        # dew point temperature [K]
        self.Tdp = None

    def __repr__(self):
        return "UCMDef: ver2Hor={b}, bldDens={c}, canyon H/W={a}/{d}={e}".format(
            b=self.verToHor,
            c=self.bldDensity,
            a=int(self.bldHeight),
            d=int(self.canWidth),
            e=round(self.canAspect,1)
            )

    def UCModel(self,BEM,T_ubl,forc,parameter):

        # Calculate air density within the canyon [kg m^-3]
        dens = forc.pres/(1000*0.287042*self.canTemp*(1.+1.607858*self.canHum))
        # Calculate air density above the canyon [kg m^-3]
        dens_ubl = forc.pres/(1000*0.287042*T_ubl*(1.+1.607858*forc.hum))
        # heat capacity for air (J kg^-1 K^-1)
        Cp_air = parameter.cp

        self.Q_wall = 0.     # sensible heat flux from building wall (convective) per unit urban area [W m^-2]
        self.Q_window = 0.   # sensible heat flux from building window (via U-factor) per unit [W m^-2]
        self.Q_road = 0.     # sensible heat flux from road (convective) per unit urban area [W m^-2]
        self.Q_hvac = 0.     # sensible heat flux from HVAC waste [W m^-2]
        self.Q_traffic = 0.  # sensible heat flux from traffic (net) per unit urban area [W m^-2]
        self.Q_vent = 0.     # Convective heat exchange flux from ventilation/infiltration [W m^-2]
        self.Q_ubl = 0.      # Convective heat exchange flux with UBL layer per unit urban area [W m^-2]
        self.ElecTotal = 0.  # Total Electricity consumption of urban area [MW]
        self.GasTotal = 0.   # Total Gas consumption of the urban area [MW]
        self.roofTemp = 0.   # average roof temperature [K]
        self.wallTemp = 0.   # average wall temperature [K]

        # Road to Canyon
        T_road = self.road.layerTemp[0]    # Road temperature [K]
        h_conv = self.road.aeroCond        # Convective heat transfer coefficient [W m^-2 K^-1]

        # Solve sensible heat balance equation for outdoor air, considering effect of heat fluxes from road, wall, window,
        # roof, vegetations, urban boundary layer (UBL), anthropogenic,....  [????] (equation 10, Bueno et al., 2013)
        # Explicit term in eq. 10 caused by road heat flux which does not contain T_can per unit depth of infinite 2D canyon [W m^-1]
        H1 = T_road*h_conv*self.roadArea
        # Implicit terms in eq. 10 caused by road heat flux which directly contains coefficient for T_can per unit depth of inifinit 2D canyon
        H2 = h_conv*self.roadArea

        # Cumulative explicit term in eq. 10 caused by adding UBL heat flux which does not contain T_can per unit depth of inifinite 2D canyon [W m^-1]
        H1 = H1 + T_ubl*self.roadArea*self.uExch*Cp_air*dens_ubl
        # Cumulative implicit terms in eq. 10 caused by adding UBL heat flux which directly contains coefficient for T_can
        H2 = H2 + self.roadArea*self.uExch*Cp_air*dens_ubl
        # Explicit term in eq. 10 caused by anthropogenic and vegetation heat fluxes release at road surface which contain T_can from previous iteration per unit depth of inifinite 2D canyon [W m^-1]
        Q = (self.roadArea)*(self.sensAnthrop + self.treeSensHeat*self.treeCoverage)

        # Building energy output to canyon, in terms of absolute (total) values
        for j in xrange(len(BEM)):
            # Call element of building energy model (BEM)
            building = BEM[j].building
            wall = BEM[j].wall
            T_indoor = building.indoorTemp          # Indoor temperature [K]
            T_wall = wall.layerTemp[0]              # surface temperature of the wall exposed to the outdoor environment [K]
            R_glazing = building.glazingRatio       # Glazing ratio
            A_wall = (1.-R_glazing)*self.facArea    # Wall area (extract area of windows) per unit depth of inifinite 2D canyon [m^2 m^-1]
            A_window = R_glazing*self.facArea       # Windows area per unit depth of infinite 2D canyon [m^2 m^-1]
            U_window = building.uValue              # Window U-value [W m^-2 K^-1] including film coefficient

            # Cumulative explicit terms in eq. 10 caused by adding heat fluxes from building elements which do not contain T_can per unit depth of inifinite 2D canyon [W m^-1]
            H1 = H1 + BEM[j].frac*(                 # fraction of the urban floor space of this typology
                T_indoor*A_window*U_window +        # term from window load per unit depth of inifinite 2D canyon [W m^-1]
                T_wall*A_wall*h_conv +              # term from wall load per unit depth of infinite 2D canyon [W m^-1]
                T_indoor*self.roofArea*BEM[j].building.vent*BEM[j].building.nFloor*Cp_air*dens +     # term from ventilation load per unit depth of inifinite 2D canyon [W m^-1]
                T_indoor*self.roofArea*BEM[j].building.infil*self.bldHeight/3600.0*Cp_air*dens)      # term from infiltration load per unit depth of inifinite 2D canyon [W m^-1]

            # Cumulative implicit terms in eq. 10 caused by adding heat fluxes from building elements which directly contain coefficient for T_can
            H2 = H2 + BEM[j].frac*(                 # fraction of the urban floor space of this typology
                A_window*U_window +                 # term from window load
                A_wall*h_conv +                     # term from wall load
                self.roofArea*BEM[j].building.vent*BEM[j].building.nFloor*Cp_air*dens +              # term from ventilation load
                self.roofArea*BEM[j].building.infil*self.bldHeight/3600.0*Cp_air*dens)               # term from infiltration load

            # Cumulative explicit term in eq. 10 caused by adding heat flux from HVAC and solar Heat Gain on windows
            # which contain T_can from previous iteration per unit depth of infinite 2D canyon [W m^-1]
            Q = Q + BEM[j].frac*(
                self.roofArea*building.sensWaste*self.h_mix + # HVAC waste heat per unit depth of inifinite 2D canyon [W m^-1]
                A_window*BEM[j].wall.solRec*(1.0-BEM[j].building.shgc)) # Solar Heat Gain on windows per unit depth of inifinite 2D canyon [W m^-1]

            # Update wall temperature [K]
            self.wallTemp = self.wallTemp + BEM[j].frac*T_wall
            # Update roof temperature [K]
            self.roofTemp = self.roofTemp + BEM[j].frac*BEM[j].roof.layerTemp[0]

            # Convective heat exchange with UBL layer [W m^-2]
            self.Q_ubl = self.Q_ubl + BEM[j].frac*self.bldDensity*(BEM[j].roof.sens + BEM[j].building.sensWaste*(1.-self.h_mix))

        # Solve for canyon temperature [K]
        self.canTemp = (H1 + Q)/H2

        # Road fraction of sensible heat flux (convective) per unit urban area [W m^-2]
        self.Q_road = h_conv*(T_road-self.canTemp)*(1.-self.bldDensity)
        # Road fraction of convective heat exchange with UBL layer per unit urban area [W m^-2]
        self.Q_ubl = self.Q_ubl + self.uExch*Cp_air*dens*(self.canTemp-T_ubl)*(1.-self.bldDensity)
        # Wall fraction of sensible heat flux from building wall (convective) per unit urban area [W m^-2]
        self.Q_wall = h_conv*(self.wallTemp-self.canTemp)*(self.verToHor)
        # Road fraction of sensible heat flux from traffic (net) per unit urban area [W m^-2]
        self.Q_traffic = self.sensAnthrop

        # Building energy output to canyon, per m^2 of urban area
        T_can = copy.copy(self.canTemp)

        for j in xrange(len(BEM)):
            # total ventilation volumetric flow rate per building footprint area [m^3 s-1 m^-2]
            V_vent = BEM[j].building.vent*BEM[j].building.nFloor
            # total infiltration volumetric flow rate per building footprint area [m^3 s-1 m^-2]
            V_infil = BEM[j].building.infil*self.bldHeight/3600.0
            # Indoor temperature [K]
            T_indoor = BEM[j].building.indoorTemp
            # window U-value [W m^-2 K^-1] including film coefficient
            U_window = BEM[j].building.uValue
            # Glazing ratio
            R_glazing = BEM[j].building.glazingRatio

            # sensible heat flux from building window (via U-factor) [W m^-2]
            self.Q_window = self.Q_window + BEM[j].frac*self.verToHor*R_glazing*U_window*(T_indoor-T_can)
            self.Q_window = self.Q_window + BEM[j].frac*self.verToHor*R_glazing*BEM[j].wall.solRec*(1.-BEM[j].building.shgc)
            # building fraction of convective heat exchange from ventilation/infiltration [W m^-2]
            self.Q_vent = self.Q_vent + BEM[j].frac*self.bldDensity*Cp_air*dens*(V_vent + V_infil)*(T_indoor-T_can)
            # building fraction of sensible heat flux from HVAC waste [W m^-2]
            self.Q_hvac = self.Q_hvac + BEM[j].frac*self.bldDensity*BEM[j].building.sensWaste*self.h_mix
            # building fraction of sensible heat flux from roof [W m^-2]
            self.Q_roof = self.Q_roof + BEM[j].frac*self.bldDensity*BEM[j].roof.sens

            # Total Electrical & Gas power [MW]
            self.ElecTotal = self.ElecTotal + BEM[j].fl_area*BEM[j].building.ElecTotal/1.e6
            self.GasTotal = self.GasTotal + BEM[j].fl_area*BEM[j].building.GasTotal/1.e6

        # Sensible Heat
        # N.B In the current UWG code, latent heat from evapotranspiration, stagnant water,
        # or anthropogenic sources is not modelled due to the difficulty of validation, and
        # lack of reliability of precipitation data from EPW files.

        # Calculate urban sensible heat [W m^-2]
        self.sensHeat = self.Q_wall + self.Q_road + self.Q_vent + self.Q_window + self.Q_hvac + self.Q_traffic + self.treeSensHeat + self.Q_roof

        # Check canyon temperature not to be too high or too low
        if self.canTemp > 350. or self.canTemp < 250:
            print(self.canTemp)

            # If desired run this command to stop the code
            # raise Exception(self.CANYON_TEMP_BOUND_ERROR)

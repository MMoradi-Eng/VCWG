
from psychrometrics import psychrometrics, moist_air_density
import logging
"""
Calculate building characteristics
Developed by Bruno Bueno
Building Technology, Massachusetts Institute of Technology (MIT), Cambridge, U.S.A.
Last update: 2012
"""
class Building(object):
    """

    properties
        % Building parameters
        floorHeight         % floor height [m]
        intHeat;            % time step internal heat gains per unit floor area [W m^-2] (bld) (sensible only)
        intHeatNight;       % nighttime internal heat gains per unit floor area [W m^-2] (floor)
        intHeatDay;         % daytime internal heat gains per unit floor area [W m^-2] (floor)
        intHeatFRad;        % radiant fraction of internal gains
        intHeatFLat;        % latent fraction of internal gains
        infil;              % Infiltration Air Change per Hour (ACH) [hr^-1]
        vent;               % Ventilation rate per unit floor area [m^3 s^-1 m^-2]
        glazingRatio;       % glazing ratio
        uValue;             % window U-value [W m^-2 K^-1] (including film coeff)
        shgc;               % window Solar Heat Gain Coefficient (SHGC)
        condType;           % cooling condensation system type {'AIR', 'WATER'}
        cop;                % COP of the cooling system (nominal)
        coolSetpointDay;    % daytime indoor cooling set-point [K]
        coolSetpointNight;  % nighttime indoor cooling set-point [K]
        heatSetpointDay;    % daytime indoor heating set-point [K]
        heatSetpointNight;  % nighttime indoor heating set-point [K]
        coolCap;            % rated cooling system capacity [W m^-2]
        heatCap;            % rated heating system capacity [W m^-2]
        heatEff;            % heating system efficiency (-)
        canyon_fraction     # fraction of waste heat released to canyon, default = 1
        mSys;               % HVAC supply mass flowrate (kg s-1 m-2)
        indoorTemp;         % indoor air temperature [K]
        indoorHum;          % indoor specific humidity [kgv kga^-1]
        Twb;                % wetbulb temperature
        Tdp;                % dew point [C]
        indoorRhum;         % indoor relative humidity

        area_floor;         % total floor space of the BEM
        FanMax;             % max fan flow rate (m^3/s) per DOE
        nFloor;             % number of floors
        RadFOcc;            % Radiant fraction of occupant
        LatFOcc;            % Latent fraction of occupant
        RadFEquip;          % Radiant fraction of equipment
        RadFLight;          % Radiant fraction of light

        Type;               % DOE reference building type
        Era;                % PRE80, PST80, NEW
        Zone;               % Climate zone number

        % Calculated values
        sensCoolDemand;     % building sensible cooling demand per unit building footprint area [W m^-2]
        sensHeatDemand;     % building sensible heating demand per unit building footprint area [W m^-2]
        copAdj;             % adjusted COP per temperature
        dehumDemand;        % Latent heat demand for dehumidification of air per unit building footprint area [W m^-2]
        coolConsump;        % cooling energy consumption per unit building footprint area OR per unit floor area [W m^-2]
        heatConsump;        % heating energy consumption per unit floor area [W m^-2]
        sensWaste;          % sensible waste heat per unit building footprint area [W m^-2]
        latWaste;           % lat waste heat per unit building footprint area [W m^-2]
        fluxMass;           % mass surface heat flux (W m-2) (mass to indoor air)
        fluxWall;           % wall surface heat flux (W m-2) (wall to inside)
        fluxRoof;           % roof surface heat flux (W m-2) (roof to inside)
        fluxSolar;          % solar heat gain per unit floor area [W m^-2] through window (SHGC)
        fluxWindow;         % heat gain/loss from window per unit floor area [W m^-2] (U-value)
        fluxInterior;       % internal heat gain adjusted for latent/LW heat per unit floor area [W m^-2]
        fluxInfil;          % heat flux from infiltration per unit floor area [W m^-2]
        fluxVent;           % heat flux from ventilation per unit floor area [W m^-2]
        ElecTotal;          % total electricity consumption per unit floor area [W m^-2]
        GasTotal;           % total gas consumption per unit floor area [W m^-2]
        Qhvac;              % total heat removed (sensible + latent) per unit building footprint area [W m^-2] (calculated in cooling system)
        Qheat;              % total heat added (sensible only) per unit building footprint area [W m^-2] (calculated in heating system)
    """

    TEMPERATURE_COEFFICIENT_CONFLICT_MSG = "FATAL ERROR!"

    def __init__(self,floorHeight,intHeatNight,intHeatDay,intHeatFRad,\
            intHeatFLat,infil,vent,glazingRatio,uValue,shgc,\
            condType,cop,coolSetpointDay,coolSetpointNight,\
            heatSetpointDay,heatSetpointNight,coolCap,heatEff,initialTemp):

            self.floorHeight =float(floorHeight)        # floor height
            self.intHeat = intHeatNight                 # timestep internal sensible heat gain per unit floor area [W m^-2]
            self.intHeatNight = intHeatNight            # nighttime internal heat gain per unit floor area [W m^-2]
            self.intHeatDay = intHeatDay                # daytime internal heat gain per unit floor area [W m^-2]
            self.intHeatFRad = intHeatFRad              # internal gain radiant fraction
            self.intHeatFLat = intHeatFLat              # internal gain latent fraction
            self.infil = infil                          # Infiltration Air Change per Hour (ACH) [hr^-1]
            self.vent = vent                            # Ventilation rate per unit floor area [m^3 s^-1 m^-2]
            self.glazingRatio = glazingRatio            # glazing ratio
            self.uValue = uValue                        # window U-value [W m^-2 K^-1] including film coefficient
            self.shgc = shgc                            # window Solar Heat Gain Coefficient (SHGC), fraction of radiation that is admitted through a window
            self.condType = condType                    # cooling condensation system type: AIR, WATER
            self.cop = cop                              # COP of cooling system (nominal)
            self.coolSetpointDay = coolSetpointDay      # daytime indoor cooling setpoint [K]
            self.coolSetpointNight = coolSetpointNight  # nighttime indoor heating setpoint [K]
            self.heatSetpointDay = heatSetpointDay      # daytimge indoor heating setpoint [K]
            self.heatSetpointNight = heatSetpointNight  # nighttime indoor heating setpoint [K]
            self.coolCap = coolCap                      # rated cooling system capacity [W m^-2]
            self.heatEff = heatEff                      # heating system capacity (-)
            self.mSys = coolCap/1004./(min(coolSetpointDay,coolSetpointNight)-14-273.15) # HVAC supply mass flowrate (kg s-1 m-2)
            self.indoorTemp = initialTemp               # Indoor Air Temperature [K]
            self.indoorHum = 0.012                      # Indoor specific humidity [kgv/kga]
            self.heatCap = 999                          # Default heat capacity value
            self.copAdj = cop                           # adjusted COP per temperature
            self.canyon_fraction = 1.0                  # Default canyon fraction

            self.Type = "null"                          # DOE reference building type
            self.Era = "null"                           # pre80, pst80, new
            self.Zone = "null"                          # Climate zone number

            # Logger will be disabled by default unless explicitly called in tests
            self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return "BuildingType: {a}, Era: {b}, Zone: {c}".format(
            a=self.Type,
            b=self.Era,
            c=self.Zone
            )

    def is_near_zero(self,val,tol=1e-14):
        return abs(float(val)) < tol

    def BEMCalc(self,UCM,BEM,forc,parameter,simTime):

        self.logger.debug("Logging at {} {}".format(__name__, self.__repr__()))

        # Building Energy Model
        self.ElecTotal = 0.0                            # total electricity consumption - (W/m^2) of floor
        self.nFloor = max(UCM.bldHeight/float(self.floorHeight),1)   # At least one floor
        self.Qheat = 0.0                                # total sensible heat added (or heating demand) per unit building footprint area [W m^-2]
        self.sensCoolDemand = 0.0                       # building sensible cooling demand per unit building footprint area [W m^-2]
        self.sensHeatDemand = 0.0                       # building sensible heating demand per unit building footprint area [W m^-2]
        self.coolConsump  = 0.0                         # cooling energy consumption per unit building footprint area OR per unit floor area [W m^-2]
        self.heatConsump  = 0.0                         # heating energy consumption per unit floor area [W m^-2]
        self.sensWaste = 0.0                            # Sensible waste heat per unit building footprint area [W m^-2]
        self.dehumDemand  = 0.0                         # Latent heat demand for dehumidification of air per unit building footprint area [W m^-2]
        self.Qhvac = 0.0                                # Total heat removed (sensible + latent)
        Qdehum = 0.0
        dens =  moist_air_density(forc.pres,self.indoorTemp,self.indoorHum)# [kgv/ m-3] Moist air density given dry bulb temperature, humidity ratio, and pressure
        evapEff = 1.                                    # evaporation efficiency in the condenser
        volVent = self.vent * self.nFloor               # total ventilation volumetric flow rate per building footprint area [m^3 s-1 m^-2]
        volInfil = self.infil * UCM.bldHeight / 3600.   # total infiltration volumetric flow rate per building footprint area [m^3 s-1 m^-2]
        T_wall = BEM.wall.layerTemp[-1]                 # Inner layer
        massFlorRateSWH = BEM.SWH * self.nFloor/3600.   # Solar water heating per building footprint area per hour [kg s^-1 m^-2] (Change of units [hr^-1] to [s^-1]
        T_ceil = BEM.roof.layerTemp[-1]                 # Inner layer
        T_mass = BEM.mass.layerTemp[0]                  # Outer layer
        T_indoor = self.indoorTemp                      # Indoor temp (initial)
        T_can = UCM.canTemp                             # Canyon temperature

        # Normalize areas to building foot print [m^2/m^2(bld)]
        facArea = UCM.verToHor/UCM.bldDensity           # Facade (exterior) area per unit building footprint area [m^2 m^-2]
        wallArea = facArea*(1.-self.glazingRatio)       # Wall area per unit building footprint area [m^2 m^-2]
        winArea = facArea*self.glazingRatio             # Window area per unit building footprint area [m^2 m^-2]
        massArea = 2*self.nFloor-1                      # ceiling and floor (top & bottom) per unit building footprint area [m^2 m^-2]
        ceilingArea = 1                                 # ceiling area per unit building footprint area [m^2 m^-2]; must be equal to 1; we are geneous!

        # Set temperature set points according to night/day set points in building schedule & simTime; need the time in [hr]
        isEqualNightStart = self.is_near_zero((simTime.secDay/3600.) - parameter.nightSetStart)
        if simTime.secDay/3600. < parameter.nightSetEnd or (simTime.secDay/3600. > parameter.nightSetStart or isEqualNightStart):
            self.logger.debug("{} Night set points @{}".format(__name__,simTime.secDay/3600.))

            # Set point temperatures in [K]
            T_cool = self.coolSetpointNight
            T_heat = self.heatSetpointNight

            # Internal heat per unit building footprint area [W m^-2]
            self.intHeat = self.intHeatNight * self.nFloor
        else:
            self.logger.debug("{} Day set points @{}".format(__name__,simTime.secDay/3600.))

            # Set point temperatures in [K]
            T_cool = self.coolSetpointDay
            T_heat = self.heatSetpointDay

            # Internal heat per unit building footprint area [W m^-2]
            self.intHeat = self.intHeatDay*self.nFloor

        # Indoor convection heat transfer coefficients
        # wall convective heat transfer coefficient [W m^-2 K^-1]
        zac_in_wall = 3.076
        # other surfaces convective heat transfer coefficient [W m^-2 K^-1]
        zac_in_mass = 3.076

        # If ceiling temperature is greater than indoor temperature use a different convective heat transfer coefficient
        if T_ceil > T_indoor:
            zac_in_ceil  = 0.948
        # If ceiling temperature is less than indoor temperature use a different convective heat transfer coefficient
        elif (T_ceil < T_indoor) or self.is_near_zero(T_ceil-T_indoor):
            zac_in_ceil  = 4.040
        else:
            print T_ceil, T_indoor
            raise Exception(self.TEMPERATURE_COEFFICIENT_CONFLICT_MSG)
            return

        # -------------------------------------------------------------
        # Heat fluxes [W m^-2]
        # -------------------------------------------------------------
        # Solar Heat Gain on windows per building footprint area [W m^-2]:
        # = radiation intensity [W m^-2] * Solar Heat Gain Coefficient (SHGC) * window area per unit building foot print area [m^2 m^-2]
        winTrans = (BEM.wall.solRec * self.shgc * winArea)

        # QL: Latent heat per unit floor area [W m^-2] from infiltration & ventilation from
        # volInfil and volVent: volumetric rate of infiltration or ventilation per unit area [m^3 s^-1 m^-2]
        # parameter.lv: latent heat of evaporation [J kgv^-1]
        # dens: density [kga m^-3]
        # UCM.canHum: canyon specific humidity [kgv kga^-1]
        # indoorHum: indoor specific humidity [kgv kga^-1]
        # Note: at the moment the infiltration and system specific humidity are considered to be the same
        # This is a serious limitation.
        # Future versions of the UWG must calculate the system specific humidity based on HVAC system parameters

        # Latent heat per building footprint area [W m^-2]
        QLinfil = volInfil * dens * parameter.lv * (UCM.canHum - self.indoorHum)
        QLvent = volVent * dens * parameter.lv * (UCM.canHum - self.indoorHum)

        # Latent heat load per unit building footprint area [W m^-2]
        QLintload = self.intHeat * self.intHeatFLat

        # Note: at the moment the infiltration and system air temperatures are considered to be the same
        # This is a serious limitation.
        # Future versions of UWG must calculate the system temperature based on HVAC system parameters

        # Heat/Cooling load per unit building footprint area [W m^-2], if any
        self.sensCoolDemand = max(
            wallArea*zac_in_wall*(T_wall - T_cool) +            # wall load per unit building footprint area [W m^-2]
            massArea*zac_in_mass*(T_mass - T_cool) +            # other surfaces load per unit building footprint area [W m^-2]
            winArea*self.uValue*(T_can-T_cool) +                # window load due to temperature difference per unit building footprint area [W m^-2]
            ceilingArea*zac_in_ceil *(T_ceil-T_cool) +          # ceiling load per unit building footprint area [W m^-2]
            self.intHeat +                                      # internal load per unit building footprint area [W m^-2]
            volInfil*dens*parameter.cp*(T_can-T_cool) +         # infiltration load per unit building footprint area [W m^-2]
            volVent*dens*parameter.cp*(T_can-T_cool) +          # ventilation load per unit building footprint area [W m^-2]
            winTrans,                                           # solar load through window per unit building footprint area [W m^-2]
            0.)

        self.sensHeatDemand = max(
            -(wallArea*zac_in_wall*(T_wall-T_heat) +            # wall load per unit building footprint area [W m^-2]
            massArea*zac_in_mass*(T_mass-T_heat) +              # other surfaces load per unit building footprint area [W m^-2]
            winArea*self.uValue*(T_can-T_heat) +                # window load due to temperature difference per unit building footprint area [W m^-2]
            zac_in_ceil*(T_ceil-T_heat) +                       # ceiling load per unit building footprint area [W m^-2]
            self.intHeat +                                      # internal load per unit building footprint area [W m^-2]
            volInfil*dens*parameter.cp*(T_can-T_heat) +         # infiltration load per unit building footprint area [W m^-2]
            volVent*dens*parameter.cp*(T_can-T_heat) +          # ventilation load per unit building footprint area [W m^-2]
            winTrans),                                          # solar load through window per unit building footprint area [W m^-2]
            0.)

        # -------------------------------------------------------------
        # HVAC system (cooling demand = [W m^-2] bld footprint)
        # -------------------------------------------------------------
        # If the canyon air temperature is greater than 288 K building energy system is under cooling mode
        if self.sensCoolDemand > 0. and UCM.canTemp > 288.:
            # Energy is used to cool volumetric flow rate of air per unit building footprint area,
            # equal to sensCoolDemand / (dens * Cp * x * (T_indoor - 10C))
            # Volumetric flow rate of air to cool per unit building footprint area [m^3 s^-1 m^-2]
            VolCool = self.sensCoolDemand / (dens*parameter.cp*(T_indoor-283.15))
            # Energy is used to dehumidify volumetric flow rate of air per unit building footprint area,
            # Assume 7.8 [gv kg^-1] of vapor in air at 10C
            # This energy is equal to VolCool * dens * (self.indoorHum - 0.9*0.0078)*parameter.lv
            # Latent heat demand for dehumidification of air per unit building footprint area [W m^-2]
            self.dehumDemand = max(VolCool * dens * (self.indoorHum - 0.9*0.0078)*parameter.lv, 0.)

            # Calculate total cooling demand in per unit building footprint area [W m^-2]
            # if cooling energy demand is greater then HVAC cooling capacity
            if (self.dehumDemand + self.sensCoolDemand) > (self.coolCap * self.nFloor):
                self.Qhvac = self.coolCap * self.nFloor
                # Part load ratio
                PLR = (self.coolCap * self.nFloor) / (self.dehumDemand + self.sensCoolDemand)
                VolCool = VolCool * PLR
                self.sensCoolDemand = self.sensCoolDemand * PLR
                self.dehumDemand = self.dehumDemand * PLR
            else:
                self.Qhvac = self.dehumDemand + self.sensCoolDemand

            # Since cooling demand may have been modified above, recalculate it
            Qdehum = VolCool * dens * parameter.lv * (self.indoorHum - 0.9*0.0078)
            # Calculate input work required by the refrigeration cycle per unit building footprint area [W m^-2]
            # COP = QL/Win or Win = QL/COP
            self.coolConsump = (max(self.sensCoolDemand+self.dehumDemand,0.0))/self.copAdj

            # Calculate waste heat from HVAC system per unit building footprint area [W m^-2]
            # Using 1st law of thermodynamics QH = Win + QL
            if (self.condType == 'AIR'):
                self.sensWaste = max(self.sensCoolDemand+self.dehumDemand,0)+self.coolConsump
                self.latWaste = 0.0
            # We have not tested this option; it must be investigated further
            elif (self.condType == 'WAT'):
                self.sensWaste = max(self.sensCoolDemand+self.dehumDemand,0)+self.coolConsump*(1.-evapEff)
                self.latWaste = max(self.sensCoolDemand+self.dehumDemand,0)+self.coolConsump*evapEff

            self.sensHeatDemand = 0.

        # -------------------------------------------------------------
        # HVAC system (heating demand = [W m^-2] bld footprint)
        # -------------------------------------------------------------
        # If the canyon air temperature is less than 288 K building energy system is under heating mode
        elif self.sensHeatDemand > 0. and UCM.canTemp < 288.:
            # Calculate total heating demand in per unit building footprint area [W m^-2]
            # Heating demand must be less than or equal to heating capacity
            self.Qheat = min(self.sensHeatDemand, self.heatCap*self.nFloor)
            # Calculate the energy consumption of the heating system per unit building footprint area [W m^-2] from heating demand divided by efficiency
            self.heatConsump  = self.Qheat / self.heatEff
            # Calculate waste heat from HVAC system per unit building footprint area [W m^-2]
            # Using 1st law of thermodynamics QL = Win - QH
            self.sensWaste = self.heatConsump - self.Qheat
            # Calculate the energy consumption of the heating system per unit floor area [W m^-2]
            self.heatConsump = self.heatConsump/self.nFloor
            # Calculate the heating energy demand system per unit floor area [W m^-2]
            self.sensHeatDemand = self.Qheat/self.nFloor
            # The heating system model assumes that the indoor air humidity is not controlled
            Qdehum = 0.0
            self.sensCoolDemand = 0.0


        # -------------------------------------------------------------
        # Evolution of the internal temperature and humidity
        # -------------------------------------------------------------
        # Solve sensible heat balance equation for indoor air, considering effect of heat fluxes from wall, mass, roof,
        # window, solar heat gain on windows, internal heat, infiltration, ventilation and HVAC (cooling or heating)
        # per unit building footprint area [W m^-2]
        # Explicit terms in eq. 2 which either do not contain Tin or contain Tin from previous iteration (Bueno et al., 2012)
        Q = self.intHeat + winTrans + self.Qheat - self.sensCoolDemand

        H1 = (T_wall*wallArea*zac_in_wall +
            T_mass*massArea*zac_in_mass +
            T_ceil*zac_in_ceil +
            T_can*winArea*self.uValue +
            T_can*volInfil * dens * parameter.cp +
            T_can*volVent * dens * parameter.cp)
        # Implicit terms in eq. 2 which directly contain coefficient for newest Tin to be solved (Bueno et al., 2012)
        H2 = (wallArea*zac_in_wall +
            massArea*zac_in_mass +
            zac_in_ceil +
            winArea*self.uValue +
            volInfil * dens * parameter.cp +
            volVent * dens * parameter.cp)

        # Assumes air temperature of control volume is sum of surface boundary temperatures
        # weighted by area and heat transfer coefficient + generated heat
        # Calculate indoor air temperature [K]
        self.indoorTemp = (H1 + Q)/H2
        # Solve the latent heat balance equation for indoor air, considering effect of internal, infiltration and
        # ventilation latent heat and latent heat demand for dehumidification per unit building footprint area [W m^-2];
        # eq. 3 (Bueno et al., 2012)
        # Calculate indoor specific humidity [kgv kga^-1]
        self.indoorHum = self.indoorHum + (simTime.dt/(dens * parameter.lv * UCM.bldHeight)) * \
            (QLintload + QLinfil + QLvent - Qdehum)

        # Calculate relative humidity ((Pw/Pws)*100) using pressure, indoor temperature, humidity
        _Tdb, _w, _phi, _h, _Tdp, _v = psychrometrics(self.indoorTemp, self.indoorHum, forc.pres)
        # Indoor relative humidity
        self.indoorRhum = _phi

        # Heat fluxes of elements [W m^-2]
        # (will be used for element calculation)
        # Wall heat flux per unit wall area [W m^-2]
        self.fluxWall = zac_in_wall * (T_indoor - T_wall)
        # Top ceiling heat flux per unit ceiling or building footprint area [W m^-2]
        self.fluxRoof = zac_in_ceil * (T_indoor - T_ceil)
        # Inner horizontal heat flux per unit floor area [W m^-2]
        self.fluxMass = zac_in_mass * (T_indoor - T_mass) + self.intHeat * self.intHeatFRad/massArea


        # Calculate heat fluxes per unit floor area [W m^-2] (These are for record keeping only)
        self.fluxSolar = winTrans/self.nFloor
        self.fluxWindow = winArea * self.uValue *(T_can - T_indoor)/self.nFloor
        self.fluxInterior = self.intHeat * self.intHeatFRad *(1.-self.intHeatFLat)/self.nFloor
        self.fluxInfil= volInfil * dens * parameter.cp *(T_can - T_indoor)/self.nFloor
        self.fluxVent = volVent * dens * parameter.cp *(T_can - T_indoor)/self.nFloor
        self.coolConsump = self.coolConsump/self.nFloor
        self.sensCoolDemand = self.sensCoolDemand/self.nFloor

        # Total Electricity consumption per unit floor area [W m^-2] which is equal to
        # cooling consumption + electricity consumption + lighting
        self.ElecTotal = self.coolConsump + BEM.Elec + BEM.Light

        # Calculate total sensible waste heat to canyon per unit building footprint area [W m^-2]
        # which can be determined from sensible waste to canyon, energy consumption for domestic hot water and gas consumption
        CpH20 = 4200.           # heat capacity of water [J Kg^-1 K^-1]
        T_hot = 49 + 273.15     # Service water temp (assume no storage) [K]
        self.sensWaste = self.sensWaste + (1/self.heatEff-1.)*(massFlorRateSWH*CpH20*(T_hot - forc.waterTemp)) + BEM.Gas*(1-self.heatEff)*self.nFloor

        # Calculate total gas consumption per unit floor area [W m^-2] which is equal to gas consumption per unit floor area +
        # energy consumption for domestic hot water per unit floor area + energy consumption of the heating system per unit floor area
        self.GasTotal = BEM.Gas + (massFlorRateSWH*CpH20*(T_hot - forc.waterTemp)/self.nFloor)/self.heatEff + self.heatConsump


"""
% Not used for this release but saved for possible future use
function Twb = wet_bulb(Tdb,Tdp,pres)

    % Copyright (c) 2015, Rolf Henry Goodwin
    % All rights reserved.
    %
    % Redistribution and use in source and binary forms, with or without
    % modification, are permitted provided that the following conditions are
    % met:
    %
    %     * Redistributions of source code must retain the above copyright
    %       notice, this list of conditions and the following disclaimer.
    %     * Redistributions in binary form must reproduce the above copyright
    %       notice, this list of conditions and the following disclaimer in
    %       the documentation and/or other materials provided with the distribution
    %
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    % ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    % LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    % INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    % CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    % POSSIBILITY OF SUCH DAMAGE.

    % Code modified to merge into a single file - Joseph Yang, 2016


    % Tdb, Tdp, Twb in K
    % p in Pa (obtained function uses hPa, so /100 needed)
    global T;
    global T_d;
    global p;
    T = Tdb;
    T_d = Tdp;
    p = pres/100;

    Twb = root_finder(@Delta_q,T_d,T);
end

function dQTw = Delta_q(T_w)
    %Delta_q finds the value of function dq(Tw)
    %INPUT wet bulb temperature T_w
    %OUTPUT dq(Tw)
    global T;
    global T_d;
    global p;

    Cp = 1005; % Heat capacity of water vapor in J/(kg*K)
    L = 2.501e6; % Latent heat of water vapor at 0 degC in J/kg
    w1 = mixing_ratio(T_d,p); % Mixing ratio corresponding to T_d and p
    w2 = mixing_ratio(T_w,p); % Mixing ratio corresponding to T_w and p

    dQTw = (L*(w2-w1))/(1+w2)-Cp*(T-T_w)*(1+0.8*w2); % Finds deltaq(Tw)

end

function r = root_finder(f,a,b)
    %root_finder calculates the roots of function f using the bisection search
    %method
    %INPUT function f, and interval a,b with the property that f(a) and f(b)
    %have opposite signs
    %OUTPUT r approximate value of root of f in interval [a,b]
    if (feval(f,a)*feval(f,b)) > 0
        disp('stop');
        error('Both endpoints have the same sign, please try again.')

    end

    while abs(b-a)>(10e-10)
        m = (a+b)/2;
        x1 = feval(f,m);
        x2 = feval(f,a);
        if (x1 > 0 && x2 < 0) || (x1 < 0 && x2 > 0)
            b = m;
        else
            a = m;
        end
    end
    r = (a+b)/2;
end

function w = mixing_ratio(T,p)
    %mixing_ratio finds the ratio of water vapor to the mass of dry air
    %INPUT Temperature and Pressure
    %OUTPUT MIXING RATIOs for inputting into wet_bulb.m
    p_a = 1013.246; % Standard sea-level atmospheric pressure in hPa
    T_a = 373.16; % Standard sea-level atmospheric temperature in Kelvin

    e1 = 11.344*(1-T/T_a);
    e2 = -3.49149*(T_a/T-1);
    f1 = -7.90298*(T_a/T-1);
    f2 = 5.02808*logn((T_a/T),10);
    f3 = -1.3816*((10^(e1)-1)/(1.e7));
    f4 = 8.1328*((10^(e2)-1)/(1.e3));
    f5 = logn(p_a,10);
    f = f1+f2+f3+f4+f5;
    e = 10^(f); % calculates vapor pressure in terms of T
    w = 0.62197*(e/(p-e)); % mass ratio g/kg
end

function [ z ] = logn(x,y)
    % logn
    %   Finds log base y of x
    z = log(x)/log(y);
end
"""

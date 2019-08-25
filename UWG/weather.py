from utilities import read_csv, str2fl
from math import pow, log, exp
from psychrometrics import HumFromRHumTemp

"""
Developed by Bruno Bueno
Building Technology, Massachusetts Institute of Technology (MIT), Cambridge, U.S.A.
Last update: 2012
"""

class Weather(object):
    """
    Weather
    Read epw file
    http://bigladdersoftware.com/epx/docs/8-2/auxiliary-programs/epw-csv-format-inout.html
    properties
        location  # location name
        staTemp   % air temperature [C}
        staTdp    % dewpoint temperature [C]
        staRhum   % air relative humidity [%]
        staPres   % air pressure [Pa]
        staInfra  % horizontal Infrared Radiation Intensity [W m^-2]
        staHor    % horizontal radiation
        staDir    % normal solar direct radiation [W m^-2]
        staDif    % horizontal solar diffuse radiation [W m^-2]
        staUdir   % wind direction [deg]
        staUmod   % wind speed [m s^-1]
        staRobs   % Precipitation [mm h^-1]
        staHum    % specific humidty [kg kg^-1]
    """

    def __init__(self,climate_file,HI,HF):
        #HI: Julian start date
        #HF: Julian final date
        #H1 and HF define the row we want

        # Open .epw file and feed csv data to self.climate_data
        try:
            self.climate_data = read_csv(climate_file)
        except Exception as e:
            raise Exception("Failed to read .epw file! {}".format(e.message))

        self.location = self.climate_data[0][1]
        cd = self.climate_data[HI:HF+1]
        self.staTemp = str2fl([cd[i][6] for i in xrange(len(cd))])           # drybulb [C]
        self.staTdp = str2fl([cd[i][7] for i in xrange(len(cd))])            # dewpoint [C]
        self.staRhum = str2fl([cd[i][8] for i in xrange(len(cd))])           # air relative humidity [%]
        self.staPres = str2fl([cd[i][9] for i in xrange(len(cd))])           # air pressure [Pa]
        self.staInfra = str2fl([cd[i][12] for i in xrange(len(cd))])         # horizontal Infrared Radiation Intensity [W m^-2]
        self.staHor = str2fl([cd[i][13] for i in xrange(len(cd))])           # horizontal radiation [W m^-2]
        self.staDir = str2fl([cd[i][14] for i in xrange(len(cd))])           # normal solar direct radiation [W m^-2]
        self.staDif = str2fl([cd[i][15] for i in xrange(len(cd))])           # horizontal solar diffuse radiation [W m^-2]
        self.staUdir = str2fl([cd[i][20] for i in xrange(len(cd))])          # wind direction [deg]
        self.staUmod = str2fl([cd[i][21] for i in xrange(len(cd))])          # wind speed [m s^-1]
        self.staRobs = str2fl([cd[i][33] for i in xrange(len(cd))])          # Precipitation [mm h^-1]
        self.staHum = [0.0] * len(self.staTemp)                              # specific humidity [kg kg^-1]
        for i in xrange(len(self.staTemp)):
            self.staHum[i] = HumFromRHumTemp(self.staRhum[i], self.staTemp[i], self.staPres[i])
        print(self.staHum)
        self.staTemp = [s+273.15 for s in self.staTemp]                      # air temperature [K]
        print(HI)
        print(HF)
    def __repr__(self):
        return "Weather: {a}, HI Tdb:{b}, HF Tdb:{c}".format(
            a=self.location,
            b=self.staTemp[0]-273.15,
            c=self.staTemp[-1]-273.15
            )

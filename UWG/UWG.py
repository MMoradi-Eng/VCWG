"""
=========================================================================
 THE VERTICAL CITY WEATHER GENERATOR (VCWG)
=========================================================================

Developed by Mohsen Moradi and Amir A. Aliabadi
Atmospheric Innovations Research (AIR) Laboratory, University of Guelph, Guelph, Canada
Last update: March 2019
Originally developed and edited by Bruno Bueno, A. Nakano, Lingfu Zhang, Joseph Yang, Saeran Vasanthakumar,
Leslie Norford, Julia Hidalgo, Gregoire Pigeon.
=========================================================================
"""

import os
import math
import cPickle
import copy
import utilities
import logging
import progress_bar
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate

from simparam import SimParam
from weather import  Weather
from building import Building
from material import Material
from element import Element
from BEMDef import BEMDef
from schdef import SchDef
from param import Param
from UCMDef import UCMDef
from forcing import Forcing
from RSMDef import RSMDef
from solarcalcs import SolarCalcs
import urbflux
from psychrometrics import psychrometrics
from readDOE import readDOE
from urbflux import urbflux
from ColModel import ColModel
from draglength import Drag_Length
from solarcalcsRedon import SolarModel2
from cdturb import CdTurb

# For debugging only
#from pprint import pprint
#from decimal import Decimal
#pp = pprint
#dd = Decimal.from_float
class UWG(object):
    """Morph a rural EPW file to urban conditions using a file with a list of urban parameters.

    args:
        epwDir: The directory in which the rural EPW file sits.
        epwFileName: The name of the rural epw file that will be morphed.
        uwgParamDir: The directory in which the UWG Parameter File (.uwg) sits.
        uwgParamFileName: The name of the UWG Parameter File (.uwg).
        destinationDir: Optional destination directory for the morphed EPW file.
            If left blank, the morphed file will be written into the same directory
            as the rural EPW file (the epwDir).
        destinationFileName: Optional destination file name for the morphed EPW file.
            If left blank, the morphed file will append "_UWG" to the original file name.
    returns:
        newClimateFile: the path to a new EPW file that has been morphed to account
            for uban conditions.
    """

    """ Section 1 - Definitions for constants / other parameters """
    MINTHICKNESS = 0.01    # Minimum layer thickness (to prevent crashing) (m)
    MAXTHICKNESS = 0.05    # Maximum layer thickness (m)
    SOILTCOND = 1          # http://web.mit.edu/parmstr/Public/NRCan/nrcc29118.pdf (Figly & Snodgrass)
    SOILVOLHEAT = 2e6      # http://www.europment.org/library/2013/venice/bypaper/MFHEEF/MFHEEF-21.pdf (average taken from Table 1)
    SOIL = Material(SOILTCOND, SOILVOLHEAT, name="soil")  # Soil material used for soil-depth padding

    # Physical constants
    G = 9.81               # gravity (m s-2)
    CP = 1004.             # heat capacity for air (J/kg K)
    VK = 0.40              # von karman constant (dimensionless)
    R = 287.               # gas constant dry air (J/kg K)
    RV = 461.5             # gas constant water vapor (J/kg K)
    LV = 2.26e6            # latent heat of evaporation (J/kg)
    SIGMA = 5.67e-08       # Stefan Boltzmann constant (W m-2 K-4)
    WATERDENS = 1000.      # water density (kg m-3)
    LVTT = 2.5008e6        #
    TT = 273.16            #
    ESTT = 611.14          #
    CL = 4.218e3           #
    CPV = 1846.1           #
    B = 9.4                # Coefficients derived by Louis (1979)
    CM = 7.4               #
    COLBURN = math.pow((0.713/0.621), (2/3.)) # (Pr/Sc)^(2/3) for Colburn analogy in water evaporation

    # Site-specific parameters
    WGMAX = 0.005 # maximum film water depth on horizontal surfaces (m)

    # File path parameter
    RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "Resources"))
    CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, epwFileName, uwgParamFileName=None, epwDir=None, uwgParamDir=None, destinationDir=None, destinationFileName=None,var_sens=None):
        self.var_sens = var_sens
        # Logger will be disabled by default unless explicitly called in tests
        self.logger = logging.getLogger(__name__)

        # User defined
        self.epwFileName = epwFileName if epwFileName.lower().endswith('.epw') else epwFileName + '.epw' # Revise epw file name if not end with epw
        self.uwgParamFileName = uwgParamFileName  # If file name is entered then will UWG will set input from .uwg file

        # If user does not overload
        self.destinationFileName = destinationFileName if destinationFileName else self.epwFileName.strip('.epw') + '_UWG.epw'
        self.epwDir = epwDir if epwDir else os.path.join(self.RESOURCE_PATH, "epw")
        self.uwgParamDir = uwgParamDir if uwgParamDir else os.path.join(self.RESOURCE_PATH,"parameters")
        self.destinationDir = destinationDir if destinationDir else os.path.join(self.RESOURCE_PATH,"epw_uwg")

        # Serialized DOE reference data
        self.readDOE_file_path = os.path.join(self.CURRENT_PATH,"readDOE.pkl")

        # EPW precision
        self.epw_precision = 1


        # init UWG variables
        self._init_param_dict = None

        # Define Simulation and Weather parameters
        self.Month = None        # starting month (1-12)
        self.Day = None          # starting day (1-31)
        self.nDay = None         # number of days
        self.dtSim = None        # simulation time step (s)
        self.dtWeather = None    # seconds (s)

        # HVAC system and internal laod
        self.autosize = None     # autosize HVAC (1 or 0)
        self.sensOcc = None      # Sensible heat from occupant
        self.LatFOcc = None      # Latent heat fraction from occupant (normally 0.3)
        self.RadFOcc = None      # Radiant heat fraction from occupant (normally 0.2)
        self.RadFEquip = None    # Radiant heat fraction from equipment (normally 0.5)
        self.RadFLight = None    # Radiant heat fraction from light (normally 0.7)

        # Define Urban microclimate parameters
        self.h_ubl1 = 500       # ubl height - day (m)
        self.h_ubl2 = 40       # ubl height - night (m)
        self.h_ublavg = None     # average boundary layer height [m]
        self.h_ref = None        # inversion height
        self.h_temp = None       # temperature height
        self.h_wind = None       # wind height
        self.c_circ = None       # circulation coefficient
        self.c_exch = None       # exchange coefficient
        self.maxDay = None       # max day threshhold
        self.maxNight = None     # max night threshhold
        self.windMin = None      # min wind speed (m/s)
        self.h_obs = None        # rural average obstacle height

        # Urban characteristics
        self.bldHeight = None    # average building height (m)
        self.h_mix = None        # mixing height (m)
        self.bldDensity = None   # building density (0-1)
        self.verToHor = None     # building aspect ratio
        self.charLength = None   # radius defining the urban area of study [aka. characteristic length] (m)
        self.alb_road = None     # road albedo
        self.d_road = None       # road pavement thickness
        self.sensAnth = None     # non-building sensible heat (W/m^2)
        self.latAnth = None      # non-building latent heat heat (W/m^2)
        self.emisRoad = None     # Emissivity of the road
        self.emisRoof = None     # Emissivity of the roof
        self.emisWall = None     # Emissivity of the wall
        self.emisVeg = None      # Emissivity of the wall
        self.albWall = None      # wall albedo
        self.albRural = None     # rural albedo
        self.emisRur = None      # Emissivity of the rural

        # Fraction of building typology stock
        self.bld = None         # 16x3 matrix of fraction of building type by era

        # climate Zone
        self.zone = 1

        # Vegetation parameters
        self.vegCover = None     # urban area veg coverage ratio
        self.treeCoverage = None # urban area tree coverage ratio
        self.vegStart = None     # vegetation start month
        self.vegEnd = None       # vegetation end month
        self.albVeg = None       # Vegetation albedo
        self.latGrss = None      # latent fraction of grass
        self.latTree = None      # latent fraction of tree
        self.rurVegCover = None  # rural vegetation cover

        # Define Traffic schedule
        self.SchTraffic = None

        # Define Road (Assume 0.5m of asphalt)
        self.kRoad = None       # road pavement conductivity (W/m K)
        self.cRoad = None       # road volumetric heat capacity (J/m^3 K)

        # Define optional Building characteristics
        self.albRoof = None     # roof albedo (0 - 1)
        self.vegRoof = None     # Fraction of the roofs covered in grass/shrubs (0-1)
        self.glzR = None        # Glazing Ratio
        self.hvac = None        # HVAC TYPE; 0 = Fully Conditioned (21C-24C); 1 = Mixed Mode Natural Ventilation (19C-29C + windows open >22C); 2 = Unconditioned (windows open >22C)

        # 1-D model parameters
        self.nz = None                 # number of points
        self.nz_u = None               # number of canopy levels in the vertical
        self.dz = None                 # vertical resolution
        self.wx = None                 # distance between buildings at street level in the x direction [m]
        self.wy = None                 # distance between buildings at street level in the y direction [m]
        self.Cbw = None                # fraction of building dimension and buildings distance (bx/wx or by/wy)
        self.theta_can = None          # Canyon orientation [deg]
        self.prandtl = None            # Turbulent Prandtl number
        self.schmidt = None            # Turbulent Schmidt number
        self.HVAC_atm_frac = None      # Fraction of sensible waste heat from building released into the atmosphere
        self.HVAC_street_frac = None   # Fraction of Sensible waste heat from building released into the atmosphere at street level
        self.LAD = None                # leaf area density profile [m2 m-3]
        self.h_tk = None               # Height of trunk [m]
        self.Ncloud = None             # Fraction of sky covered by cloud
        self.LAI = None                # Leaf area index (LAI) [m^2 m^-2]

    def __repr__(self):
        return "UWG: {} ".format(self.epwFileName)

    def is_near_zero(self,num,eps=1e-10):
        return abs(float(num)) < eps

    def read_epw(self):
        """Section 2 - Read EPW file
        properties:
            self.climateDataPath
            self.newPathName
            self._header    # header data
            self.epwinput   # timestep data for weather
            self.lat        # latitude
            self.lon        # longitude
            self.GMT        # GMT
            self.nSoil      # Number of soil depths
            self.Tsoil      # nSoil x 12 matrix for soil temperture (K)
            self.depth_soil # nSoil x 1 matrix for soil depth (m)
        """

        # Make dir path to epw file
        self.climateDataPath = os.path.join(self.epwDir, self.epwFileName)

        # Open epw file and feed csv data to climate_data
        try:
            climate_data = utilities.read_csv(self.climateDataPath)
        except Exception as e:
            raise Exception("Failed to read epw file! {}".format(e.message))

        # Read header lines (1 to 8) from EPW and ensure TMY2 format.
        self._header = climate_data[0:8]

        # Read weather data from EPW for each time step in weather file. (lines 8 - end)
        self.epwinput = climate_data[8:]

        # Read Lat, Long (line 1 of EPW)
        self.lat = float(self._header[0][6])
        self.lon = float(self._header[0][7])
        self.GMT = float(self._header[0][8])

        # Read in soil temperature data (assumes this is always there)
        # ref: http://bigladdersoftware.com/epx/docs/8-2/auxiliary-programs/epw-csv-format-inout.html
        soilData = self._header[3]
        self.nSoil = int(soilData[1])           # Number of ground temperature depths
        self.Tsoil = utilities.zeros(self.nSoil,12)  # nSoil x 12 matrix for soil temperture (K)
        self.depth_soil = utilities.zeros(self.nSoil,1)   # nSoil x 1 matrix for soil depth (m)

        # Read monthly data for each layer of soil from EPW file
        for i in xrange(self.nSoil):
            self.depth_soil[i][0] = float(soilData[2 + (i*16)]) # get soil depth for each nSoil
            # Monthly data
            for j in xrange(12):
                self.Tsoil[i][j] = float(soilData[6 + (i*16) + j]) + 273.15 # 12 months of soil T for specific depth

        # Set new directory path for the moprhed EPW file
        self.newPathName = os.path.join(self.destinationDir, self.destinationFileName)

    def read_input(self):
        """Section 3 - Read Input File (.m, file)
        Note: UWG_Matlab input files are xlsm, XML, .m, file.
        properties:
            self._init_param_dict   # dictionary of simulation initialization parameters

            self.sensAnth           # non-building sensible heat (W/m^2)
            self.SchTraffic         # Traffice schedule

            self.BEM                # list of BEMDef objects extracted from readDOE
            self.Sch                # list of Schedule objects extracted from readDOE

        """

        uwg_param_file_path = os.path.join(self.uwgParamDir,self.uwgParamFileName)

        if not os.path.exists(uwg_param_file_path):
            raise Exception("Param file: '{}' does not exist.".format(uwg_param_file_path))

        # Open .uwg file and feed csv data to initializeDataFile
        try:
            uwg_param_data = utilities.read_csv(uwg_param_file_path)
        except Exception as e:
            raise Exception("Failed to read .uwg file! {}".format(e.message))

        # The initialize.uwg is read with a dictionary so that users changing
        # line endings or line numbers doesn't make reading input incorrect
        self._init_param_dict = {}
        count = 0
        while  count < len(uwg_param_data):
            row = uwg_param_data[count]
            row = [row[i].replace(" ", "") for i in xrange(len(row))] # strip white spaces

            # Optional parameters might be empty so handle separately
            is_optional_parameter = (
                row != [] and \
                    (
                    row[0] == "albRoof" or \
                    row[0] == "vegRoof" or \
                    row[0] == "glzR" or \
                    row[0] == "hvac"
                    )
                )

            if row == [] or "#" in row[0]:
                count += 1
                continue
            elif row[0] == "SchTraffic":
                # SchTraffic: 3 x 24 matrix
                trafficrows = uwg_param_data[count+1:count+4]
                self._init_param_dict[row[0]] = map(lambda r: utilities.str2fl(r[:24]),trafficrows)
                count += 4
            elif row[0] == "bld":
                #bld: 17 x 3 matrix
                bldrows = uwg_param_data[count+1:count+17]
                self._init_param_dict[row[0]] = map(lambda r: utilities.str2fl(r[:3]),bldrows)
                count += 17
            elif row[0] == "LAD":
                # LAD profile
                LADrows = uwg_param_data[count+1:count+3]
                self._init_param_dict[row[0]] = map(lambda r: utilities.str2fl(r[:(len(LADrows[0])-1)]),LADrows)
                count += 3
            elif is_optional_parameter:
                self._init_param_dict[row[0]] = float(row[1]) if row[1] != "" else None
                count += 1
            else:
                self._init_param_dict[row[0]] = float(row[1])
                count += 1

        ipd = self._init_param_dict

        # Define Simulation and Weather parameters
        if self.Month is None: self.Month = ipd['Month']
        if self.Day is None: self.Day = ipd['Day']
        if self.nDay is None: self.nDay = ipd['nDay']
        if self.dtSim is None: self.dtSim = ipd['dtSim']
        if self.dtWeather is None: self.dtWeather = ipd['dtWeather']

        # HVAC system and internal laod
        if self.autosize is None: self.autosize = ipd['autosize']
        if self.sensOcc is None: self.sensOcc = ipd['sensOcc']
        if self.LatFOcc is None: self.LatFOcc = ipd['LatFOcc']
        if self.RadFOcc is None: self.RadFOcc = ipd['RadFOcc']
        if self.RadFEquip is None: self.RadFEquip = ipd['RadFEquip']
        if self.RadFLight is None: self.RadFLight = ipd['RadFLight']

        # Define Urban microclimate parameters
        if self.h_ubl1 is None: self.h_ubl1 = ipd['h_ubl1']
        if self.h_ubl2 is None: self.h_ubl2 = ipd['h_ubl2']
        if self.h_ublavg is None: self.h_ublavg = ipd['h_ublavg']
        if self.h_ref is None: self.h_ref = ipd['h_ref']
        if self.h_temp is None: self.h_temp = ipd['h_temp']
        if self.h_wind is None: self.h_wind = ipd['h_wind']
        if self.c_circ is None: self.c_circ = ipd['c_circ']
        if self.c_exch is None: self.c_exch = ipd['c_exch']
        if self.maxDay is None: self.maxDay = ipd['maxDay']
        if self.maxNight is None: self.maxNight = ipd['maxNight']
        if self.windMin is None: self.windMin = ipd['windMin']
        if self.h_obs is None: self.h_obs = ipd['h_obs']

        # Urban characteristics
        if self.bldHeight is None: self.bldHeight = ipd['bldHeight']
        if self.h_mix is None: self.h_mix = ipd['h_mix']
        if self.bldDensity is None: self.bldDensity = ipd['bldDensity']
        if self.verToHor is None: self.verToHor = ipd['verToHor']
        if self.charLength is None: self.charLength = ipd['charLength']
        if self.alb_road is None: self.alb_road = ipd['albRoad']
        if self.d_road is None: self.d_road = ipd['dRoad']
        if self.sensAnth is None: self.sensAnth = ipd['sensAnth']
        if self.latAnth is None: self.latAnth = ipd['latAnth']
        if self.emisRoad is None: self.emisRoad = ipd['emisRoad']
        if self.emisRoof is None: self.emisRoof = ipd['emisRoof']
        if self.emisWall is None: self.emisWall = ipd['emisWall']
        if self.emisVeg is None: self.emisVeg = ipd['emisVeg']
        if self.albWall is None: self.albWall = ipd['albWall']
        if self.albRural is None: self.albRural = ipd['albRural']
        if self.emisRur is None: self.emisRur = ipd['emisRur']


        # climate Zone
        if self.zone is None: self.zone = ipd['zone']

        # Vegetation parameters
        if self.vegCover is None: self.vegCover = ipd['vegCover']
        if self.treeCoverage is None: self.treeCoverage = ipd['treeCoverage']
        if self.vegStart is None: self.vegStart = ipd['vegStart']
        if self.vegEnd is None: self.vegEnd = ipd['vegEnd']
        if self.albVeg is None: self.albVeg = ipd['albVeg']
        if self.latGrss is None: self.latGrss = ipd['latGrss']
        if self.latTree is None: self.latTree = ipd['latTree']
        if self.rurVegCover is None: self.rurVegCover = ipd['rurVegCover']

        # Define Traffic schedule
        if self.SchTraffic is None: self.SchTraffic = ipd['SchTraffic']

        # Define Road (Assume 0.5m of asphalt)
        if self.kRoad is None: self.kRoad = ipd['kRoad']
        if self.cRoad is None: self.cRoad = ipd['cRoad']

        # Building stock fraction
        if self.bld is None: self.bld = ipd['bld']

        # Optional parameters
        if self.albRoof is None: self.albRoof = ipd['albRoof']
        if self.vegRoof is None: self.vegRoof = ipd['vegRoof']
        if self.glzR is None: self.glzR = ipd['glzR']
        if self.hvac is None: self.hvac = ipd['hvac']

        # 1-D model parameters
        if self.nz is None: self.nz = int(ipd['nz'])
        if self.nz_u is None: self.nz_u = int(ipd['nz_u'])
        if self.dz is None: self.dz = int(ipd['dz'])
        if self.wx is None: self.wx = ipd['wx']
        if self.wy is None: self.wy = ipd['wy']
        if self.Cbw is None: self.Cbw = ipd['Cbw']
        if self.theta_can is None: self.theta_can = ipd['theta_can']
        if self.prandtl is None: self.prandtl = ipd['prandtl']
        if self.schmidt is None: self.schmidt = ipd['schmidt']
        if self.HVAC_atm_frac is None: self.HVAC_atm_frac = ipd['HVAC_atm_frac']
        if self.HVAC_street_frac is None: self.HVAC_street_frac = ipd['HVAC_street_frac']
        if self.LAD is None: self.LAD = ipd['LAD']
        if self.h_tk is None: self.h_tk = ipd['h_tk']
        if self.Ncloud is None: self.Ncloud = ipd['Ncloud']
        if self.LAI is None: self.LAI = ipd['LAI']


    def set_input(self):
        """ Set inputs from .uwg input file if not already defined, the check if all
        the required input parameters are there.
        """

        # If a uwgParamFileName is set, then read inputs from .uwg file.
        # User-defined class properties will override the inputs from the .uwg file.
        if self.uwgParamFileName is not None:
            print "\nReading uwg file input."
            self.read_input()
        else:
            print "\nNo .uwg file input."

        # Required parameters
        is_defined = (type(self.Month) == float or type(self.Month) == int) and \
            (type(self.Day) == float or type(self.Day) == int) and \
            (type(self.nDay) == float or type(self.nDay) == int) and \
            type(self.dtSim) == float and type(self.dtWeather) == float and \
            (type(self.autosize) == float or type(self.autosize) == int) and \
            type(self.sensOcc) == float and type(self.LatFOcc) == float and \
            type(self.RadFOcc) == float and type(self.RadFEquip) == float and \
            type(self.h_temp) == float and type(self.h_wind) == float and \
            type(self.c_circ) == float and type(self.c_exch) == float and \
            type(self.maxDay) == float and type(self.maxNight) == float and \
            type(self.windMin) == float and type(self.h_obs) == float and \
            type(self.bldHeight) == float and type(self.h_mix) == float and \
            type(self.bldDensity) == float and type(self.verToHor) == float and \
            type(self.charLength) == float and type(self.alb_road) == float and \
            type(self.d_road) == float and type(self.sensAnth) == float and \
            type(self.latAnth) == float and type(self.bld) == type([]) and \
            self.is_near_zero(len(self.bld)-16.0) and \
            (type(self.zone) == float or type(self.zone) == int) and \
            (type(self.vegStart) == float or type(self.vegStart) == int) and \
            (type(self.vegEnd) == float or type(self.vegEnd) == int) and \
            type(self.vegCover) == float and type(self.treeCoverage) == float and \
            type(self.albVeg) == float and type(self.latGrss) == float and \
            type(self.latTree) == float and type(self.rurVegCover) == float and \
            type(self.kRoad) == float and type(self.cRoad) == float and \
            type(self.SchTraffic) == type([]) and self.is_near_zero(len(self.SchTraffic)-3.0)

        if not is_defined:
            raise Exception("The required parameters have not been defined correctly. Check input parameters and try again.")

        # Modify zone to be used as python index
        self.zone = int(self.zone)-1

    def instantiate_input(self):
        """Section 4 - Create UWG objects from input parameters

            self.simTime            # simulation time parameter obj
            self.weather            # weather obj for simulation time period
            self.forcIP             # Forcing obj
            self.forc               # Empty forcing obj
            self.geoParam           # geographic parameters obj
            self.RSM                # Rural site & vertical diffusion model obj
            self.USM                # Urban site & vertical diffusion model obj
            self.UCM                # Urban canopy model obj
            self.UBL                # Urban boundary layer model

            self.road               # urban road element
            self.rural              # rural road element

            self.soilindex1         # soil index for urban rsoad depth
            self.soilindex2         # soil index for rural road depth

            self.BEM                # list of BEMDef objects
            self.Sch                # list of Schedule objects
        """

        climate_file_path = os.path.join(self.epwDir, self.epwFileName)
        self.simTime = SimParam(self.dtSim,self.dtWeather,self.Month,self.Day,self.nDay)  # simulation time parametrs
        self.weather = Weather(climate_file_path,self.simTime.timeInitial,self.simTime.timeFinal) # weather file data for simulation time period
        self.forcIP = Forcing(self.weather.staTemp,self.weather) # initialized Forcing class
        self.forc = Forcing() # empty forcing class
        # Initialize geographic Param and Urban Boundary Layer Objects
        nightStart = 18.        # arbitrary values for begin/end hour for night setpoint
        nightEnd = 8.
        maxdx = 250.;            # max dx (m)

        self.geoParam = Param(self.h_ubl1,self.h_ubl2,self.h_ref,self.h_temp,self.h_wind,self.c_circ,\
            self.maxDay,self.maxNight,self.latTree,self.latGrss,self.albVeg,self.vegStart,self.vegEnd,\
            nightStart,nightEnd,self.windMin,self.WGMAX,self.c_exch,maxdx,self.G,self.CP,self.VK,self.R,\
            self.RV,self.LV,math.pi,self.SIGMA,self.WATERDENS,self.LVTT,self.TT,self.ESTT,self.CL,\
            self.CPV,self.B, self.CM, self.COLBURN)
        #!!!!! UBL IS REMOVED !!!!!
        # self.UBL = UBLDef('C',self.charLength, self.weather.staTemp[0], maxdx, self.geoParam.dayBLHeight, self.geoParam.nightBLHeight)

        # Defining road
        emis = 0.93
        asphalt = Material(self.kRoad,self.cRoad,'asphalt')
        road_T_init = 293.
        road_horizontal = 1
        road_veg_coverage = min(self.vegCover/(1-self.bldDensity),1.) # fraction of surface vegetation coverage

        # define road layers
        road_layer_num = int(math.ceil(self.d_road/0.05))
        thickness_vector = map(lambda r: 0.05, range(road_layer_num)) # 0.5/0.05 ~ 10 x 1 matrix of 0.05 thickness
        material_vector = map(lambda n: asphalt, range(road_layer_num))

        self.road = Element(self.alb_road,emis,thickness_vector,material_vector,road_veg_coverage,\
            road_T_init,road_horizontal,name="urban_road")

        self.rural = copy.deepcopy(self.road)
        self.rural.vegCoverage = self.rurVegCover
        self.rural._name = "rural_road"

        # Define BEM for each DOE type (read the fraction)
        if not os.path.exists(self.readDOE_file_path):
            raise Exception("readDOE.pkl file: '{}' does not exist.".format(readDOE_file_path))

        readDOE_file = open(self.readDOE_file_path, 'rb') # open pickle file in binary form
        refDOE = cPickle.load(readDOE_file)
        refBEM = cPickle.load(readDOE_file)
        refSchedule = cPickle.load(readDOE_file)
        readDOE_file.close()

        # Define building energy models
        k = 0
        r_glaze = 0             # Glazing ratio for total building stock
        SHGC = 0                # SHGC addition for total building stock
        alb_wall = 0            # albedo wall addition for total building stock
        h_floor = 3.05          # average floor height

        total_urban_bld_area = math.pow(self.charLength,2)*self.bldDensity*self.bldHeight/h_floor  # total building floor area
        area_matrix = utilities.zeros(16,3)

        self.BEM = []           # list of BEMDef objects
        self.Sch = []           # list of Schedule objects

        for i in xrange(16):    # 16 building types
            for j in xrange(3): # 3 built eras
                if self.bld[i][j] > 0.:
                    # Add to BEM list
                    self.BEM.append(refBEM[i][j][self.zone])
                    #print(self.BEM)
                    self.BEM[k].frac = self.bld[i][j]
                    self.BEM[k].fl_area = self.bld[i][j] * total_urban_bld_area

                    # Overwrite with optional parameters if provided
                    if self.glzR:
                        self.BEM[k].building.glazingRatio = self.glzR
                    if self.albRoof:
                        self.BEM[k].roof.albedo = self.albRoof
                    if self.vegRoof:
                        self.BEM[k].roof.vegCoverage = self.vegRoof

                    # Keep track of total urban r_glaze, SHGC, and alb_wall for UCM model
                    r_glaze = r_glaze + self.BEM[k].frac * self.BEM[k].building.glazingRatio ##
                    SHGC = SHGC + self.BEM[k].frac * self.BEM[k].building.shgc
                    alb_wall = alb_wall + self.BEM[k].frac * self.BEM[k].wall.albedo

                    # Add to schedule list
                    self.Sch.append(refSchedule[i][j][self.zone])
                    k += 1

        self.BEM[0].wall.emissivity = self.emisWall
        self.road.emissivity = self.emisRoad
        self.BEM[0].roof.emissivity = self.emisRoof


        #self.UCM = UCMDef(self.bldHeight,self.bldDensity,self.verToHor,self.treeCoverage,self.sensAnth,self.latAnth,T_init,H_init,\
        #self.weather.staUmod[0],self.geoParam,r_glaze,SHGC,alb_wall,self.road)
        #self.UCM.h_mix = self.h_mix

        # ==============================================================================================================
        # 1-D Model (Sec.1 Start): define constant parameters and Initialize variables
        # ==============================================================================================================
        # Note bx and by mst be exactly the same; this requires wx and wy to be exactly the same
        self.bx = self.Cbw*self.wx
        self.by = self.Cbw*self.wy
        self.verToHor = self.bldHeight/(self.bx+self.wx)
        self.bldDensity = self.bx/(self.bx+self.wx)
        self.lambdap = self.bldDensity**2
        self.lambdaf = self.bldDensity*self.verToHor
        # define probabilities
        # "pb(z)" Probability that a building has a height greater or equal to z (In the current version of the model a simple
        # canyon is considered. So, "pb" is one within the canyon and zero above the canyon.)
        # "ss(z)" Probability that a building has a height equal to z (In the current version of the model a simple
        # canyon is considered so this probability is one at building average height h mean (nz_u) but zero elsewhere.)
        self.pb = numpy.zeros(self.nz + 1)
        self.ss = numpy.zeros(self.nz + 1)
        self.ss[self.nz_u] = 1
        for i in range(0, self.nz + 1):
            if i <= self.nz_u:
                self.pb[i] = 1
            else:
                self.pb[i] = 0

        # Generate mesh for the column (1-D) model: grid resolution is kept constant over the domain [m]
        self.z = numpy.linspace(0, self.nz * self.dz, self.nz + 1)

        # vol: volume fraction of air in each urban unit cell
        self.vol = numpy.zeros(self.nz)
        # sf: fraction of air at the interface between cells (sf) [please verify, not sure what we are doing here!]
        self.sf = numpy.zeros(self.nz)

        for i in range(0, self.nz):
            self.vol[i] = 1-self.lambdap*self.pb[i]  # ???? pb[i+1]
            # "sf" is calculated from Nazarian's code (https://github.com/nenazarian/MLUCM/blob/master/Column_Model/column_lkPro.f90)
            self.sf[i] = 1-self.lambdap*self.ss[i]

        # Coefficient for the destruction of turbulent dissipation rate
        self.Ceps = 1 / 1.14

        # Coefficient used in the equation of diffusion coefficient
        self.Ck = 0.4

        # self.Ck = self.var_sens    # @@@@ Sensitivity Analysis @@@@@@

        # Coefficient which will be used to determine length scales
        self.Cmu = 0.09

        # Calculate section drag coefficient (Cdrag) due to buildings
        # Calculate "Cdrag" based on Krayenhoff's code
        DragLength = Drag_Length(self.nz, self.nz_u, self.z, self.lambdap, self.lambdaf, self.bldHeight, self.Ceps, self.Ck, self.Cmu, self.pb)
        self.dlk = DragLength.Length_Scale()[0]
        self.dls = DragLength.Length_Scale()[1]
        self.Cdrag = DragLength.Drag_Coef()

        # Calculate "Cdrag" based on Nazarian's code (https://github.com/nenazarian/MLUCM/blob/master/Column_Model/column_lkPro.f90)
        # Drag coefficients in x and y directions are similar
        #DragLength = Drag_Length(self.nz,self.z,self.lambdap,self.lambdaf,self.bldHeight,self.bx,self.by,self.wx,self.wy,self.Ceps, self.Ck,self.Cmu)
        #self.dlk, self.dls = DragLength.Length_Scale()
        #self.Cdragx, self.Cdragy = DragLength.Drag_Coef()
        #self.Cdrag = self.Cdragx

        # Reference site class (also include VDM)
        self.RSM = RSMDef(self.lat,self.lon,self.GMT,self.h_obs,self.weather.staTemp[0],self.weather.staPres[0],self.geoParam,self.z,self.dz,self.nz)
        self.USM = RSMDef(self.lat,self.lon,self.GMT,self.bldHeight/10.,self.weather.staTemp[0],self.weather.staPres[0], self.geoParam, self.z, self.dz,self.nz)

        T_init = self.weather.staTemp[0]
        H_init = self.weather.staHum[0]

        # define variables including x and y components of wind speed, turbulent kinetic energy, potential temperature
        # specific humidity and reference temperature
        self.vx = numpy.zeros(self.nz)
        self.vy = numpy.zeros(self.nz)
        self.tke = numpy.zeros(self.nz)
        self.th = numpy.zeros(self.nz)
        self.qn = numpy.zeros(self.nz)
        self.th0 = numpy.zeros(self.nz)
        # Initialize variables
        for i in range(0, self.nz):
            self.vx[i] = 0.1             # x component of horizontal wind speed [m s^-1]
            self.vy[i] = 0.1             # y component of horizontal wind speed [m s^-1]
            self.tke[i] = 0.15           # Turbulent kinetic energy [m^2 s^-2]
            self.th[i] = 300             # Potential temperature [K]
            self.th0[i] = 300            # Reference potential temperature [K]
            self.qn[i] = 0.02            # Specific humidity [kgv kga^-1]
        self.tveg_tmp = 290              # vegetation temperature [K]

        # Create Leaf area density (LAD) [m^2 m^-3] function by interpolating within the data.
        # Tree height should be equal or less than average building height.
        # Vegetation only considered in canyon column
        self.h_LAD = self.LAD[0]
        LAD = self.LAD[1]
        self.f_LAD = interp1d(self.h_LAD, LAD)

        # Initialize turbulent diffusion coefficient (Km)
        # "Km" will be used in vertical diffusion model (VDM) to calculate vertical turbulent flux in urban area
        # Initial Guess for bulk Richardson number (It will be recalculated in column (1-D) model)
        Ri_b = 1
        # Calculate turbulent diffusion coefficent [m^2 s^-1] based on stability (i.e. bulk Richardson number) by optimizing mixing length
        TurbDiff = CdTurb(self.nz, self.Ck, self.tke, self.dlk, 0, Ri_b, 0)
        self.Km = TurbDiff.TurbCoeff()

        # ==============================================================================================================
        # 1-D Model (Sec.1 End)
        # ==============================================================================================================

        self.UCM = UCMDef(self.bldHeight, self.bldDensity, self.verToHor, self.treeCoverage, self.sensAnth,self.latAnth, T_init, H_init,self.weather.staUmod[0], self.geoParam, r_glaze, SHGC, alb_wall, self.road)
        self.UCM.h_mix = self.h_mix
        # Initial Value for Roof, Road and wall Temperatures of the canyon [K]
        self.UCM.roofTemp = 292
        self.UCM.roadTemp = 292
        self.UCM.wallTemp = 292

        # Define Road Element & buffer to match ground temperature depth
        roadMat, newthickness = procMat(self.road,self.MAXTHICKNESS,self.MINTHICKNESS)

        for i in xrange(self.nSoil):
            # if soil depth is greater than the thickness of the road
            # we add new slices of soil at max thickness until road is greater or equal

            is_soildepth_equal = self.is_near_zero(self.depth_soil[i][0] - sum(newthickness),1e-15)

            if is_soildepth_equal or (self.depth_soil[i][0] > sum(newthickness)):
                while self.depth_soil[i][0] > sum(newthickness):
                    newthickness.append(self.MAXTHICKNESS)
                    roadMat.append(self.SOIL)
                self.soilindex1 = i
                break

        self.road = Element(self.road.albedo, self.road.emissivity, newthickness, roadMat,\
            self.road.vegCoverage, self.road.layerTemp[0], self.road.horizontal, self.road._name)

        # Define Rural Element
        ruralMat, newthickness = procMat(self.rural,self.MAXTHICKNESS,self.MINTHICKNESS)

        for i in xrange(self.nSoil):
            # if soil depth is greater than the thickness of the road
            # we add new slices of soil at max thickness until road is greater or equal

            is_soildepth_equal = self.is_near_zero(self.depth_soil[i][0] - sum(newthickness),1e-15)

            if is_soildepth_equal or (self.depth_soil[i][0] > sum(newthickness)):
                while self.depth_soil[i][0] > sum(newthickness):
                    newthickness.append(self.MAXTHICKNESS)
                    ruralMat.append(self.SOIL)

                self.soilindex2 = i
                break

        self.rural = Element(self.rural.albedo, self.rural.emissivity, newthickness,\
            ruralMat,self.rural.vegCoverage,self.rural.layerTemp[0],self.rural.horizontal, self.rural._name)
        self.rural.albedo = self.albRural
        self.rural.emissivity = self.emisRur
    def hvac_autosize(self):
        """ Section 6 - HVAC Autosizing (unlimited cooling & heating) """

        for i in xrange(len(self.BEM)):
            if self.is_near_zero(self.autosize)==False:
                self.BEM[i].building.coolCap = 9999.
                self.BEM[i].building.heatCap = 9999.

    def simulate(self):
        """ Section 7 - UWG main section

            self.N                  # Total hours in simulation
            self.ph                 # per hour
            self.dayType            # 3=Sun, 2=Sat, 1=Weekday
            self.ceil_time_step     # simulation timestep (dt) fitted to weather file timestep

            # Output of object instance vector
            self.WeatherData        # Nx1 vector of forc instance
            self.UCMData            # Nx1 vector of UCM instance
            self.UBLData            # Nx1 vector of UBL instance
            self.RSMData            # Nx1 vector of RSM instance
            self.USMData            # Nx1 vector of USM instance
        """

        self.N = int(self.simTime.days * 24)       # total number of hours in simulation
        n = 0                                      # weather time step counter
        self.ph = self.simTime.dt/3600.            # dt (simulation time step) in hours

        # Data dump variables
        time = range(self.N)

        self.WeatherData = [None for x in xrange(self.N)]
        self.UCMData = [None for x in xrange(self.N)]
        # !!!!! UBL IS REMOVED !!!!!
        # self.UBLData = [None for x in xrange(self.N)]
        self.RSMData = [None for x in xrange(self.N)]
        self.USMData = [None for x in xrange(self.N)]

        print '\nSimulating new temperature and humidity values for {} days from {}/{}.\n'.format(
            int(self.nDay), int(self.Month), int(self.Day))
        self.logger.info("Start simulation")

        # Start progress bar at zero
        progress_bar.print_progress(0, 100.0, prefix = "Progress:", bar_length = 25)

        # Define variables will be used in coupling between two models and post-processing
        # "S_can" will represent wind speed within the canyon
        S_can = numpy.zeros(self.simTime.nt-1)

        # Variable to save temperature from previous iteration to check convergence criterion
        MeanTempOld_1D = self.th[0]
        MeanTempOld_UWG = self.UCM.canTemp

        # Terms in solar radiation model (for record keeping only)
        Azimuth = numpy.zeros(self.simTime.nt)
        Zenith = numpy.zeros(self.simTime.nt)
        vf_RoadSky = numpy.zeros(self.simTime.nt)
        vf_WallRoad = numpy.zeros(self.simTime.nt)
        vf_SkyWall = numpy.zeros(self.simTime.nt)
        vf_WallWall = numpy.zeros(self.simTime.nt)
        vf_RoadWAll = numpy.zeros(self.simTime.nt)
        vf_SkyTree = numpy.zeros(self.simTime.nt)
        vf_RoadTree = numpy.zeros(self.simTime.nt)
        vf_WallTree = numpy.zeros(self.simTime.nt)
        tau_SkyRoad = numpy.zeros(self.simTime.nt)
        tau_WallRoad = numpy.zeros(self.simTime.nt)
        tau_SkyWall = numpy.zeros(self.simTime.nt)
        tau_WallWall = numpy.zeros(self.simTime.nt)
        SDir_Trees = numpy.zeros(self.simTime.nt)
        SDir_TreesTransm = numpy.zeros(self.simTime.nt)
        SDir_TreesReflec = numpy.zeros(self.simTime.nt)
        SDir_TreesAbs = numpy.zeros(self.simTime.nt)
        SDir_WallA = numpy.zeros(self.simTime.nt)
        SDir_WallB = numpy.zeros(self.simTime.nt)
        SDir_Road = numpy.zeros(self.simTime.nt)
        SDif_Road = numpy.zeros(self.simTime.nt)
        SDif_Wall = numpy.zeros(self.simTime.nt)
        SDif_Trees = numpy.zeros(self.simTime.nt)
        S_Sky = numpy.zeros(self.simTime.nt)             # Total shortwave radiation absorbed by sky
        S_Road = numpy.zeros(self.simTime.nt)            # Total shortwave radiation absorbed by road
        S_Garden = numpy.zeros(self.simTime.nt)
        S_WallA = numpy.zeros(self.simTime.nt)           # Total shortwave radiation absorbed by wall A
        S_WallB = numpy.zeros(self.simTime.nt)           # Total shortwave radiation absorbed by wall B
        S_Trees = numpy.zeros(self.simTime.nt)           # Total shortwave radiation absorbed by tree
        L_atm_emt = numpy.zeros(self.simTime.nt)
        L_road_emt = numpy.zeros(self.simTime.nt)
        L_wall_emt = numpy.zeros(self.simTime.nt)
        L_roof_emt = numpy.zeros(self.simTime.nt)
        L_trees_emt = numpy.zeros(self.simTime.nt)
        L_tt = numpy.zeros(self.simTime.nt)
        L_tr = numpy.zeros(self.simTime.nt)
        L_tw = numpy.zeros(self.simTime.nt)
        L_wall_abs = numpy.zeros(self.simTime.nt)         # Long wave radiation received by wall
        L_road_abs = numpy.zeros(self.simTime.nt)         # Long wave radiation received by road
        L_trees_abs = numpy.zeros(self.simTime.nt)        # Long wave radiation received by tree
        L_wall_net = numpy.zeros(self.simTime.nt)         # Net long wave radiation received by walls
        L_road_net = numpy.zeros(self.simTime.nt)         # Net long wave radiation received by road
        L_trees_net = numpy.zeros(self.simTime.nt)        # Net long wave radiation received by tree
        L_tot_net = numpy.zeros(self.simTime.nt)          # Total long wave radiation in urban area

        # define variables used to store data
        iO = 0

        Output_TimInd = numpy.arange(8640,int(self.N*(3600/self.dtSim)),int(3600/self.dtSim))


        U2w = numpy.zeros((len(Output_TimInd), self.nz))
        V2w = numpy.zeros((len(Output_TimInd), self.nz))
        Tu2w = numpy.zeros((len(Output_TimInd), self.nz))
        Tr2w = numpy.zeros((len(Output_TimInd), self.nz))
        TempDiu_TANAB = numpy.zeros((len(Output_TimInd), self.nz))  # Urban temperature profile [K]
        UDiu_TANAB = numpy.zeros((len(Output_TimInd), self.nz))     # x component of horizontal wind speed [m s^-1]
        VDiu_TANAB = numpy.zeros((len(Output_TimInd), self.nz))     # y component of horizontal wind speed [m s^-1]
        TKEDiu_TANAB = numpy.zeros((len(Output_TimInd), self.nz))   # Turbulent kinetic energy (TKE) [m^2 s^-2]
        qDiu_TANAB = numpy.zeros((len(Output_TimInd), self.nz))     # Specific humidity [kgv kga^-1]
        TempDiu_Rural = numpy.zeros((len(Output_TimInd), self.nz))  # Rural temperature [K]
        WindDiu_Rural = numpy.zeros((len(Output_TimInd), self.nz))  # Rural wind speed [m s^-1]
        KtDiu_Rural = numpy.zeros((len(Output_TimInd), self.nz+1))  # Rural diffusion coefficient [m^2 s^-1]
        lm_1Diu_Rural = numpy.zeros((len(Output_TimInd), self.nz))  # term1 of length scale in rural area [m]
        lm_2Diu_Rural = numpy.zeros((len(Output_TimInd), self.nz))  # term2 of length scale in rural area [m]

        # Variables used to plot quantity of interest
        varProf = numpy.zeros((len(Output_TimInd), self.nz))
        varTimeSeries1 = numpy.zeros(self.simTime.nt-1)
        varTimeSeries2 = numpy.zeros(self.simTime.nt-1)
        Pv = numpy.zeros(self.simTime.nt-1)
        Psat = numpy.zeros(self.simTime.nt-1)
        gamma = numpy.zeros(self.simTime.nt - 1)
        timeseries = [x for x in range(0,self.simTime.nt-1)]

        for it in range(1,self.simTime.nt,1):# for every simulation time-step (i.e 5 min) defined by uwg

            # Set initial value for error to start iteration in each time step
            canTemp_Err_1D = 20
            canTemp_Err_UWG = 20
            while (canTemp_Err_1D > 0.001 and canTemp_Err_UWG > 0.001):

              # Update water temperature (estimated)
              if self.is_near_zero(self.nSoil):
                  self.forc.deepTemp = sum(self.forcIP.temp)/float(len(self.forcIP.temp))             # for BUBBLE/CAPITOUL/Singapore only
                  self.forc.waterTemp = sum(self.forcIP.temp)/float(len(self.forcIP.temp)) - 10.      # for BUBBLE/CAPITOUL/Singapore only
              else:
                  self.forc.deepTemp = self.Tsoil[self.soilindex1][self.simTime.month-1] #soil temperature by depth, by month
                  self.forc.waterTemp = self.Tsoil[2][self.simTime.month-1]

              # There's probably a better way to update the weather...
              self.simTime.UpdateDate()

              self.logger.info("\n{0} m={1}, d={2}, h={3}, s={4}".format(__name__, self.simTime.month, self.simTime.day, self.simTime.secDay/3600., self.simTime.secDay))

              self.ceil_time_step = int(math.ceil(it * self.ph))-1  # simulation time increment raised to weather time step
                                                                   # minus one to be consistent with forcIP list index
              # Updating forcing instance
              self.forc.infra = self.forcIP.infra[self.ceil_time_step]        # horizontal Infrared Radiation Intensity (W m-2)
              self.forc.wind = max(self.forcIP.wind[self.ceil_time_step], self.geoParam.windMin) # wind speed (m s-1)
              self.forc.uDir = self.forcIP.uDir[self.ceil_time_step]          # wind direction
              self.forc.hum = self.forcIP.hum[self.ceil_time_step]            # specific humidty (kg kg-1)
              self.forc.pres = self.forcIP.pres[self.ceil_time_step]          # Pressure (Pa)
              self.forc.temp = self.forcIP.temp[self.ceil_time_step]          # air temperature (C)
              self.forc.rHum = self.forcIP.rHum[self.ceil_time_step]          # Relative humidity (%)
              self.forc.prec = self.forcIP.prec[self.ceil_time_step]          # Precipitation (mm h-1)
              self.forc.dif = self.forcIP.dif[self.ceil_time_step]            # horizontal solar diffuse radiation (W m-2)
              self.forc.dir = self.forcIP.dir[self.ceil_time_step]            # normal solar direct radiation (W m-2)
              self.UCM.canHum = copy.copy(self.forc.hum)                      # Canyon humidity (absolute) same as rural

              #RoofTemp[it] = self.UCM.roofTemp
              # Update solar flux
              self.solar = SolarCalcs(self.UCM, self.BEM, self.simTime, self.RSM, self.forc, self.geoParam, self.rural)
              self.rural, self.UCM, self.BEM = self.solar.solarcalcs()

              # ========================================================================================================
              # 1-D Model (Sec.2 Start)
              # ========================================================================================================
              # Get relevant data for solar calculation using the old solar model in UWG
              self.solar.solarangles()

              # Perform solar calculation based on models developed by (Redon et al., 2017) and (Lee and Park, 2008)
              SolarCal2 = SolarModel2(self.forc, self.solar, self.UCM,self.BEM,self.road, self.nz, self.nz_u, self.dz, self.bldHeight,
                                      self.wy, self.tveg_tmp, self.f_LAD, self.h_LAD, self.h_tk,self.albVeg,self.Ncloud,self.LAI,self.var_sens,self.emisVeg,self.albWall)

              Azimuth[it],Zenith[it],vf_RoadSky[it],vf_WallRoad[it],vf_SkyWall[it],vf_WallWall[it],vf_RoadWAll[it], \
              vf_SkyTree[it],vf_RoadTree[it],vf_WallTree[it],tau_SkyRoad[it],tau_WallRoad[it],tau_SkyWall[it],tau_WallWall[it],\
              SDir_Trees[it],SDir_TreesTransm[it],SDir_TreesReflec[it],SDir_TreesAbs[it],SDir_WallA[it],SDir_WallB[it],SDir_Road[it],\
              SDif_Road[it],SDif_Wall[it],SDif_Trees[it],S_Sky[it],S_Road[it],S_Garden[it],S_WallA[it],S_WallB[it],S_Trees[it],\
              L_atm_emt[it],L_road_emt[it],L_wall_emt[it],L_roof_emt[it],L_trees_emt[it],L_tt[it],L_tr[it],L_tw[it],L_wall_abs[it],\
              L_road_abs[it],L_trees_abs[it],L_wall_net[it],L_road_net[it],L_trees_net[it],L_tot_net[it] = SolarCal2.SolCalRedon()

              # Update terms in UWG which are affected by solar radiation model.These terms are part of UCM and BEM model.
              self.UCM.road.solRec = (SDir_Road[it] + SDif_Road[it])                                       # Solar received by road [W m^-2]
              self.UCM.road.solAbs = S_Road[it]                                                            # Solar absorbed by road [W m^-2]
              for jBEM in xrange(len(self.BEM)):
                  self.BEM[jBEM].wall.solRec = (SDir_WallA[it] + SDif_Wall[it])                            # Solar received by wall [W m^-2]
                  self.BEM[jBEM].roof.solRec = self.rural.solRec                                           # Solar received by roof [W m^-2]
                  self.BEM[jBEM].wall.solAbs = S_WallA[it]                                                 # Solar absorbed by wall [W m^-2]
                  self.BEM[jBEM].roof.solAbs = (1.-self.BEM[jBEM].roof.albedo)*self.BEM[jBEM].roof.solRec  # Solar absorbed by roof [W m^-2]

              # Calculate sensible and latent heat flux due to vegetations including trees and grass [W m^-2]
              self.UCM.treeSensHeat = (1 - self.geoParam.treeFLat) * (self.UCM.road.solRec + S_Trees[it])
              self.UCM.treeLatHeat = self.geoParam.treeFLat * (self.UCM.road.solRec + S_Trees[it])

              # ========================================================================================================
              # 1-D Model (Sec.2 End)
              # ========================================================================================================

              # Update building & traffic schedule
              # Assign day type (1 = weekday, 2 = sat, 3 = sun/other)
              if self.is_near_zero(self.simTime.julian % 7):
                  self.dayType = 3                                        # Sunday
              elif self.is_near_zero(self.simTime.julian % 7 - 6.):
                  self.dayType = 2                                        # Saturday
              else:
                  self.dayType = 1                                        # Weekday

              # Update anthropogenic heat load for each hour [W m^-2] ????? (building & UCM)
              self.UCM.sensAnthrop = self.sensAnth * (self.SchTraffic[self.dayType-1][self.simTime.hourDay])

              # Update the energy components for building types defined in initialize.uwg
              for i in xrange(len(self.BEM)):
                  # Set point temperature [K]
                  # Add from temperature schedule for cooling
                  self.BEM[i].building.coolSetpointDay   = self.Sch[i].Cool[self.dayType-1][self.simTime.hourDay] + 273.15
                  self.BEM[i].building.coolSetpointNight = self.BEM[i].building.coolSetpointDay
                  # Add from temperature schedule for heating
                  self.BEM[i].building.heatSetpointDay   = self.Sch[i].Heat[self.dayType-1][self.simTime.hourDay] + 273.15
                  self.BEM[i].building.heatSetpointNight = self.BEM[i].building.heatSetpointDay

                  # Internal Heat Load Schedule per unit floor area [W m^-2]
                  # Electricity consumption per unit floor area [W m^-2] = max for electrical plug process * electricity fraction for the day
                  self.BEM[i].Elec  = self.Sch[i].Qelec * self.Sch[i].Elec[self.dayType-1][self.simTime.hourDay]
                  # Lighting per unit floor area [W m^-2] = max for light * light fraction for the day
                  self.BEM[i].Light = self.Sch[i].Qlight * self.Sch[i].Light[self.dayType-1][self.simTime.hourDay]
                  # Number of occupants x occ fraction for day  ??????
                  self.BEM[i].Nocc  = self.Sch[i].Nocc * self.Sch[i].Occ[self.dayType-1][self.simTime.hourDay]
                  # Sensible Q occupant * fraction occupant sensible Q * number of occupants  ??????
                  self.BEM[i].Qocc  = self.sensOcc * (1 - self.LatFOcc) * self.BEM[i].Nocc

                  # SWH and ventilation schedule
                  # Solar water heating per unit floor area [W m^-2] = Peak Service Hot Water per unit floor [kg hr^-1 m^-2] * SWH fraction for the day
                  self.BEM[i].SWH = self.Sch[i].Vswh * self.Sch[i].SWH[self.dayType-1][self.simTime.hourDay]
                  # Ventilation rate per unit floor area [m^3 s^-1 m^-2]
                  self.BEM[i].building.vent = self.Sch[i].Vent
                  # Gas consumption per unit floor area [W m^-2] = max for gas * Gas fraction for the day
                  self.BEM[i].Gas = self.Sch[i].Qgas * self.Sch[i].Gas[self.dayType-1][self.simTime.hourDay]

                  # This is quite messy, should update
                  # Update internal heat and corresponding fractional loads per unit floor area [W m^-2]
                  intHeat = self.BEM[i].Light + self.BEM[i].Elec + self.BEM[i].Qocc
                  self.BEM[i].building.intHeatDay = intHeat
                  self.BEM[i].building.intHeatNight = intHeat
                  # Fraction of radiant heat from light and equipment of whole internal heat per unit floor area [W m^-2]
                  self.BEM[i].building.intHeatFRad = (self.RadFLight * self.BEM[i].Light + self.RadFEquip * self.BEM[i].Elec) / intHeat
                  # fraction of latent heat (from occupants) of whole internal heat per unit floor area [W m^-2]
                  self.BEM[i].building.intHeatFLat = self.LatFOcc * self.sensOcc * self.BEM[i].Nocc/intHeat

                  # Update envelope temperature layers [K]
                  self.BEM[i].T_wallex = self.BEM[i].wall.layerTemp[0]   # Wall temperature exposed to outdoor environment [K]
                  self.BEM[i].T_wallin = self.BEM[i].wall.layerTemp[-1]  # Wall temperature exposed to indoor environment [K]
                  self.BEM[i].T_roofex = self.BEM[i].roof.layerTemp[0]   # Roof temperature exposed to outdoor environment [K]
                  self.BEM[i].T_roofin = self.BEM[i].roof.layerTemp[-1]  # Roof temperature exposed to indoor environment [K]

              # Update rural heat fluxes & update vertical diffusion model (VDM)
              self.rural.infra = self.forc.infra - self.rural.emissivity * self.SIGMA * self.rural.layerTemp[0]**4.

              # Update heat fluxes [W m^-2] and surface temperature [K] in rural area
              self.rural.SurfFlux(self.forc, self.geoParam, self.simTime, self.forc.hum, self.forc.temp, self.forc.wind, 2., 0.,'rural',self.var_sens)

              # Update vertical diffusion model (VDM): Calculate temperature profile, wind speed profile and density profile in rural area
              self.RSM.VDM(self.forc, self.rural, self.geoParam, self.simTime,self.forc.dir,self.h_ublavg,self.var_sens)

              # Update UWG wind speed within the canyon by taking average of velocity profiles within the canyon
              WindUrban = numpy.sqrt(numpy.mean(self.vx[0:self.nz_u])**2+numpy.mean(self.vy[0:self.nz_u])**2)

              # Calculate urban heat fluxes, update UCM & UBL
              # !!!!! UBL IS REMOVED !!!!!
              # self.UCM, self.UBL, self.BEM = urbflux(self.UCM, self.UBL, self.BEM, self.forc, self.geoParam, self.simTime, self.RSM,L_wall_net[it], L_road_net[it],self.var_sens,WindUrban)
              self.UCM, self.BEM = urbflux(self.UCM, self.BEM, self.forc, self.geoParam, self.simTime, self.RSM,L_wall_net[it], L_road_net[it],self.var_sens,WindUrban)
              self.UCM.UCModel(self.BEM, self.RSM.tempProf[-1], self.forc, self.geoParam)
              # !!!!! UBL IS REMOVED !!!!!
              # self.UBL.UBLModel(self.UCM, self.RSM, self.rural, self.forc, self.geoParam, self.simTime)

              # ========================================================================================================
              # 1-D Model (Sec.3 Start)
              # ========================================================================================================
              # Calculate density profile of density [kg m^-3]
              rho_prof = numpy.zeros(self.nz)
              for i_rho in range(0,self.nz):
                  # a constant density lapse rate of - 0.000133 [kg m-3 m-1]
                  # rho = rho_0-0.000133*(z-z_0)
                  rho_prof[i_rho] = self.UCM.rhoCan-0.000133*(self.z[i_rho]-0)

              # Update total sensible waste heat to canyon per unit building footprint area [W m^-2]
              SensHt_HVAC = self.BEM[0].building.sensWaste+self.BEM[1].building.sensWaste+self.BEM[2].building.sensWaste

              # Call column (1-D) model
              ColModelParam = ColModel(self.UCM.wallTemp,self.UCM.roofTemp,self.UCM.roadTemp,self.RSM.tempProf[self.nz-1],self.forc.hum,self.forc.wind,self.forc.uDir, self.vx, self.vy, self.tke, self.th, self.qn,
                                      self.nz, self.Ck, self.dlk, self.nz_u, self.dz, self.simTime.dt, self.vol,self.road.vegCoverage, self.lambdap, self.lambdaf, self.bldHeight, self.CP,
                                      self.th0, self.Cdrag, self.pb, self.ss, self.prandtl,self.schmidt, self.G, self.Ceps, self.dls,self.sf, rho_prof,self.h_LAD,self.f_LAD,L_tot_net[it],S_Trees[it],L_trees_abs[it],it,self.var_sens,SensHt_HVAC,self.theta_can,self.HVAC_street_frac,self.HVAC_atm_frac)

              self.vx, self.vy, self.tke, self.th, self.qn, ustarCol, self.Km, self.tveg_tmp,Ri_b,Tveg = ColModelParam.ColumnModelCal()

              # Update new mean canyon temperature of 1-D model
              MeanTempNew_1D = numpy.mean(self.th)
              # Update new canyon temperature of UWG
              MeanTempNew_UWG = self.UCM.canTemp

              # Calculate error between current and previous iteration
              canTemp_Err_1D = abs(MeanTempNew_1D - MeanTempOld_1D) / MeanTempNew_1D
              canTemp_Err_UWG = abs(MeanTempNew_UWG - MeanTempOld_UWG) / MeanTempNew_UWG

              # Update old mean canyon temperature of 1-D model
              MeanTempOld_1D = MeanTempNew_1D
              # Update old canyon temperature of UWG
              MeanTempOld_UWG = MeanTempNew_UWG

              # Update UWG temperature within the canyon
              self.UCM.canTemp = numpy.mean(self.th[0:self.nz_u])

              # ========================================================================================================
              # 1-D Model (Sec.3 End)
              # ========================================================================================================

              self.logger.info("dbT = {}".format(self.UCM.canTemp-273.15))
              if n > 0:
                  logging.info("dpT = {}".format(self.UCM.Tdp))
                  logging.info("RH  = {}".format(self.UCM.canRHum))

              if self.is_near_zero(self.simTime.secDay % self.simTime.timePrint) and n < self.N:

                  self.logger.info("{0} ----sim time step = {1}----\n\n".format(__name__, n))

                  self.WeatherData[n] = copy.copy(self.forc)
                  _Tdb, _w, self.UCM.canRHum, _h, self.UCM.Tdp, _v = psychrometrics(self.UCM.canTemp, self.UCM.canHum, self.forc.pres)

                  self.UCMData[n] = copy.copy(self.UCM)
                  self.RSMData[n] = copy.copy(self.RSM)

                  self.logger.info("dbT = {}".format(self.UCMData[n].canTemp-273.15))
                  self.logger.info("dpT = {}".format(self.UCMData[n].Tdp))
                  self.logger.info("RH  = {}".format(self.UCMData[n].canRHum))

                  # Print progress bar
                  sim_it = round((it/float(self.simTime.nt))*100.0,1)
                  progress_bar.print_progress(sim_it, 100.0, prefix = "Progress:", bar_length = 25)

                  n += 1

            # Store data at times of interest
            if iO < len(Output_TimInd):
                if it == Output_TimInd[iO]:
                    TempDiu_TANAB[:][iO] = self.th
                    UDiu_TANAB[:][iO] = self.vx
                    VDiu_TANAB[:][iO] = self.vy
                    TKEDiu_TANAB[:][iO] = self.tke
                    qDiu_TANAB[:][iO] = self.qn
                    TempDiu_Rural[:][iO] = self.RSM.tempProf
                    WindDiu_Rural[:][iO] = self.RSM.windProf
                    KtDiu_Rural[:][iO] = self.RSM.Kt
                    lm_1Diu_Rural[:][iO] = self.RSM.lm_1
                    lm_2Diu_Rural[:][iO] = self.RSM.lm_2
                    iO += 1


        ProfilesFilename = ["Output/VCWG_Profiles_D" + str(nday) + ".txt" for nday in range(1, int(self.N / 24))]
        for nf in range(0,len(ProfilesFilename)):
            outputFileProf = open(ProfilesFilename[nf], "w")
            outputFileProf.write("#### \t The Vertical City Weather Generator (VCWG)  \t #### \n")
            outputFileProf.write("#0:z \t 1:U_0000 \t 2:U_0100 \t 3:U_0200 \t 4:U_0300 \t 5:U_0400 \t 6:U_0500 \t 7:U_0600 \t 8:U_0700 \t 9:U_0800 \t 10:U_0900 \t 11:U_1000 \t 12:U_1100 \t 13:U_1200 \t 14:U_1300	\t 15:U_1400 \t 16:U_1500 \t 17:U_1600 \t 18:U_1700 \t 19:U_1800 \t 20:U_1900 \t 21:U_2000 \t 22:U_2100 \t 23:U_2200 \t 24:U_2300 \
                                        \t 25:V_0000 \t 26:V_0100 \t 27:V_0200 \t 28:V_0300 \t 29:V_0400 \t 30:V_0500 \t 31:V_0600 \t 32:V_0700 \t 33:V_0800 \t 34:V_0900 \t 35:V_1000 \t 36:V_1100 \t 37:V_1200 \t 38:V_1300 \t 39:V_1400 \t 40:V_1500 \t 41:V_1600 \t 42:V_1700 \t 43:V_1800 \t 44:V_1900 \t 45:V_2000 \t 46:V_2100 \t 47:V_2200 \t 48:V_2300 \
                                         \t 49:T_0000 \t 50:T_0100 \t 51:T_0200 \t 52:T_0300 \t 53:T_0400 \t 54:T_0500 \t 55:T_0600 \t 56:T_0700 \t 57:T_0800 \t 58:T_0900 \t 59:T_1000 \t 60:T_1100 \t 61:T_1200 \t 62:T_1300 \t 63:T_1400 \t 64:T_1500 \t 65:T_1600 \t 66:T_1700 \t 67:T_1800 \t 68:T_1900 \t 69:T_2000 \t 70:T_2100 \t 71:T_2200 \t 72:T_2300 \
                                          \t 73:TKE_0000 \t 74:TKE_0100 \t 75:TKE_0200 \t 76:TKE_0300 \t 77:TKE_0400 \t 78:TKE_0500 \t 79:TKE_0600 \t 80:TKE_0700 \t 81:TKE_0800 \t 82:TKE_0900 \t 83:TKE_1000 \t 84:TKE_1100 \t 85:TKE_1200 \t 86:TKE_1300 \t 87:TKE_1400 \t 88:TKE_1500 \t 89:TKE_1600 \t 90:TKE_1700 \t 91:TKE_1800 \t 92:TKE_1900 \t 93:TKE_2000 \t 94:TKE_2100 \t 95:TKE_2200 \t 96:TKE_2300 \
                                           \t 97:q_0000 \t 98:q_0100 \t 99:q_0200 \t 100:q_0300 \t 101:q_0400 \t 102:q_0500 \t 103:q_0600 \t 104:q_0700 \t 105:q_0800 \t 106:q_0900 \t 107:q_1000 \t 108:q_1100 \t 109:q_1200 \t 110:q_1300 \t 111:q_1400 \t 112:q_1500 \t 113:q_1600 \t 114:q_1700 \t 115:q_1800 \t 116:q_1900 \t 117:q_2000 \t 118:q_2100 \t 119:q_2200 \t 120:q_2300 \
                                            \t 121:Trur_0000 \t 122:Trur_0100 \t 123:Trur_0200 \t 124:Trur_0300 \t 125:Trur_0400 \t 126:Trur_0500 \t 127:Trur_0600 \t 128:Trur_0700 \t 129:Trur_0800 \t 130:Trur_0900 \t 131:Trur_1000 \t 132:Trur_1100 \t 133:Trur_1200 \t 134:Trur_1300 \t 135:Trur_1400 \t 136:Trur_1500 \t 137:Trur_1600 \t 138:Trur_1700 \t 139:Trur_1800 \t 140:Trur_1900 \t 141:Trur_2000 \t 142:Trur_2100 \t 143:Trur_2200 \t 144:Trur_2300 \n")
            for i in range(0, self.nz):
                outputFileProf.write("%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t \
                                      %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t \
                                      %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t \
                                      %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t \
                                      %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t \
                                      %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n"

                                    % (self.z[i],UDiu_TANAB[nf+0][i],UDiu_TANAB[nf+1][i],UDiu_TANAB[nf+2][i],UDiu_TANAB[nf+3][i],
                                       UDiu_TANAB[nf+4][i],UDiu_TANAB[nf+5][i],UDiu_TANAB[nf+6][i],UDiu_TANAB[nf+7][i],UDiu_TANAB[nf+8][i],
                                       UDiu_TANAB[nf+9][i],UDiu_TANAB[nf+10][i],UDiu_TANAB[nf+11][i],UDiu_TANAB[nf+12][i],UDiu_TANAB[nf+13][i],
                                       UDiu_TANAB[nf+14][i],UDiu_TANAB[nf+15][i],UDiu_TANAB[nf+16][i],UDiu_TANAB[nf+17][i],UDiu_TANAB[nf+18][i],
                                       UDiu_TANAB[nf+19][i],UDiu_TANAB[nf+20][i],UDiu_TANAB[nf+21][i],UDiu_TANAB[nf+22][i],UDiu_TANAB[nf+23][i],

                                       VDiu_TANAB[nf+0][i], VDiu_TANAB[nf+1][i], VDiu_TANAB[nf+2][i], VDiu_TANAB[nf+3][i], VDiu_TANAB[nf+4][i],
                                       VDiu_TANAB[nf+5][i], VDiu_TANAB[nf+6][i], VDiu_TANAB[nf+7][i], VDiu_TANAB[nf+8][i], VDiu_TANAB[nf+9][i],
                                       VDiu_TANAB[nf+10][i], VDiu_TANAB[nf+11][i], VDiu_TANAB[nf+12][i], VDiu_TANAB[nf+13][i], VDiu_TANAB[nf+14][i],
                                       VDiu_TANAB[nf+15][i], VDiu_TANAB[nf+16][i], VDiu_TANAB[nf+17][i], VDiu_TANAB[nf+18][i], VDiu_TANAB[nf+19][i],
                                       VDiu_TANAB[nf+20][i], VDiu_TANAB[nf+21][i], VDiu_TANAB[nf+22][i], VDiu_TANAB[nf+23][i],

                                       TempDiu_TANAB[nf+0][i], TempDiu_TANAB[nf+1][i], TempDiu_TANAB[nf+2][i], TempDiu_TANAB[nf+3][i], TempDiu_TANAB[nf+4][i],
                                       TempDiu_TANAB[nf+5][i], TempDiu_TANAB[nf+6][i], TempDiu_TANAB[nf+7][i], TempDiu_TANAB[nf+8][i], TempDiu_TANAB[nf+9][i],
                                       TempDiu_TANAB[nf+10][i], TempDiu_TANAB[nf+11][i], TempDiu_TANAB[nf+12][i], TempDiu_TANAB[nf+13][i], TempDiu_TANAB[nf+14][i],
                                       TempDiu_TANAB[nf+15][i], TempDiu_TANAB[nf+16][i], TempDiu_TANAB[nf+17][i], TempDiu_TANAB[nf+18][i], TempDiu_TANAB[nf+19][i],
                                       TempDiu_TANAB[nf+20][i], TempDiu_TANAB[nf+21][i], TempDiu_TANAB[nf+22][i], TempDiu_TANAB[nf+23][i],

                                       TKEDiu_TANAB[nf+0][i], TKEDiu_TANAB[nf+1][i], TKEDiu_TANAB[2][i], TKEDiu_TANAB[3][i], TKEDiu_TANAB[4][i], TKEDiu_TANAB[5][i],
                                       TKEDiu_TANAB[nf+6][i], TKEDiu_TANAB[nf+7][i], TKEDiu_TANAB[8][i], TKEDiu_TANAB[9][i], TKEDiu_TANAB[10][i], TKEDiu_TANAB[11][i],
                                       TKEDiu_TANAB[nf+12][i], TKEDiu_TANAB[nf+13][i], TKEDiu_TANAB[14][i], TKEDiu_TANAB[15][i], TKEDiu_TANAB[16][i], TKEDiu_TANAB[17][i],
                                       TKEDiu_TANAB[nf+18][i], TKEDiu_TANAB[nf+19][i], TKEDiu_TANAB[20][i], TKEDiu_TANAB[21][i], TKEDiu_TANAB[22][i], TKEDiu_TANAB[23][i],

                                       qDiu_TANAB[nf+0][i], qDiu_TANAB[nf+1][i], qDiu_TANAB[nf+2][i], qDiu_TANAB[nf+3][i], qDiu_TANAB[nf+4][i], qDiu_TANAB[nf+5][i], qDiu_TANAB[nf+6][i],
                                       qDiu_TANAB[nf+7][i], qDiu_TANAB[nf+8][i], qDiu_TANAB[nf+9][i], qDiu_TANAB[nf+10][i], qDiu_TANAB[nf+11][i], qDiu_TANAB[nf+12][i], qDiu_TANAB[nf+13][i],
                                       qDiu_TANAB[nf+14][i], qDiu_TANAB[nf+15][i], qDiu_TANAB[nf+16][i], qDiu_TANAB[nf+17][i], qDiu_TANAB[nf+18][i], qDiu_TANAB[nf+19][i], qDiu_TANAB[nf+20][i],
                                       qDiu_TANAB[nf+21][i], qDiu_TANAB[nf+22][i], qDiu_TANAB[nf+23][i],

                                       TempDiu_Rural[nf+0][i], TempDiu_Rural[nf+1][i], TempDiu_Rural[nf+2][i], TempDiu_Rural[nf+3][i], TempDiu_Rural[nf+4][i], TempDiu_Rural[nf+5][i],
                                       TempDiu_Rural[nf+6][i], TempDiu_Rural[nf+7][i], TempDiu_Rural[nf+8][i], TempDiu_Rural[nf+9][i], TempDiu_Rural[nf+10][i], TempDiu_Rural[nf+11][i],
                                       TempDiu_Rural[nf+12][i], TempDiu_Rural[nf+13][i], TempDiu_Rural[nf+14][i], TempDiu_Rural[nf+15][i], TempDiu_Rural[nf+16][i], TempDiu_Rural[nf+17][i],
                                       TempDiu_Rural[nf+18][i], TempDiu_Rural[nf+19][i], TempDiu_Rural[nf+20][i], TempDiu_Rural[nf+21][i], TempDiu_Rural[nf+22][i], TempDiu_Rural[nf+23][i]))
            outputFileProf.close()


    def write_epw(self):
        """ Section 8 - Writing new EPW file
        """
        epw_prec = self.epw_precision # precision of epw file input

        for iJ in xrange(len(self.UCMData)):
            # [iJ+self.simTime.timeInitial-8] = increments along every weather timestep in epw
            # [6 to 21]                       = column data of epw
            self.epwinput[iJ+self.simTime.timeInitial-8][6] = "{0:.{1}f}".format(self.UCMData[iJ].canTemp - 273.15, epw_prec) # dry bulb temperature  [C]
            self.epwinput[iJ+self.simTime.timeInitial-8][7] = "{0:.{1}f}".format(self.UCMData[iJ].Tdp, epw_prec)              # dew point temperature [C]
            self.epwinput[iJ+self.simTime.timeInitial-8][8] = "{0:.{1}f}".format(self.UCMData[iJ].canRHum, epw_prec)          # relative humidity     [%]
            self.epwinput[iJ+self.simTime.timeInitial-8][21] = "{0:.{1}f}".format(self.WeatherData[iJ].wind, epw_prec)        # wind speed [m s^-1]

        # Writing new EPW file
        epw_new_id = open(self.newPathName, "w")

        for i in xrange(8):
            new_epw_line = '{}\r\n'.format(reduce(lambda x,y: x+","+y, self._header[i]))
            epw_new_id.write(new_epw_line)

        for i in xrange(len(self.epwinput)):
            printme = ""
            for ei in xrange(34):
                printme += "{}".format(self.epwinput[i][ei]) + ','
            printme = printme + "{}".format(self.epwinput[i][ei])
            new_epw_line = "{0}\r\n".format(printme)
            epw_new_id.write(new_epw_line)

        epw_new_id.close()

        print "\nNew climate file '{}' is generated at {}.".format(self.destinationFileName, self.destinationDir)

    def run(self):

        # run main class methods
        self.read_epw()
        self.set_input()
        self.instantiate_input()
        self.hvac_autosize()
        self.simulate()
        self.write_epw()


def procMat(materials,max_thickness,min_thickness):
    """ Processes material layer so that a material with single
    layer thickness is divided into two and material layer that is too
    thick is subdivided
    """
    newmat = []
    newthickness = []
    k = materials.layerThermalCond
    Vhc = materials.layerVolHeat

    if len(materials.layerThickness) > 1:

        for j in xrange(len(materials.layerThickness)):
            # Break up each layer that's more than max thickness (0.05m)
            if materials.layerThickness[j] > max_thickness:
                nlayers = math.ceil(materials.layerThickness[j]/float(max_thickness))
                for i in xrange(int(nlayers)):
                    newmat.append(Material(k[j],Vhc[j],name=materials._name))
                    newthickness.append(materials.layerThickness[j]/float(nlayers))
            # Material that's less then min_thickness is not added.
            elif materials.layerThickness[j] < min_thickness:
                print "WARNING: Material '{}' layer found too thin (<{:.2f}cm), ignored.".format(materials._name, min_thickness*100)
            else:
                newmat.append(Material(k[j],Vhc[j],name=materials._name))
                newthickness.append(materials.layerThickness[j])

    else:

        # Divide single layer into two (UWG assumes at least 2 layers)
        if materials.layerThickness[0] > max_thickness:
            nlayers = math.ceil(materials.layerThickness[0]/float(max_thickness))
            for i in xrange(int(nlayers)):
                newmat.append(Material(k[0],Vhc[0],name=materials._name))
                newthickness.append(materials.layerThickness[0]/float(nlayers))
        # Material should be at least 1cm thick, so if we're here,
        # should give warning and stop. Only warning given for now.
        elif materials.layerThickness[0] < min_thickness*2:
            newthickness = [min_thickness/2., min_thickness/2.]
            newmat = [Material(k[0],Vhc[0],name=materials._name), Material(k[0],Vhc[0],name=materials._name)]
            print "WARNING: a thin (<2cm) single material '{}' layer found. May cause error.".format(materials._name)
        else:
            newthickness = [materials.layerThickness[0]/2., materials.layerThickness[0]/2.]
            newmat = [Material(k[0],Vhc[0],name=materials._name), Material(k[0],Vhc[0],name=materials._name)]
    return newmat, newthickness

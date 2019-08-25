
"""
Building Energy Model (BEM) class definition
Developed by Bruno Bueno
Building Technology, Massachusetts Institute of Technology (MIT), Cambridge, U.S.A.
Last update: 2012
"""

class BEMDef(object):
    """

    attributes:
        building;       # building class type
        mass;           # mass element class
        wall;           # wall element class
        roof;           # roof element class

        frac;           # fraction of the urban floor space of this typology
        fl_area;        # Total building floor area in the urban area

        ElecTotal;      # Actual total electricity per unit floor area [W m^-2]
        Elec;           # Actual electricity consumption per unit floor area [W m^-2]
        Gas;            # Actual gas consumption per unit floor area [W m^-2]
        Light;          # Actual light per unit floor area [W m^-2]
        Qocc;           # Actual heat load from occupant [W m^-2]
        SWH;            # Actual hot water usage per unit floor area [kg hr^-1 m^-2]
        Nocc;           # Actual number of people

        T_wallex;       # Wall surface temp (ext) [K]
        T_wallin;       # Wall surface temp (int) [K]
        T_roofex;       # Roof surface temp (ext) [K]
        T_roofin;       # Roof surface temp (int) [K]
    """

    def __init__(self,building,mass,wall,roof,frac):
        self.building = building    # building class type
        self.mass = mass            # mass element class
        self.wall = wall            # wall element class
        self.roof = roof            # roof element class
        self.frac = frac            # fraction of the urban floor space of this typology (~ bld_footprint/urba_footprint ?)

    def __repr__(self):
        return "BEMDef: {a}, {b}, {c}, wall={d}".format(
            a=self.building.Type,
            b=self.building.Zone,
            c=self.building.Era,
            d=self.wall._name
            )

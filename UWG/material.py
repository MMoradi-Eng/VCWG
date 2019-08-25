"""Material class"""

"""
UWG Material
Developed by Bruno Bueno
Building Technology, Massachusetts Institute of Technology (MIT), Cambridge, U.S.A.
Last update: 2012

Attributes:
    thermalCond: Thermal conductivity (W m^-1 K^-1)
    volHeat: Volumetric heat capacity (J m^-3 K^-1)
    name: Name of the material.
"""

class Material(object):
    def __init__(self, thermalCond, volHeat, name='noname'):
        self._name = name # purely for internal purpose
        self.thermalCond = thermalCond
        self.volHeat = volHeat

    def __repr__(self):
        return "Material: {a}, k={b}, spec vol={c}".format(
            a=self._name,
            b=self.thermalCond,
            c=self.volHeat
            )

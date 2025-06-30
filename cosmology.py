"""
Cosmology module for handling the Flat LCDM cosmology calculations.
"""

class FlatCosmology:
    """
    Core cosmology class
    """

    def __init__(self, h, omega_matter):
        """
        Read in h and Om0 and build cosmology from that.
        """
        self.h = h
        self.omega_m = omega_matter
        self.hubble_constant = 100 * self.h
        self.omega_lambda = 1 - self.omega_m
        self.omega_k = 0.
        self.omgega_radiation = 0.
    
    def comoving_distance() ->
import math
from scipy.special import erf, erfinv


class FPA(object):
    """
    Sensor Parameters for Focal Plane Array
    
    """
    hc = 1.98644582e-25  # (m ** 3 kg / s ** 2)

    # noise and sensor parameters
    NEFD = 6e-18
    OT = 0.6  # (0-1) optical transmission
    QE = 0.54  # (0-1) quantum efficiency
    wL = 550e-9  # 3e-6  # (m) wavelength
    ReadNoise = 30  # (counts)
    # ifov = 60e-6  # (rad) pixel field of view
    ifov = 0.0001095679  # Nikon D5200 w/ 50mm
    AA = 22.5 / (100 ** 2)  # 60e-4  # (m**2) aperture area  22.5 cm2 for nikon 5200 w 50mm f1.4
    dt = 0.25  # (sec) integration time
    EOD = 0.5  # (0-1) EOD

    # evaluated parameters
    blur = 1 / math.sqrt(8) / erfinv(math.sqrt(EOD))

    # conversion factors
    cf = (AA * dt * QE * OT * wL / hc)
    C_Rad2Cnts = ifov ** 2 * cf
    C_Wcm2Cnts = 1e4 * cf
    W_Wm22Cnts = cf

    # Note Conversion factor for Target to Counts is at a specific Sensor to Target Range

    PixelNoise2 = ReadNoise ** 2 + ((NEFD * 1e4) * cf) ** 2 / dt

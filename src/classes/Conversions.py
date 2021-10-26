import numpy as np
from datetime import datetime, timedelta, MINYEAR

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12)  # noon (the epoch name is unrelated)
J2000_JD = timedelta(2451545)  # julian epoch in julian dates


def dt2jd(now=None):
    if now is None:
        now = datetime.utcnow()

    K = now.year
    M = now.month
    I = now.day

    hours = now.hour
    minutes = now.minute
    seconds = now.second
    microseconds = now.microsecond

    term1 = 367 * K
    term2 = int(7 * (K + int(M + 9) // 12) // 4)
    term3 = int(275 * M // 9)
    term4 = I
    term5 = 1721013.5
    term6 = (hours + minutes / 60.0 + seconds / 3600.0 + microseconds / (3600.0 * 1E6)) / 24.0

    if 100 * K + M - 190002.5 > 0:
        term7 = 0
    else:
        term7 = 0.5

    return term1 - term2 + term3 + term4 + term5 + term6 + term7


def jd2dt(jd=None):
    if jd is None:
        return datetime.utcnow()

    ijd = int(jd + 0.5)
    fjd = jd - ijd

    L = ijd + 68569
    N = 4 * L // 146097
    L = L - (146097 * N + 3) // 4
    I = 4000 * (L + 1) // 1461001
    L = L - 1461 * I // 4 + 31
    J = 80 * L // 2447
    K = L - 2447 * J // 80
    L = J // 11
    J = J + 2 - 12 * L
    I = 100 * (N - 49) + I + L

    year = int(I)
    month = int(J)
    day = int(K)

    hours = fjd * 24
    if hours < 12:
        hours += 12
    ihours = int(hours)

    minutes = (hours - ihours) * 60
    iminutes = int(minutes)

    seconds = (minutes - iminutes) * 60
    iseconds = int(seconds)

    microseconds = (seconds - iseconds) * 1E6
    imicroseconds = int(microseconds)

    dt = datetime(year, month, day, ihours, iminutes, iseconds, imicroseconds, tzinfo=datetime.timezone.utc)
    return dt


def JD2HourAngle(jd):
    """ Convert the given Julian date to hour angle.

    Arguments:
        jd: [float] Julian date.

    Return:
        hour_angle: [float] Hour angle (deg).

    """

    T = (jd - 2451545) / 36525.0
    hour_angle = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T ** 2 - (T ** 3) / 38710000.0

    return hour_angle


def JD2LST(julian_date, lon):
    """ Convert Julian date to Local Sidreal Time and Greenwich Sidreal Time.

    Arguments;
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees

    Return:
        [tuple]: (LST, GST): [tuple of floats] a tuple of Local Sidreal Time and Greenwich Sidreal Time
            (degrees)
    """

    t = (julian_date - J2000_JD.days) / 36525.0

    # Greenwich Sidreal Time
    GST = 280.46061837 + 360.98564736629 * (julian_date - 2451545) + 0.000387933 * t ** 2 - ((t ** 3) / 38710000)
    GST = (GST + 360) % 360

    # Local Sidreal Time
    LST = (GST + lon + 360) % 360

    return LST, GST


def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Julian Date with epoch J2000.0.

    @param year: [int] year
    @param month: [int] month
    @param day: [int] day of the date
    @param hour: [int] hours
    @param minute: [int] minutes
    @param second: [int] seconds
    @param millisecond: [int] milliseconds (optional)
    @param UT_corr: [float] UT correction in hours (difference from local time to UT)

    @return :[float] julian date, epoch 2000.0
    """

    # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond * 1000))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)

    # Convert seconds to day fractions
    return julian.days + (julian.seconds + julian.microseconds / 1000000.0) / 86400.0


def datetime2JD(dt, UT_corr=0.0):
    """ Converts a datetime object to Julian date.

    Arguments:
        dt: [datetime object]

    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)

    Return:
        jd: [float] Julian date
    """

    return date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond / 1000.0,
                   UT_corr=UT_corr)


def jd2Date(jd, UT_corr=0, dt_obj=False):
    """ Converts the given Julian date to (year, month, day, hour, minute, second, millisecond) tuple.
    Arguments:
        jd: [float] Julian date
    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)
        dt_obj: [bool] returns a datetime object if True. False by default.
    Return:
        (year, month, day, hour, minute, second, millisecond)
    """

    dt = timedelta(days=jd)

    try:
        date = dt + JULIAN_EPOCH - J2000_JD + timedelta(hours=UT_corr)

        # If the date is out of range (i.e. before year 1) use year 1. This is the limitation in the datetime
    # library. Time handling should be switched to astropy.time
    except OverflowError:
        date = datetime(MINYEAR, 1, 1, 0, 0, 0)

    # Return a datetime object if dt_obj == True
    if dt_obj:
        return date

    return date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond / 1000.0


def RaDec2AzEl(Ra, Dec, lat, lon, JD):

    T_UT1 = (JD - 2451545) / 36525
    ThetaGMST = (67310.54841 + (876600*3600 + 8640184.812866) * T_UT1 +
                 0.093104 * (T_UT1 ** 2) - (6.2 * 10 ** -6) * (T_UT1 ** 3))

    ThetaGMST = np.mod((np.mod(ThetaGMST, 86400 * (ThetaGMST / abs(ThetaGMST)))/240), 360)
    ThetaLST = ThetaGMST + lon

    d2r = math.pi / 180
    r2d = 180 / math.pi

    LHA = np.mod(ThetaLST - Ra, 360) * d2r
    # lat *= d2r
    # lon *= d2r
    # Ra *= d2r
    # Dec *= d2r
    lat, lon, Ra, Dec = map(lambda x: d2r * x, [lat, lon, Ra, Dec])

    El = math.asin(math.sin(lat) * math.sin(Dec) + math.cos(lat) * math.cos(Dec) * math.cos(LHA))

    Az = np.mod(math.atan2(-math.sin(LHA) * math.cos(Dec) / math.cos(El),
                (math.sin(Dec) - math.sin(El) * math.sin(lat)) / (math.cos(El) * math.cos(lat))) * r2d, 360)

    return Az, El * r2d


RaDec2AzEl_vect = np.vectorize(RaDec2AzEl, excluded=['lat', 'lon'])


def AzEl2RaDec(Az, El, lat, lon, JD):
    """
    :param Az: Local Azimuth Angle (degrees)
    :param El: Local Elevation angle (degrees)
    :param lat: Site latitude in degrees (-90:90)
    :param lon: Site longitude in degrees (-180:180)
    :param JD:
    :return:
    """
    T_UT1 = (JD - 2451545) / 36525
    ThetaGMST = (67310.54841 + (876600 * 3600 + 8640184.812866) * T_UT1 +
                 0.093104 * (T_UT1 ** 2) - (6.2 * 10 ** -6) * (T_UT1 ** 3))

    ThetaGMST = np.mod((np.mod(ThetaGMST, 86400 * (ThetaGMST / abs(ThetaGMST))) / 240), 360)
    ThetaLST = ThetaGMST + lon

    d2r = math.pi / 180
    r2d = 180 / math.pi

    lat, lon, Az, El = map(lambda x: d2r * x, [lat, lon, Az, El])

    Dec = np.arcsin(np.sin(El) * np.sin(lat) + np.cos(El) * np.cos(lat) * np.cos(Az))
    LHA = np.arctan2(-np.sin(Az) * np.cos(El) / np.cos(Dec),
                     (np.sin(El) - np.sin(Dec) * np.sin(lat)) / (np.cos(Dec) * np.cos(lat)))

    Ra = np.mod(ThetaLST - LHA*r2d, 360)

    return Ra, Dec


AzEl2RaDec_vect = np.vectorize(AzEl2RaDec, excluded=['lat', 'lon'])


if __name__ == '__main__':
    dt_start = jd2Date(2458900.466029105200, dt_obj=True)
    dt_mid = jd2Date(2458900.466057176700, dt_obj=True)
    dt_stop = jd2Date(2458900.466085248000, dt_obj=True)

    jd1 = dt2jd()

    ha = JD2HourAngle(jd1)
    lst, gst = JD2LST(jd1, -86.5)

    ha

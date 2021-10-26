import math

import numpy as np
from scipy.special import erfinv, erf

from classes.FPA import FPA


def target_streak_matrix_01(d1, d2, blur, s12):
    Ns = s12.shape[0]
    b = blur * math.sqrt(2)

    M = np.zeros([d1, d2])

    d1r = np.array(list(range(d1 + 1)))
    d2r = np.array(list(range(d2 + 1)))

    for iis in range(Ns):
        a1 = np.expand_dims(np.diff(erf((d1r - s12[iis, 0]) / b)) / 2, -1)
        a2 = np.expand_dims(np.diff(erf((d2r - s12[iis, 1]) / b)) / 2, -1)
        M += np.dot(a1, a2.T)

    return M / Ns


def make_streak_mask(d1=5, d2=5, blur=0.5, L=1.5, ang=35):
    ang = math.radians(ang)
    t = np.linspace(-1, 1, 41)
    s12 = np.array([(L * math.cos(ang) / 2 * t + d1 / 2),
                    (L * math.sin(ang) / 2 * t + d2 / 2)]).T
    return target_streak_matrix_01(d1, d2, blur, s12)


def detection_mask(d1, d2, m):
    A = np.hstack([m.reshape([-1, 1]), np.ones([d1 * d2, 1])])
    p1 = A.T.dot(A)
    M = np.linalg.inv(p1).dot(A.T)
    nsf = M.dot(M.T)
    nsf = nsf[0, 0]


def detections_above_threshold(N1, N2, DetPeak, FAR, Idet):
    fpID = np.linspace(1, (N1 * N2), (N1 * N2))

    xyp = np.array([np.floor((fpID - 1) / N1) + 1, np.mod(fpID - 1, N1) + 1])
    XI = np.argsort(DetPeak, 'descend')
    snr2 = DetPeak[XI]

    TsnrT = math.sqrt(2) * erfinv(1 - 2 * FAR)  # Theoretical SNR Threshold

    NDET = np.round(FAR * N1 * N2)  # number of possible detections
    snrT = np.sqrt(snr2[NDET])

    OSM = np.array([xyp[XI[:NDET], :],
                    np.sqrt(DetPeak[XI[:NDET]]),
                    Idet[XI[:NDET]]])


def FPA_Inject_Target_Streak_on_FPA_01(FPA, SP, td1, td2, s12, It, Rt):
    S12 = s12 + np.hstack([td1, td2])
    Mt = target_streak_matrix_01(FPA.shape[0], FPA.shape[1], SP.blur, S12)
    FPA += Mt.dot(It / (Rt.dot(Rt) * SP.C_Wm22Cnts))

    return FPA


def FPA_NoiseFPA_counts(SP, FPA):
    FPunc = np.sqrt(FPA + SP.PixelNoise2)+9
    FPAn = FPA + FPunc * np.random.randn(FPA.shape)
    return FPAn


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # m = make_streak_mask()
    # detection_mask(5, 5, m)
    SP = FPA()

    FRAMESIZE = 2000
    d1 = (2048 - FRAMESIZE) / 2
    zi = list(range(int(1+d1), int(2048 - d1)))

    bg_low, bg_high = -0.1, 0.1
    std_low, std_high = 3.0, 5.0

    bg_mean = np.random.uniform(bg_low, bg_high, 1)
    bg_std = np.random.uniform(std_low, std_high, 1)

    fpC = np.random.normal(loc=bg_mean, scale=bg_std, size=(4020, 6036))

    fpC = fpC[zi[0]:zi[-1], zi[0]:zi[-1]]

    D1 = fpC.shape[0]
    D2 = fpC.shape[1]

    TrgInt = 35  # (W/sr) exluces atmo abs
    Range = 1200 * 1000  # (m) range to target from sensor
    L = 25  # (pixels) length of streak
    ang = 27  # (deg) angle of target streak
    xp = 200  # location of streak
    yp = 200  # location of streak

    d1 = 50
    d2 = 50

    ang = math.radians(ang)
    t = np.linspace(-1, 1, 41)
    s12 = np.array([(L * math.cos(ang) / 2 * t + d1 / 2),
                    (L * math.sin(ang) / 2 * t + d2 / 2)]).T

    tm = target_streak_matrix_01(d1, d2, SP.blur, s12)
    tm = (tm - tm.min()) / tm.ptp()
    tm = tm * 100

    plt.imshow(tm, 'gray')
    plt.pause(0.01)

    fpidx = list(range(-26, 25))
    fpx = [_ + xp for _ in fpidx]
    fpy = [_ + yp for _ in fpidx]
    fpC[fpy[0]:fpy[-1], fpx[0]:fpx[-1]] += tm

    plt.imshow(fpC, 'gray')
    plt.pause(0.001)
    plt.show()
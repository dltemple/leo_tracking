from FFStruct import FFStruct
from astropy.io import fits


def read(filename, array=False):
    """ Read a FF structure from a FITS file.

    Arguments:
        directory: [str] Path to directory containing file
        filename: [str] Name of FF*.fits file (either with FF and extension or without)

    Keyword arguments:
        array: [ndarray] True in order to populate structure's array element (default is False)
        full_filename: [bool] True if full file name is given explicitly, a name which may differ from the
            usual FF*.fits format. False by default.

    Return:
        [ff structure]

    """

    fid = open(filename, "rb")

    # Init an empty FF structure
    ff = FFStruct()

    # Read in the FITS
    hdulist = fits.open(fid)

    # Read the header
    head = hdulist[0].header

    # Read in the data from the header
    ff.nrows = head['NAXIS1']
    ff.ncols = head['NAXIS2']
    ff.nbits = head['BITPIX']

    # Read in the image data
    data = hdulist[0].data
    ff.maxpixel = np.max(data)
    ff.avepixel = np.mean(data)
    ff.stdpixel = np.std(data)

    if array:
        ff.array = np.dstack([ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel])

        ff.array = np.swapaxes(ff.array, 0, 1)
        ff.array = np.swapaxes(ff.array, 0, 2)

    # CLose the FITS file
    hdulist.close()
    fid.close()

    return ff

# CARMENES 2 MOLECFIT
#
# Format CARMENES Data in a single spectrum
# and create the parameter file for Molecfit

# 18 Nov 2016
# ============================================

# import argparse
# from gooey import Gooey, GooeyParser
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import csv
from operator import itemgetter
from itertools import groupby
from scipy.interpolate import interp1d
from PyAstronomy import pyasl
# from matplotlib import cm
from astropy.constants import c


def corrected_overlap(wavelength_masked, flux_masked, good_overlap):
    """
    Creates the new 1D corrected spectrum
    ----------------------------------
    good overlap: contains on each line
    the wl before which to insert overlap
    corrected wl array
    corrected flux array
    """
    waves = wavelength_masked.flatten()
    fluxes = flux_masked.flatten()
    # Wavelength without the overlapping regions
    waves_corrected = waves.compressed()
    fluxes_corrected = fluxes.compressed()

    for i in range(len(good_overlap)):
        # Indice where the overlappin region ends
        idx = np.where(waves_corrected == good_overlap[i][0])[0]
        to_del = np.where(waves_corrected[:idx] > good_overlap[i][1][0])[0]
        #print to_del
        waves_corrected = np.delete(waves_corrected, to_del)
        fluxes_corrected = np.delete(fluxes_corrected, to_del)
        # Insert the 'mean' spectrum at this indice
        idx = np.where(waves_corrected == good_overlap[i][0])[0]
        waves_corrected = np.insert(waves_corrected, idx, good_overlap[i][1])
        fluxes_corrected = np.insert(fluxes_corrected, idx, good_overlap[i][2])

    return waves_corrected, fluxes_corrected


def two_spectra_overlap(wavelength_masked, flux_masked, overlap, nb_overlaps):
    """
    Extracts the two spectra which are in the overlapping region.
    Selects in the full spectrum the part of each order
    which overlap with the following.

    INPUTS: WAVELENGTH_MASKED, FLUX_MASKED = masked arrays of the CARMENES spectrum
            OVERLAP = number of the overlapping region (usually btw 0 and 10)
            NB_OVERLAPS = total number of overlapping regions

    OUTPUTS: WL_LEFT, FLUX_LEFT = array of wavelength and flux, order N
             WL_RIGHT, FLUX_RIGHT = array of wavelength and flux, order N+1
    """
    i = overlap
    # First overlap
    if i == 0:
        wl_left = wavelength_masked[i][wavelength_masked[i].mask].data
        flux_left = flux_masked[i][wavelength_masked[i].mask].data

        wl_right_full = wavelength_masked[i+1][wavelength_masked[i+1].mask].data
        flux_right_full = flux_masked[i+1][wavelength_masked[i+1].mask].data
        # right order has two overlapping regions, select the first_overlap
        # can be better take the first half of the list should be enough
        mid_order = wl_right_full[0] + (wl_right_full[-1]-wl_right_full[0])/2.
        idx_wl = np.where(wl_right_full < mid_order)

        wl_right = wl_right_full[idx_wl]
        flux_right = flux_right_full[idx_wl]

    # Last overlap
    elif i == nb_overlaps-1:
        # can be delete
        wl_left_full = wavelength_masked[i][wavelength_masked[i].mask].data
        flux_left_full = flux_masked[i][wavelength_masked[i].mask].data
        mid_order = wl_left_full[0] + (wl_left_full[-1]-wl_left_full[0])/2.
        idx_wl = np.where(wl_left_full > mid_order)
        wl_left = wl_left_full[idx_wl]
        flux_left = flux_left_full[idx_wl]

        wl_right = wavelength_masked[i+1][wavelength_masked[i+1].mask].data
        flux_right = flux_masked[i+1][wavelength_masked[i+1].mask].data

    # Middle overlaps
    else:
        # LEFT
        wl_left_full = wavelength_masked[i][wavelength_masked[i].mask].data
        flux_left_full = flux_masked[i][wavelength_masked[i].mask].data
        mid_order = wl_left_full[0] + (wl_left_full[-1]-wl_left_full[0])/2.
        idx_wl = np.where(wl_left_full > mid_order)
        wl_left = wl_left_full[idx_wl]
        flux_left = flux_left_full[idx_wl]
        # RIGHT
        wl_right_full = wavelength_masked[i+1][wavelength_masked[i+1].mask].data
        flux_right_full = flux_masked[i+1][wavelength_masked[i+1].mask].data
        mid_order = wl_right_full[0] + (wl_right_full[-1]-wl_right_full[0])/2.
        idx_wl = np.where(wl_right_full < mid_order)
        wl_right = wl_right_full[idx_wl]
        flux_right = flux_right_full[idx_wl]

    return wl_left, flux_left, wl_right, flux_right


def crosscorrelation(w, f, tw, tf, rvmin=-4., rvmax=4.0, drv=0.1):
    drvs = np.arange(rvmin, rvmax, drv)
    cc = np.zeros(len(drvs))
    for i, rv in enumerate(drvs):
        fi = interp1d(tw+rv, tf)
        cc[i] = np.sum(f * fi(w))

    return drvs, cc


def continuous_nb_in_list(data):
    """
    Finds in a list the sequences of consecutive numbers
    """

    continuous = []
    # data = [2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    # data = np.where(waves.mask)[0]
    # Find continuous numbers in data
    # for k, g in groupby(data, keyfunc):
    # g = group : groups each value in data with its index
    # lambda    : small function which substract each value by its index
    # k = key   : result of the lambda function
    for k, g in groupby(enumerate(data), lambda (i, x): i-x):
        group = map(itemgetter(1), g)
        continuous.append(group)

    return continuous


def rv_shift(wl_0, flux_0, wl_1, flux_1):
    """
    Find the rv shift needed to flux1 to match flux0
    INPUTS: WL_0, FLUX_0 = wavelength and flux of the spectrum 0
            WL_1, FLUX_1 = wavelength and flux of the spectrum 1
                           template shifted to match spectrum 0
    OUTPUTS: RV = the rv shift
             WL_1_CORR = wavelength 1 shifted to match spectrum 0
    """
    try:
        # Cross correlation
        rv, cc = pyasl.crosscorrRV(
            wl_0, flux_0, wl_1, flux_1,
            rvmin=-4., rvmax=4.0, drv=0.01, mode='doppler', skipedge=20.)
        maxind = np.argmax(cc)
        print "Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s"

        # Doppler shift the wavelengths
        wl_1_corr = wl_1 * (1. + (rv[maxind])/c.to('km/s').value)

        return rv[maxind], wl_1_corr

    except pyasl.PE.PyAValError:
        print "Failed cross correlation - No RV shift"

    return 0, wl_1


def find_continuum(wl, flux):
    #
    # Defines continuum points as the ones above the median
    # Calculate the threshold, standard deviation and mean
    #
    """
    # IDEA: make a second iteration
    # of the continuum definition so it will be better
    # def define_cont(flux_overlapping, value):
    # return continuum, threshold
    """

    threshold = 0.
    mean_cont = 0.
    std_cont = 0.

    abv_median = np.where(flux > np.median(flux))[0]
    if any([abv_median.size == 0, wl.size == 0]):
        print 'No continuum definition, empty array'
        pass
    elif wl[-1]-wl[0] < 3.:
        print 'No continuum definition, overlap < 3A'
        pass
    else:
        possible_cont = continuous_nb_in_list(abv_median.tolist())
        idx_cont = max(possible_cont, key=len)
        # Stats on the continuum
        mean_cont = np.mean(flux[idx_cont])
        std_cont = np.std(flux[idx_cont])
        threshold = mean_cont - np.sqrt(std_cont)
        # plt.plot(wl_i[idx_cont], flux_i[idx_cont], 'yo')

    print 'Threshold at: ', threshold

    return threshold, mean_cont, std_cont


def suppress_artefact(wl, flux, jump=0.15, interact=False):
    """
    -- Use with parsimony --
    Deletes artefacts (sharp peaks) in the spectrum
    if the difference in flux btw two consecutive points is larger than JUMP

    INPUTS: WL, FLUX = arrays wavelength and flux of the spectrum
            JUMP = threshold of the difference btw two consecutive points
            INTERACT = bool, False by default
                       if True plot is shown and waits for confirmation by user

    OUTPUTS: NEW_WL, NEW_FLUX = modified wavelength and flux arrays
    """
    index = []
    # Whats the difference with ediff1 in wrange_exclude ???
    diff = np.diff(flux)
    artefact = [(wl[i], i) for i in range(len(diff)) if abs(diff[i]) > jump]
    for i in range(len(artefact)):
        print '- WL artefact: ', artefact[i][0]
        if interact:
            plt.figure(i+10)
            plt.plot(wl, flux, 'mo-')
            plt.title('Is it an artefact?')
            plt.axis([artefact[0][0]-5, artefact[0][0]+5, 0, 1])
            yes = raw_input(" >> If it's an artefact, type yes: ")
            if str(yes) == 'yes':
                print 'Deleting 5 points ... !'
                index.append((artefact[i][1]-2, artefact[i][1]-1, artefact[i][1], artefact[i][1]+1, artefact[i][1]+2))
        else:
            print 'Deleting 5 points ... !'
            index.append((artefact[i][1]-2, artefact[i][1]-1, artefact[i][1], artefact[i][1]+1, artefact[i][1]+2))

    new_wl = np.delete(wl, index)
    new_flux = np.delete(flux, index)

    return new_wl, new_flux


def mean_spectrum(wl_right, flux_right, wl_left, flux_left, mean_cont, std_cont):
    """
    Calculates the mean spectrum of two overlapping spectra
    and chooses the spectrum which does not decrease in brightness at the edges
    INPUTS: WL_0, FLUX_0 = wavelength and flux of the spectrum 0
            WL_1, FLUX_1 = wavelength and flux of the spectrum 1
            THRESHOLD = value below the continuum and its noise

    OUTPUTS: NEW_FLUX = new spectrum on the overlapping region
    """
    # Interpolate spectra left and right
    fct_left = interp1d(wl_left, flux_left)
    fct_right = interp1d(wl_right, flux_right)

    # Define overlapping wavelength range
    if wl_right[0] > wl_left[0]:
        wl_start = wl_right[0]
    else:
        wl_start = wl_left[0]

    if wl_right[-1] > wl_left[-1]:
        wl_end = wl_left[-1]
    else:
        wl_end = wl_right[-1]

    new_wl = np.arange(wl_start, wl_end, 0.01)

    # Define the commun flux
    flux_start = []
    flux_end = []
    flux_middle = []

    idx_start = 0
    idx_end = len(new_wl)-1

    for idx, wl in enumerate(new_wl):
        if std_cont == 0.:
            break
        elif abs(fct_right(wl)-fct_left(wl)) > std_cont:
                flux_start.append(fct_left(wl))
        else:
            idx_start = idx
            break

    for idx, wl in reversed(list(enumerate(new_wl))):
        if std_cont == 0.:
            break
        elif abs(fct_right(wl) - fct_left(wl)) > std_cont:
            flux_end.insert(0, fct_right(wl))
        else:
            idx_end = idx
            break

    for idx, wl in enumerate(new_wl[idx_start:idx_end+1], start=idx_start):
        mean_flux = np.mean([fct_left(wl), fct_right(wl)])
        flux_middle.append(mean_flux)

    new_flux = np.concatenate([flux_start, flux_middle, flux_end])

    for i, wl in reversed(list(enumerate(wl_left))):
        if wl < new_wl[0]:
            new_wl = np.insert(new_wl, 0, wl)
            new_flux = np.insert(new_flux, 0, flux_left[i])
    for i, wl in enumerate(wl_right):
        if wl > new_wl[-1]:
            new_wl = np.append(new_wl, wl)
            new_flux = np.append(new_flux, flux_right[i])

    return new_wl, new_flux


class Spectrum(object):
    def __init__(self, name, header, wavelength, flux, non_corr_flux, cont):
            self.name = name
            self.header = header
            self.wavelength = wavelength
            self.flux = flux
            self.non_corr_flux = non_corr_flux
            self.cont = cont

    @classmethod
    def from_file(cls, filename):
        """
        # Create an object of the Spectrum class from a CARMENES FITS file
        # INPUTS: CLS = instance created when Spectrum class is called,
        #                 before init function
        #         FILENAME = name of the CARMENES FITS file
        # OUTPUT: SPECTRUM = object of the Spectrum class
        """
        print cls
        name = filename.strip()
        hdu = fits.open(filename.strip())
        header = hdu[0].header
        wavelength = hdu[4].data
        flux = hdu[1].data
        non_corr_flux = hdu[3].data
        cont = hdu[2].data
        spectrum = cls(name, header, wavelength, flux, non_corr_flux, cont)
        return spectrum

    def mask_overlap_rgn(self):
        """
        # Create a mask for the overlapping orders of CARMENES
        # Apply this mask to the wavelengths and fluxes
        # INPUTS:  SELF
        # OUTPUTS: WAVES = masked array of the wl
        #          FLUXES = masked array of the flux
        #          COUNT = number of overlapping orders
        #          MASK = mask (redundant bc included in the masked arrays)
        """
        count = 0
        for i in range(self.wavelength.shape[0]):
            # Identify the overlapping regions on the wavelengths
            # First wavelength array
            if i == 0:
                wave_start = ma.masked_where(
                    self.wavelength[i] > self.wavelength[i+1][0],
                    self.wavelength[i], copy=True)
                waves = wave_start
                print 'First wave', waves.shape
                if np.sum(wave_start.mask) != 0:
                    count += 1
                    print 'How many masked elements?', np.sum(wave_start.mask)

            # Last wavelength array
            elif i == self.wavelength.shape[0]-1:
                wave_end = ma.masked_where(
                    self.wavelength[i] < self.wavelength[i-1][-1],
                    self.wavelength[i], copy=True)
                waves = ma.vstack((waves, wave_end))
                print 'Last wave', waves.shape
                if np.sum(wave_end.mask) != 0:
                    count += 1
                    print 'How many masked elements?', np.sum(wave_end.mask)
            # All the other ones
            else:
                wave_mid = ma.masked_where(
                    (self.wavelength[i] > self.wavelength[i+1][0]) |
                    (self.wavelength[i] < self.wavelength[i-1][-1]),
                    self.wavelength[i], copy=True)
                waves = ma.vstack((waves, wave_mid))
                # print 'Middle', waves.shape
                if np.sum(wave_mid.mask) != 0:
                    count += 1
                    # print 'How many masked elements?', np.sum(wave_mid.mask)

        # count -1 bc it counts the orders instead of the overlaps
        print 'Number of overlapping orders: ', count-1
        # Apply the mask to the flux
        fluxes = ma.array(self.flux, mask=waves.mask)

        return waves, fluxes, count-1, waves.mask

    def treat_overlap(self, wavelength_masked, flux_masked, nb_overlaps):
        """
        Create an unique spectrum for each overlapping regions
        ---------------------------------------------------------------------
        Steps:
        - Delete artefact found in the data
        - Find a Doppler shift in the first two overlapping regions
        (after the shift seems too small to be found,
        I guess resampling is needed... YET TO BE DONE !)
        - Define a very basic continuum and a threshold under which
        the decrease in brightness at the edges of the orders are avoided
        - Take the mean spectrum in the overlapping region
        - Put together the corrected overlapping regions
        and the rest of the spectrum
        ---------------------------------------------------------------------
        INPUTS: WAVELENGTH_MASKED, FLUX_MASKED = masked arrays of the CARMENES spectrum
                NB_OVERLAPS = total number of overlapping regions

        OUTPUTS: WAVES_CORRECT, FLUXES_CORRECT = corrected CARMENES spectrum
        """
        idx_true = np.where(wavelength_masked.flatten().mask == True)
        group_idx = continuous_nb_in_list(idx_true[0])
        good_overlap = []
        plt.figure(1)
        for i in range(nb_overlaps):
            print '\n-- Overlap nb:', i+1, '--'
            # WL after which the corrected flux should be inserted
            wl_insert = spectrum10.wavelength.flatten()[group_idx[i][-1]+1]
            print 'WL insert: ', wl_insert

            # Find the two orders which overlap
            wl_left, flux_left, wl_right, flux_right = two_spectra_overlap(wavelength_masked, flux_masked, i, nb_overlaps)

            # Suppress artefact in the overlapping region
            wl_left, flux_left = suppress_artefact(wl_left, flux_left, interact=False)
            wl_right, flux_right = suppress_artefact(wl_right, flux_right, interact=False)

            # Plotting
            plt.plot(wl_right, flux_right, 'k.-', label='Right')
            plt.plot(wl_left, flux_left, 'b.-', label='Left')

            # RV shift
            if i == 0 or i == 1:
                # RV shift in the first two overlapping regions
                # !! Shift seems too small in the following orders, need to be IMPROVED !!
                shift, wl_left = rv_shift(wl_right, flux_right, wl_left, flux_left)
                plt.plot(wl_left, flux_left, 'go-', label='Left RV shift')

            # Find the continuum mean and std in the overlapping region
            threshold, mean, std = find_continuum(wl_left, flux_left)
            # Computes the 'mean' spectrum in the overlapping region
            new_wl, new_flux = mean_spectrum(wl_right, flux_right, wl_left, flux_left, mean, std)

            # Save the correct overlap and the position (wl_insert)
            good_overlap.append([wl_insert, new_wl, new_flux])

            # Plotting
            #plt.legend()

        wl_correct, flux_correct = corrected_overlap(wavelength_masked, flux_masked, good_overlap)
        plt.plot(wl_correct, flux_correct, 'm.--', label='Corrected spectrum')
        plt.legend()

        return wl_correct, flux_correct

    def wrange_exclude(self, waves, fluxes):
        """
        # Create the exclusion range in wavelength for Molecfit
        # INPUTS :
        # OUTPUTS: None
        #          Write wavelength_exclude.dat file used by Molecfit
        """
        waves = self.wavelength.flatten()
        # Wavelength without the overlapping regions
        waves_clean = waves.compressed()
        # Differences btw two consecutive elmts
        diff = np.ediff1d(waves_clean)
        # Mask is True when the WL are far to each other
        mask = [diff[i] > 10.*np.mean(diff) for i in range(diff.shape[0])]
        diff_masked = ma.masked_array(diff, mask=mask)

        # Wavelength ranges to exclude
        # [expression for i in list if condition]
        # WL should already be in microns, Molecfit doesnt convert the masks
        wranges = [(waves_clean[i]*0.0001, waves_clean[i+1]*0.0001)
                   for i in range(diff_masked.shape[0])
                   if diff_masked.mask[i]]

        print 'Writing wrange_exclude.dat ...'
        with open('wrange_exclude.dat', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(wranges)

        for i in range(len(wranges)):
            plt.plot((wranges[i][0]*10000., wranges[i][0]*10000.),
                     (-0.5, 1.5), 'k--')
            plt.plot((wranges[i][1]*10000., wranges[i][1]*10000.),
                     (-0.5, 1.5), 'b--')

        plt.plot(spectrum.wavelength.flatten(), spectrum.flux.flatten(), 'g-')
        plt.xlabel('Wavelength in microns')
        plt.ylabel('Flux')
        plt.title('CARMENES spectrum')
        return None

    def wrange_include(self, h2o=True, o2=False, co2=True, ch4=False):
        """
        # Create the inclusion range in wavelength for Molecfit
        # INPUTS :
        # OUTPUTS: None
        #          Write wavelength_include.dat file used by Molecfit
        """

        wranges = []
        wrange_h2o = [1.10, 1.12]
        wrange2_h2o = [1.34, 1.36]
        wrange_o2 = [1.26, 1.29]
        wrange_co2 = [1.56, 1.64]
        wrange_ch4 = [1.64, 1.72]

        if h2o:
            wranges.append(wrange_h2o)
            wranges.append(wrange2_h2o)
        if o2:
            wranges.append(wrange_o2)

        if co2:
            wranges.append(wrange_co2)

        if ch4:
            wranges.append(wrange_ch4)

        print 'Writing wrange_include.dat ...'
        with open('wrange_include.dat', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(wranges)

        return None

    def input_molecfit(self):
        # Creates an input file readable for Molecfit
        # Bin Table with data
        tb_hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name='WAVE', format='1D', array=self.wavelength.flatten()),
             fits.Column(name='SPEC', format='1D', array=self.flux.flatten()),
             fits.Column(name='CONT', format='1D', array=self.cont.flatten()),
             fits.Column(name='SIG', format='1D', array=self.non_corr_flux.flatten())])

        # Header
        head_hdu = fits.PrimaryHDU(header=self.header)
        hdu_list = fits.HDUList([head_hdu, tb_hdu])

        print('Writing input_file_molecfit.fits ... ')
        hdu_list.writeto('input_file_molecfit.fits', clobber=True)

        return None


    # def estimate_snr(self):
    #     print('SNR calculation ...')
    #     mean = np.mean([self.flux[i] for i in range(self.wavelength.shape[0])])
    #     std = np.std([self.flux[i] for i in range(self.wavelength.shape[0])])
    #     print('snr =', mean / std)
    #     return mean / std
    #     # can use self.wave, self.flux
    #     # does the plot


class Uves(Spectrum):
    pass


def add(a, b, minus=False):
    if b is not None:
        if minus:
            return a-sum(b)
        return a + sum(b)
    else:
        return a


# @Gooey(default_size=(610, 710))
# def arguments():
#     parser = argparse.ArgumentParser(description='Simple calculator.')
#     parser.add_argument('a', help='First number to add', type=float)
#     parser.add_argument('--extra', '-e', help='Second number to add',
#                         type=float, nargs='+')
#     parser.add_argument('--minus', '-m', help='Switch to subtraction',
#                         default=False, action="store_true")

#     args = parser.parse_args()
#     return args

if __name__ == "__main__":

    # Create Spectrum object from filename
    filename = 'car-20160420T20h45m44s-sci-cabj-nir_A.fits'
    filename2 = 'car-20160420T20h26m49s-sci-cabj-nir_A.fits'
    spectrum10 = Spectrum.from_file(filename)
    spectrum11 = Spectrum.from_file(filename2)
    
    # Mask the overlaps
    wavelength_masked, flux_masked, nb_overlaps, mask = Spectrum.mask_overlap_rgn(spectrum10)

    # Correct the overlaps
    wl, flux = Spectrum.treat_overlap(spectrum10, wavelength_masked, flux_masked, nb_overlaps)

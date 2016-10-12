# ===================================
# COMPARE Tapas, Telfit, Molecfit
# plotting the transmission spectra
#
# Solene 14.06.2016
# ===================================
#
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import linalg as LA
#
#
# TAPAS
# wl and flux classed decreasing, reverse array:  array[::-1]
file_tapas = '/home/solene/atmos/tapas/crires1203/tapas_000001.ipac'
rawwl_tapas, rawtrans_tapas = np.loadtxt(file_tapas, skiprows=38, unpack=True)
wl_tapas = rawwl_tapas[::-1]
trans_tapas = rawtrans_tapas[::-1]

# MOLECFIT
#
file_molecfit = '/home/solene/atmos/For_Solene/1203nm/output/molecfit_crires_solene_tac.fits'
hdu_molecfit = fits.open(file_molecfit)
data_molecfit = hdu_molecfit[1].data
cols_molecfit = hdu_molecfit[1].columns
# cols_molecfit.info()
rawwl_molecfit = data_molecfit.field('mlambda')
wl_molecfit = rawwl_molecfit*10e2
trans_molecfit = data_molecfit.field('mtrans')
cflux_molecfit = data_molecfit.field('cflux')

# TELFIT
#
file_telfit = '/home/solene/atmos/trans_telfit.txt'
wl_telfit, trans_telfit, wl_datatelfit, flux_datatelfit = np.loadtxt(
    file_telfit, unpack=True)


# Cross-correlation
# from PyAstronomy example
#
# TAPAS is the "template" shifted to match Molecfit
rv, cc = pyasl.crosscorrRV(
    wl_molecfit, trans_molecfit, wl_tapas, trans_tapas,
    rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=50)

maxind = np.argmax(cc)
print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")


# Doppler shift TAPAS
#
wlcorr_tapas = wl_tapas * (1. + rv[maxind]/299792.)
# transcorr_tapas, wlcorr_tapas = pyasl.dopplerShift(
#     wl_tapas[::-1], trans_tapas[::-1], rv[maxind],
#     edgeHandling=None, fillValue=None)  # Fancy way


# RMS between two spectra TAPAS, MOLECFIT
# do the same with the data and try to better fit the continuum with molecfit
# Selecting 2nd detector only
# USELESS
wlstart = wl_datatelfit[0]
wlend = wl_datatelfit[-1]
ind_molecfit = np.where((wl_molecfit > wlstart) & (wl_molecfit < wlend))
wl_molecfit2 = wl_molecfit[ind_molecfit]
trans_molecfit2 = trans_molecfit[ind_molecfit]

ind_tapas = np.where((wl_tapas > wlstart) & (wl_tapas < wlend))
wl_tapas2 = wl_tapas[ind_tapas]
trans_tapas2 = trans_tapas[ind_tapas]

# Interpolation
# f_molecfit = interp1d(wl_molecfit, trans_molecfit, kind='cubic')  # takes forever...
# wlcorr_tapasnew = wlcorr_tapas[500:-500]  # raw adjustment of the wl limits
# plt.plot(wl_molecfit, trans_molecfit, 'o', wlcorr_tapasnew, f_molecfit(wlcorr_tapasnew), '.')

f_molecfit = interp1d(wl_molecfit, trans_molecfit)#  , kind='cubic')  # takes forever...
f_tapas = interp1d(wlcorr_tapas, trans_tapas)

# Euclidean distance at each point
stack_molecfit = np.stack((flux_datatelfit, f_molecfit(wl_datatelfit)), axis=-1)
stack_tapas = np.stack((flux_datatelfit, f_tapas(wl_datatelfit)), axis=-1)
norm_molecfit = LA.norm(stack_molecfit, axis=1)
norm_tapas = LA.norm(stack_tapas, axis=1)
# trans_stack = np.stack((trans_tapas[500:-500], f_molecfit(wlcorr_tapasnew)), axis=-1)
# norm_trans = LA.norm(trans_stack, axis=1)
plt.plot(wl_datatelfit, norm_tapas, 'r.')  # see that the continuum is offset 1.4
plt.plot(wl_datatelfit, norm_molecfit, 'k.')
# RMS
err_molec = flux_datatelfit - f_molecfit(wl_datatelfit)
err_tapas = flux_datatelfit, f_tapas(wl_datatelfit)

rms_molec = sqrt(mean_squared_error(flux_datatelfit, f_molecfit(wl_datatelfit)))
rms_tapas = sqrt(mean_squared_error(flux_datatelfit, f_tapas(wl_datatelfit)))


# Plotting
#
plt.figure(1)
plt.subplot(211)
plt.plot(wl_datatelfit, flux_datatelfit, 'g.-', label='Data 2nd detector')
plt.plot(wl_molecfit, trans_molecfit, 'r-', label='Molecfit')
plt.plot(wl_tapas, trans_tapas, 'b-', label='Tapas')
plt.title('Comparison atmospheric transmission \n CRIRES data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend(loc=3.)
plt.subplot(212)
# plt.plot(wl_tapas, trans_tapas, 'b-', label='Tapas')
plt.plot(wl_datatelfit, flux_datatelfit, 'g.-', label='Data 2nd detector')
plt.plot(wl_molecfit, trans_molecfit, 'r-', label='Molecfit')
plt.plot(wlcorr_tapas, trans_tapas, 'b--', label='Tapas corrected')

# plot 2nd detector only with WL from the data
plt.plot(wl_datatelfit, flux_datatelfit, 'g.-', label='Data 2nd detector')
plt.plot(wl_datatelfit, f_molecfit(wl_datatelfit), 'r-', label='Molecfit')
plt.plot(wl_datatelfit, f_tapas(wl_datatelfit), 'b--', label='Tapas corrected')
# plt.plot(wl_telfit, trans_telfit, 'r-', label='Telfit')

plt.plot(wl_datatelfit, (flux_datatelfit - f_tapas(wl_datatelfit)), 'b.', label='Tapas residuals')
plt.plot(wl_datatelfit, (flux_datatelfit - f_molecfit(wl_datatelfit)), 'r.', label='Molecfit residuals')

plt.plot(wl_datatelfit, flux_datatelfit, 'g.-', label='Data 2nd detector')
plt.plot(wl_molecfit, trans_molecfit, 'r-', label='Molecfit')
plt.plot(wl_molecfit, cflux_molecfit, 'b-', label='Corrected data - Molecfit')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
# plt.plot(model.x, model.y, 'k-', label='Gaussian fit')
# $\mu=%.2f, \sigma=%.2f$' %(wavestart, waveend)
plt.legend(loc=3.)
plt.show()

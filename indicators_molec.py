# ===================================
# CALCULATES Ioff and Ires
# Indicators described in Molecfit II
#
# Solene 20.09.2016
# ===================================
#
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
# from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from numpy import linalg as LA

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

# Interpolation
f_molecfit = interp1d(wl_molecfit, cflux_molecfit,  kind='cubic')
ftrans_molecfit = interp1d(wl_molecfit, trans_molecfit,  kind='cubic')
# f_tapas = interp1d(wlcorr_tapas, trans_tapas)

# **1** BINNED DATA
#  3 delta-lambda = 0.036
# Mean and std deviation of bins on the telluric CORRECTED spectrum
fluxmean_bin_means, bin_edges, binnumber = stats.binned_statistic(
    wl_datatelfit, f_molecfit(wl_datatelfit), statistic='mean',
    bins=np.floor((wl_datatelfit[-1]-wl_datatelfit[0])/0.036))

fluxstd_bin_means, _, _ = stats.binned_statistic(
    wl_datatelfit, f_molecfit(wl_datatelfit), statistic=np.std,
    bins=np.floor((wl_datatelfit[-1]-wl_datatelfit[0])/0.036))

bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

# **2** Bins where average TRANSMISSION is > 0.99
flux_trans_mean_bin_means, _, _ = stats.binned_statistic(
    wl_datatelfit, ftrans_molecfit(wl_datatelfit), statistic='mean',
    bins=np.floor((wl_datatelfit[-1]-wl_datatelfit[0])/0.036))
# cont_bin_means = flux_trans_mean_bin_means[flux_trans_mean_bin_means > 0.99]
ind_cont = np.where(flux_trans_mean_bin_means > 0.99)
ind_out = np.where((flux_trans_mean_bin_means < 0.95) &
                   (flux_trans_mean_bin_means > 0.1))

# plt.plot(bin_centers[ind_cont], flux_trans_mean_bin_means[ind_cont], 'kx')

# **3** Interpolation of the continuum cubic
# f_cont = interp1d(bin_centers[ind_cont], flux_trans_mean_bin_means[ind_cont],  kind='cubic')
# Extrapolation with constant value spline
f_cont = InterpolatedUnivariateSpline(
    bin_centers[ind_cont], flux_trans_mean_bin_means[ind_cont], ext=3)
# bbox=[bin_centers[ind_cont][0], bin_centers[ind_cont][-1]],


# **5** Subtract cont to mean flux
# and Divide offset and std by interpolated continuum mean value
sys_offset = (fluxmean_bin_means - f_cont(bin_centers)) / f_cont(bin_centers)
flux_std = fluxstd_bin_means / f_cont(bin_centers)

# **6** independant WL = Divide by average absorption
absorp_molecfit = 1 - flux_trans_mean_bin_means
sys_offset_final = sys_offset / absorp_molecfit
flux_std_final = flux_std / absorp_molecfit

plt.figure(1)
plt.plot(wl_datatelfit, flux_datatelfit, 'b.-', label='Raw data')
# plt.hlines(flux_bin_means, bin_edges[:-1],
#            bin_edges[1:], colors='g', lw=5, label='binned statistic of data')
plt.plot(bin_centers, fluxmean_bin_means, 'rx-', label='Mean binned data')
plt.plot(bin_centers, fluxstd_bin_means, 'kx-', label='Standard deviation binned data')
plt.legend()

plt.figure(2)
plt.plot(wl_datatelfit, flux_datatelfit, 'g.-', label='Data 2nd detector')
plt.plot(wl_molecfit, trans_molecfit, 'r-', label='Molecfit')
plt.plot(wl_datatelfit, f_molecfit(wl_datatelfit),
         'b-', label='Corrected data - Molecfit')
plt.plot(wl_datatelfit, f_cont(wl_datatelfit),
         'k-', label='Interpolated Continuum')
plt.plot(sys_offset_final[ind_out], flux_std_final[ind_out], 'kx')
plt.plot(flux_trans_mean_bin_means[ind_out],
         sys_offset_final[ind_out], 'kx', label='Ioff vs Transmission')
plt.plot(flux_trans_mean_bin_means[ind_out],
         flux_std_final[ind_out], 'r.', label='Ires vs Transmission')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend(loc=3.)
plt.show()

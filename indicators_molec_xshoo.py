# ===================================
# CALCULATES Ioff and Ires
# Indicators described in Molecfit II
# For X SHOOTER SPECTRA RAQUEL
# Solene 30.09.2016
# ===================================
#
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats

# MOLECFIT
#
file_molecfit = '/home/solene/atmos/raquel_xshooter/output/molecfit_xshoo_raquel2_tac.fits'
hdu_molecfit = fits.open(file_molecfit)
data_molecfit = hdu_molecfit[1].data
cols_molecfit = hdu_molecfit[1].columns
# cols_molecfit.info()
raw_wl_molecfit = data_molecfit.field('lambda')*10e2  # input wl
raw_flux_molecfit = data_molecfit.field('flux')*10e2  # input flux
wl_molecfit = data_molecfit.field('mlambda')*10e2     # corrected wl
trans_molecfit = data_molecfit.field('mtrans')        # transmission flux
cflux_molecfit = data_molecfit.field('cflux')*10e2    # corrected flux
# np.sum(np.isnan(cflux_molecfit))                      # check for NaN values

# Interpolation
f_molecfit = interp1d(wl_molecfit, cflux_molecfit)  # ,  kind='cubic')
ftrans_molecfit = interp1d(wl_molecfit, trans_molecfit)  # ,  kind='cubic')

#  BIN DATA
#  3 delta-lambda = 1.07
# Mean and std deviation of bins on the telluric CORRECTED spectrum
delta = 2.142  #  5delta
fluxmean_bin_means, bin_edges, binnumber = stats.binned_statistic(
    wl_molecfit, cflux_molecfit, statistic='mean',
    bins=np.floor((wl_molecfit[-1]-wl_molecfit[0])/delta))

fluxstd_bin_means, _, _ = stats.binned_statistic(
    wl_molecfit, cflux_molecfit, statistic=np.std,
    bins=np.floor((wl_molecfit[-1]-wl_molecfit[0])/delta))

bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

# Bins where average TRANSMISSION is > 0.99
flux_trans_mean_bin_means, _, _ = stats.binned_statistic(
    wl_molecfit, trans_molecfit, statistic='mean',
    bins=np.floor((wl_molecfit[-1]-wl_molecfit[0])/delta))

ind_cont = np.where(flux_trans_mean_bin_means > 0.99)
ind_out = np.where((flux_trans_mean_bin_means < 0.95) &
                   (flux_trans_mean_bin_means > 0.1))

# plt.plot(bin_centers[ind_cont], flux_trans_mean_bin_means[ind_cont], 'kx')

# INTERPOLATION CONTINUUM
# Interpolation polynomial
f_cont = interp1d(bin_centers[ind_cont], fluxmean_bin_means[ind_cont], kind='linear', bounds_error=False, fill_value=(fluxmean_bin_means[ind_cont][0], fluxmean_bin_means[ind_cont][-1]))

# Extrapolation with constant value spline
# f_cont = InterpolatedUnivariateSpline(
#     bin_centers[ind_cont], fluxmean_bin_means[ind_cont], ext=3)

# Subtract cont to mean flux
# and Divide offset and std by interpolated continuum mean value
sys_offset = (fluxmean_bin_means - f_cont(bin_centers)) / f_cont(bin_centers)
flux_std = fluxstd_bin_means / f_cont(bin_centers)

# Independant WL = Divide by average absorption
absorp_molecfit = 1 - flux_trans_mean_bin_means
sys_offset_final = sys_offset / absorp_molecfit
flux_std_final = flux_std / absorp_molecfit


# PLOTTING
# Figure 2 in Molecfit II Mean+Std
plt.figure(1)
plt.plot(raw_wl_molecfit, raw_flux_molecfit, 'g.-', label='Raw data')
# plt.hlines(flux_bin_means, bin_edges[:-1],
#            bin_edges[1:], colors='g', lw=5, label='binned statistic of data')
plt.plot(bin_centers, fluxmean_bin_means, 'rx-', label='Mean binned data')
plt.plot(bin_centers, fluxstd_bin_means, 'kx-', label='Standard deviation binned data')
plt.legend()

# Indicators Ioff and Ires
plt.figure(2)
plt.plot(raw_wl_molecfit, raw_flux_molecfit, 'g.-', label='Raw data')
plt.plot(wl_molecfit, trans_molecfit, 'r-', label='Molecfit')
plt.plot(wl_datatelfit, f_molecfit(wl_datatelfit),
         'b-', label='Corrected data - Molecfit')
plt.plot(wl_datatelfit, f_cont(wl_datatelfit),
         'k-', label='Interpolated Continuum')
plt.plot(sys_offset_final[ind_out], flux_std_final[ind_out], 'kx')
plt.plot(flux_trans_mean_bin_means[ind_out],
         sys_offset_final[ind_out], 'kx', label='Ioff')
plt.plot(flux_trans_mean_bin_means[ind_out],
         flux_std_final[ind_out], 'r.', label='Ires')

# Selected continuum points
plt.figure(3)
plt.plot(wl_molecfit, cflux_molecfit, 'k-', label='Corrected data')
plt.plot(bin_centers[ind_cont], fluxmean_bin_means[ind_cont], 'ro', label='Continuum points')
plt.plot(bin_centers, fluxmean_bin_means, 'b', label='Mean data')
plt.plot(bin_centers[ind_cont], f_cont(bin_centers[ind_cont]), 'b-')

# Figure 3 in Molecfit II shared axis
f = plt.figure()
plt.subplots_adjust(hspace=0.001)
ax1 = plt.subplot(211)
ax1.plot(flux_trans_mean_bin_means[ind_out],sys_offset_final[ind_out], 'k.', label='Ioff')
plt.ylim(-1, 1)
plt.ylabel('Ioff')
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(flux_trans_mean_bin_means[ind_out],flux_std_final[ind_out], 'r.', label='Ires')
plt.ylabel('Ires')
plt.ylim(-1, 2)
plt.xlabel('Transmission')
xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
plt.setp(xticklabels, visible=True)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend(loc=3.)
plt.show()

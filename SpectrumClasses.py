import astropy.units as u
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from lmfit import Model
from lmfit.models import LinearModel, VoigtModel
from bisect import bisect_left
from scipy.special import voigt_profile
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
from astropy.convolution import Gaussian1DKernel, convolve

class Spectrum:
    """Spectrum processing tools and functions."""

    def __init__(self,wavelength,flux):
        linear_model = LinearModel(prefix = 'l_')
        self.params = linear_model.make_params()
        cm_voigt_model = VoigtModel(prefix = 'v_')
        self.params.update(cm_voigt_model.make_params())
        self.cm = linear_model - cm_voigt_model
        self.params['v_amplitude'].set(value = 150)
        self.params['v_sigma'].set(value = 5)
        self.params['l_intercept'].set(value = 25)
        self.params['l_slope'].set(value = 0)

    def normalize_individual_line(self,line,plot=False,ivar=True,central=False):
        wl = self.wavelength
        fl = self.flux

        distance = hdistances[line]
        centroid = hlines[line]
        self.params['v_center'].set(value = centroid)
        crop1 = bisect_left(wl, centroid - distance)
        crop2 = bisect_left(wl, centroid + distance)

        cropped_wl = wl[crop1:crop2]
        cropped_fl = fl[crop1:crop2]
        if ivar is True:
            iv = self.ivar
            cropped_iv = iv[crop1:crop2]
            weight = np.sqrt(cropped_iv)
        else:
            weight = np.ones(len(cropped_wl))

        try:
            result = self.cm.fit(cropped_fl, self.params, x = cropped_wl, nan_policy = 'omit',
                                 weights=weight)
            message = result.message
            if message != "Fit succeeded.":
                print(f"Fitting of line {line} did not succeeded")
                return None, None, None, False

        except Exception as e:
            print(e)
            return None, None, None, False

        centroid = result.params['v_center']
        self.hlines_centroids[line] = np.round(centroid.value,3)
        crop1 = bisect_left(wl, centroid - distance)
        crop2 = bisect_left(wl, centroid + distance)

        cropped_wl = wl[crop1:crop2]
        cropped_fl = fl[crop1:crop2]

        slope = result.params['l_slope']
        intercept = result.params['l_intercept']
        continuum = (slope * cropped_wl + intercept)
        fl_normalized = cropped_fl / continuum
        mean_norm_flux = np.mean(fl_normalized)

        if mean_norm_flux < 0.75 or mean_norm_flux > 1.1:
            print(f"Line {line} ignored due to mean normalized flux being {mean_norm_flux}")
            return None, None, None, False

        if plot:
            plt.plot(cropped_wl-centroid, cropped_fl)
            plt.plot(cropped_wl-centroid, result.eval(result.params, x=cropped_wl))
            plt.plot(cropped_wl-centroid, cropped_wl*slope + intercept)
            plt.show()
            plt.plot(cropped_wl-centroid, fl_normalized)
            plt.show()

        if ivar is True:
            cropped_iv = iv[crop1:crop2]
            normalized_ivar = cropped_iv*continuum**2
        else:
            normalized_ivar = weight
        if central is True:
            cropped_wl = cropped_wl - centroid

        return cropped_wl, fl_normalized, normalized_ivar, True

    def normalize_balmer(self):
        """Continuum-normalization of any spectrum by fitting each line individually"""

        wavelength = self.wavelength
        flux= self.flux

        lines = ['alpha','beta','gamma','delta','epsilon','zeta']
        num_lines = len(lines)
        wavelength_norm = []
        flux_norm = []

        for index, line in enumerate(lines):
            wavelength_segment, flux_segment = self.normalize_individual_line(line)
            wavelength_norm = np.append(wavelength_segment, wavelength_norm)
            flux_norm = np.append(flux_segment, flux_norm)

        return wavelength_norm, flux_norm

    def line_eqw(self, line):
        """Returns the equivalent width of a given line on the Spectrum."""

        fit_eqw = self.fit_line(line)[0]
        eqw = (equivalent_width(fit_eqw)/u.AA).value

        return eqw

    def fit_lines(self):
        """Adjusts every line usgin Voigt Profiles. Returns a array"""

        fitted_parameters = np.ones([6,4])

        for i, line in enumerate(hlines.keys()):
            parameters = self.fit_line(line)[1]
            for j, parameter in enumerate(parameters):
                fitted_parameters[i,j] = parameter

        return fitted_parameters

    def fit_line(self, line, plot=False, centralized=False):
        """Adjusts a line using a Voigt Profile. Returns a Spectrum1D object"""
        
        target_line = self.normalize_line(line, centralized=centralized)
        wavelength_segment = np.array(target_line[0])
        flux_segment = target_line[1]
        fitted_line, fitted_params = self._fit_voigt(target_line, line)
        fitted_line_spectrum = Spectrum1D(flux=fitted_line*norm_unit,
                                          spectral_axis=wavelength_segment*u.AA)

        if plot:
            plt.plot(wavelength_segment,flux_segment,color='dimgrey')
            plt.plot(wavelength_segment,fitted_line,color='indianred')

            centroid = hlines[line]
            distance = hdistances[line]
            plt.xlim(centroid - distance, centroid + distance)

            plt.title(rf'Fitting of H$_\{line}$')
            plt.xlabel('Wavelength [A]')
            plt.ylabel('Normalized Flux')

            plt.show()

        return fitted_line_spectrum, fitted_params

    def _fit_voigt(self, spectrum, line):
        """Fits a voigt profile given a spectrum. Line gives the initial guess."""

        voigt_params = voigt_model.make_params(amplitude=-0.5, center=hlines[line], sigma=1.0,
                                               gamma=0.5)
        voigt_params['amplitude'].set(max=0)
        voigt_params['sigma'].set(min=0) 
        voigt_params['gamma'].set(min=0)        

        wavelength_segment = spectrum[0]
        adjusted_flux = spectrum[1] - 1.0
        voigt_fit = voigt_model.fit(adjusted_flux,voigt_params,x=wavelength_segment)

        fitted_line = voigt_fit.best_fit + 1
        fit_params = voigt_fit.params
        fitted_params = np.array([fit_params['amplitude'], fit_params['center'],
                                  fit_params['sigma'], fit_params['gamma']])

        return fitted_line, fitted_params

    def convolve_line(self, line, resolution, plot=False):
        """
        Convolves a given line of the spectra using a given
        Resolution R. It assumes a gaussian convolution.
        """

        wavelength, flux = self.normalize_line(line, centralized=False)

        if self.generic_model is True:
            line_space = {
                'alpha'   : 1286,
                'beta'    : 1157,
                'gamma'   : 778,
                'delta'   : 515,
                'epsilon' : 354,
                'zeta'    : 181
            }
            spline = scipy.interpolate.CubicSpline(wavelength.values,flux.values)
            wavelength = wavelength.reset_index(drop=True)
            wavelength = np.linspace(wavelength.values[0],wavelength.values[-1],
                                         line_space[line])
            flux = spline(wavelength)
            stddev = (wavelength[0]*10**(0.0001))/resolution
        else:
            stddev = (wavelength.iloc[0]*10**(0.0001))/resolution
        
        gaussian_kernel = Gaussian1DKernel(stddev=stddev)
        convoluted_flux = convolve(flux,gaussian_kernel,
                                   normalize_kernel=True,boundary='extend')

        if plot:
            plt.plot(wavelength, flux)
            plt.plot(wavelength, convoluted_flux)
            plt.show()

        return wavelength, convoluted_flux

    def plot_spectrum(self,spectral_space=[3800,7500],lines=False,radial_correction=False):
        wv_min, wv_max = spectral_space
        wavelength = self.wavelength
        crop1 = bisect_left(wavelength, wv_min)
        crop2 = bisect_left(wavelength, wv_max)
        wavelength = self.wavelength[crop1:crop2]
        flux = self.flux[crop1:crop2]

        upper_flux = flux.max()
        lower_flux = np.max([-0.5,0.95*flux.min()])
        max_plotted_flux = 1.05*upper_flux

        balmer_lines = [(r'H$_\alpha$','#ff0000', 6562.79),
                        (r'H$_\beta$','#00efff', 4861.35),
                        (r'H$_\gamma$','#2800ff', 4340.472),
                        (r'H$_\delta$','#7e00db', 4101.734),
                        (r'H$_\epsilon$','#8100a9', 3970.075),
                        (r'H$_\zeta$','#780088', 3889.064)]

        plt.plot(wavelength, flux, color='#2f2f2f', label='Spectra')

        if lines is True:
            if radial_correction is True:
                if not hasattr(self, 'radial_velocity'):
                    rad_vel = self.get_radial_velocity()
                else:
                    rad_vel = self.radial_velocity
            else:
                rad_vel = 0

            for line, color, wavelength_balmer in balmer_lines:
                if wavelength_balmer < wv_max and wavelength_balmer > wv_min:
                    delta_wavelength = wavelength_balmer*(rad_vel/299792.458)
                    plt.vlines(wavelength_balmer+delta_wavelength, lower_flux, max_plotted_flux,
                               color=color, linestyle='dashed', label=line, linewidth=0.85)
            plt.legend(loc='upper right', edgecolor='#2f2f2f')

        plt.title('Plotted Spectra')
        plt.xlabel('Wavelength [A]')
        if self.__class__.__name__ == "ModelSpectrum":
            flux_str = r'Flux [10$^{7}$'
        else:
            flux_str = r'Flux [10$^{-17}$'
        plt.ylabel(flux_str + r' erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]')

        plt.xlim(wv_min,wv_max)
        plt.ylim(lower_flux,max_plotted_flux)

        plt.show()

class RealSpectrum(Spectrum):
    """Child class. Inherits Spectrum, but with methods tailored towards real Spectra."""

    def __init__(self, spectral_data, simplified=False):
        self.wavelength = spectral_data['Wavelength']
        self.flux = spectral_data['Flux']
        self.simplified = simplified
        if simplified is False:
            self.ivar = spectral_data['Ivar']

        Spectrum.__init__(self, self.wavelength, self.flux)

        self.hlines_centroids = {'alpha'   : 6562.790,
                                 'beta'    : 4861.350,
                                 'gamma'   : 4340.472,
                                 'delta'   : 4101.734,
                                 'epsilon' : 3970.075,
                                 'zeta'    : 3889.064
                                 }

    def continuum_normalize(self, get_continuum=False, plot=False):
        """Continuum-normalization"""

        index_limit = np.where(self.wavelength<7500)[0][-1]
        wavelength = self.wavelength[:index_limit]
        flux = self.flux[:index_limit]
        try:
            stddev = np.sqrt(self.ivar[:index_limit])
        except:
            stddev = np.ones(flux.size)

        lines_mask = (
            ((wavelength > 6900) * (wavelength < 7500))    +\
            ((wavelength > 5060) * (wavelength < 6250))    +\
            ((wavelength > 4600) * (wavelength < 4630))    +\
            ((wavelength > 4195) * (wavelength < 4210))    +\
            ((wavelength > 4027) * (wavelength < 4032))    +\
            ((wavelength > 3925) * (wavelength < 3927))    +\
            ((wavelength > 3859) * (wavelength < 3860.5))  +\
            ((wavelength > 3000) * (wavelength < 3700))
        )            

        spl = scipy.interpolate.make_splrep(wavelength[lines_mask], flux[lines_mask], k = 1,
                                            s=500,w=stddev[lines_mask])
        continuum = scipy.interpolate.splev(wavelength, spl)
        
        normalized_flux = flux/continuum
        normalized_continuum = np.ones(len(wavelength))

        if plot:
            fig, ax = plt.subplots()


            ax.plot(wavelength, flux, label='Spectrum', color='#525252', linewidth=1)
            ax.plot(wavelength,continuum, label='continuum',
                    color='#df3030', linestyle='--', linewidth=1.5)

            plt.title("Fitted Continuum")
            ax.set_xlabel('Wavelength [A]')
            ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]')
            ax.set_xlim([3800,7500])

            ax.fill_between(wavelength, flux, continuum, where=(flux < continuum), color='gray',
                            alpha=0.3, label='Absorption Lines')

            plt.show()

            fig, ax = plt.subplots()

            ax.plot(wavelength, normalized_flux, label='Normalized Spectrum', color='#525252',
                    linewidth=1)
            plt.title("Normalized Spectrum")
            ax.set_xlabel('Wavelength [A]')
            ax.set_ylabel('Normalized Flux')
            ax.set_xlim([3800,7500])

            ax.fill_between(wavelength, normalized_flux, normalized_continuum,
                            where=(normalized_flux < normalized_continuum), color='gray',
                            alpha=0.3, label='Absorption Lines')

            plt.show()


        if get_continuum:
            return continuum
        else:
            return normalized_flux

    def normalize_line(self, line, plot=False, centralized=True, ivar=False):
        wavelength = self.wavelength
        normalized_flux = self.continuum_normalize()
        if self.simplified is not True:
            normalized_ivar = self.normalize_ivar()
        else:
            normalized_ivar = np.ones(len(wavelength))

        if centralized is not False:
            centroid = self.find_centroid(line)
        else:
            centroid = hlines[line]
        distance = hdistances[line]

        crop1 = bisect_left(wavelength, centroid - distance)
        crop2 = bisect_left(wavelength, centroid + distance)

        if centralized is True:
            line_wavelength = wavelength[crop1:crop2]-centroid
        else:
            line_wavelength = wavelength[crop1:crop2]
        line_flux = normalized_flux[crop1:crop2]
        line_ivar = normalized_ivar[crop1:crop2]

        if plot:
            plt.plot(line_wavelength, line_flux)
            plt.show()

        if ivar is True:
            return line_wavelength, line_flux, line_ivar
        return line_wavelength, line_flux
            

    def normalize_ivar(self):
        index_limit = np.where(self.wavelength<7500)[0][-1]
        ivar_array = self.ivar[:index_limit]
        continuum  = self.continuum_normalize(get_continuum=True)

        normalized_ivar = np.copy(ivar_array)
        for index, ivar in enumerate(normalized_ivar):
            normalized_ivar[index] = ivar_array[index]*continuum[index]**2

        return normalized_ivar


    def get_radial_velocity(self,lines=['alpha','beta','gamma','delta','epsilon','zeta']):
        c = 299792.458
        fitted_velocities = np.ones(len(lines))
        for index, line in enumerate(lines):
            self.normalize_individual_line(line)
            observed_centroid = self.hlines_centroids[line]
            rest_centroid = hlines[line]
            line_velocity = ((observed_centroid - rest_centroid)/rest_centroid)*c
            fitted_velocities[index] = line_velocity
        radial_velocity = np.median(fitted_velocities)

        self.radial_velocity = radial_velocity

        return radial_velocity

    def get_spectra_space(self,line):
        wavelength = self.wavelength
        centroid = hlines[line]
        distance = hdistances[line]

        crop1 = bisect_left(wavelength, centroid - distance)
        crop2 = bisect_left(wavelength, centroid + distance)

        spectra_space = wavelength[crop1:crop2]

        return spectra_space

    def plot_ivar(self,title=None):
        if not hasattr(self, 'ivar'):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute 'ivar'. "
                "This function only makes sense when ivar values are set. Please ensure"
                " simplified is not set as True"
            )

        wv = self.wavelength
        fl = self.flux
        iv = self.ivar

        fig, ax1 = plt.subplots()

        ax1.plot(wv,fl,color='black',label='Observed Spectrum')
        ax1.set_ylabel('Flux',fontsize=12)
        ax1.set_xlabel('Wavelength',fontsize=12)
        ax1.set_xlim(wv.min(),wv.max())
        ax1.set_ylim(-0.1,fl.max()*1.05)

        ax2 = ax1.twinx()
        ax2.semilogy(wv,iv,color='#004D40',alpha=0.5,label='Inverse Variance')
        ax2.set_ylabel('log(ivar)',fontsize=12)

        on_bad_pixel = False
        start_idx = None

        for i in range(len(iv)):
            if iv[i] == 0 and not on_bad_pixel:
                on_bad_pixel = True
                start_idx = i
            elif iv[i] != 0 and on_bad_pixel:
                ax1.axvspan(wv[start_idx], wv[i-1], color='#D81B60', alpha=0.6, lw=1)
                on_bad_pixel = False
        if on_bad_pixel:
            ax1.axvspan(wv[start_idx], wv.iloc[-1], color='#D81B60', alpha=0.6, lw=1)

        if np.sum(iv == 0) > 0:
            ax1.axvspan(0, 1, color='#D81B60', alpha=0.6, lw=0.5, label='Ivar = 0')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        ax1.legend(handles,labels,loc='upper left')
        if title is None:
            plt.title('Inverse Variance of Observed Spectra',fontsize=14)
        else:
            plt.title(title,fontsize=14)
        plt.show()

class Observation(RealSpectrum):
    """Granchild class. Inherits RealSpectrum, but assume observed info"""

    def __init__(self, spectra_file):
        self.filepath = spectra_file

        self._read_data()

        RealSpectrum.__init__(self, self.data, simplified=False)

        if self.origin == 'DESI':
            self.bprp = self.BPmag
        if self.origin == 'SDSS':
            self.ug   = self.umag - self.gmag
            self.gr   = self.gmag - self.rmag
            self.ri   = self.rmag - self.imag
            self.iz   = self.imag - self.zmag
            self.bprp = self.BPmag - self.RPmag

    def _read_data(self):
        with open(self.filepath,'r') as datafile:
            lines = datafile.readlines()

        self.origin = lines[0].split(': ')[1].strip()
        self.objid  = lines[1].split(': ')[1].strip()
        self.PMJ    = lines[2].split(': ')[1].strip()
        self.TYPE   = lines[3].split(': ')[1].strip()
        self.RA     = float(lines[4].split(': ')[1].strip())
        self.DEC    = float(lines[5].split(': ')[1].strip())
        self.sn     = float(lines[6].split(': ')[1].strip())
        self.umag   = float(lines[8].split(': ')[1].strip())
        self.gmag   = float(lines[9].split(': ')[1].strip())
        self.rmag   = float(lines[10].split(': ')[1].strip())
        self.imag   = float(lines[11].split(': ')[1].strip())
        self.zmag   = float(lines[12].split(': ')[1].strip())
        self.Gmag   = float(lines[14].split(': ')[1].strip())
        self.BPmag  = float(lines[15].split(': ')[1].strip())
        self.RPmag  = float(lines[16].split(': ')[1].strip())

        self.data = pd.read_csv(self.filepath, sep = ' ', skiprows=[x for x in range(0,18)],
                                names=['Wavelength','Flux','Ivar'])

class ModelSpectrum(Spectrum):
    """Child class. Inherits Spectrum, but with methods tailored towards model Spectra."""

    def __init__(self, spectral_data,generic=False):
        self.wavelength = spectral_data.Wavelength
        self.flux = spectral_data.Flux
        self.generic_model = generic

        Spectrum.__init__(self, self.wavelength, self.flux)

    def continuum_normalize(self, plot=False):
        """Continuum-normalization"""

        wavelength = self.wavelength
        flux = self.flux
        lines_mask = (
                    (wavelength > 10500)  +\
                    ((wavelength > 9750) * (wavelength < 9850))    +\
                    ((wavelength > 9300) * (wavelength < 9400))    +\
                    ((wavelength > 6900) * (wavelength < 7500))    +\
                    ((wavelength > 5060) * (wavelength < 6250))    +\
                    ((wavelength > 4600) * (wavelength < 4630))    +\
                    ((wavelength > 4195) * (wavelength < 4210))    +\
                    ((wavelength > 4027) * (wavelength < 4032))    +\
                    ((wavelength > 3925) * (wavelength < 3927))    +\
                    ((wavelength > 3859) * (wavelength < 3860.5))  +\
                    ((wavelength > 3000) * (wavelength < 3700))
                )

        spl = scipy.interpolate.make_splrep(wavelength[lines_mask], flux[lines_mask], k = 3)
        continuum = scipy.interpolate.splev(wavelength, spl)
        
        normalized_flux = flux/continuum
        normalized_continuum = np.ones(len(wavelength))

        if plot:
            fig, ax = plt.subplots()


            ax.plot(wavelength, flux, label='Spectra', color='#525252', linewidth=1)
            ax.plot(wavelength,continuum, label='continuum',
                    color='#df3030', linestyle='--', linewidth=1.5)

            plt.title("Fitted Continuum")
            ax.set_xlabel('Wavelength [A]')
            ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]')
            ax.set_xlim([3800,7500])

            ax.fill_between(wavelength, flux, continuum, where=(flux < continuum), color='gray',
                            alpha=0.3, label='Absorption Lines')

            plt.show()

            fig, ax = plt.subplots()

            ax.plot(wavelength, normalized_flux, label='Normalized Spectrum', color='#525252',
                    linewidth=1)
            plt.title("Normalized Spectrum")
            ax.set_xlabel('Wavelength [A]')
            ax.set_ylabel('Normalized Flux')
            ax.set_xlim([3800,7500])

            ax.fill_between(wavelength, normalized_flux, normalized_continuum,
                            where=(normalized_flux < normalized_continuum), color='gray',
                            alpha=0.3, label='Absorption Lines')

            plt.show()


        return normalized_flux

    def normalize_line(self, line, plot=False, centralized=False):
        """Returns the continuum-normalized line"""

        wavelength = self.wavelength
        normalized_flux = self.continuum_normalize()

        centroid = hlines[line]
        distance = hdistances[line]

        crop1 = bisect_left(wavelength, centroid - distance)
        crop2 = bisect_left(wavelength, centroid + distance)

        line_wavelength = wavelength[crop1:crop2]
        line_flux = normalized_flux[crop1:crop2]

        if centralized is True:
            line_wavelength = line_wavelength - centroid

        if plot:
            plt.plot(line_wavelength, line_flux)
            plt.show()

        return line_wavelength, line_flux

def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile function.
    x: wavelength or frequency array
    amplitude: amplitude of the profile
    center: center of the profile
    sigma: Gaussian sigma (related to Doppler broadening)
    gamma: Lorentzian gamma (related to pressure broadening)
    """
    return amplitude * voigt_profile(x - center, sigma, gamma)

voigt_model = Model(voigt)

hlines = {
    'alpha'   : 6562.790,
    'beta'    : 4861.350,
    'gamma'   : 4340.472,
    'delta'   : 4101.734,
    'epsilon' : 3970.075,
    'zeta'    : 3889.064
}

hdistances = {
    'alpha'   : 300,
    'beta'    : 200,
    'gamma'   : 120,
    'delta'   : 80,
    'epsilon' : 55,
    'zeta'    : 30
}

norm_unit = u.dimensionless_unscaled

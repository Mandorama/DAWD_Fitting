import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RectBivariateSpline
from SpectrumClasses import ModelSpectrum, Observation, hlines
from scipy.ndimage import label, find_objects
import speclite.filters as sf

os.environ["XDG_SESSION_TYPE"] = "xcb"

class ModelGrid():
    """Class to hold information about a grid of models"""

    def __init__(self,filepath):
        self.filepath = filepath
        self.directory = os.path.dirname(self.filepath)

        self._read_grid()

    def _read_grid(self):
        with open(self.filepath, 'r') as grid:
            lines = grid.readlines()

        self.name = lines[0].split(': ')[1].strip()
        self.code = lines[1].split(': ')[1].strip()
        if 'TLUSTY' in self.code:
            self.tlusty_model = True
        else:
            self.tlusty_model = False
        regl = lines[2].split(': ')[1].strip()
        if regl == 'T':
            self.regular_grid = True
        else:
            self.regular_grid = False
        self.teff_grid = self._read_param_grid(lines[3].split(': ')[1].strip(),'teff')
        self.logg_grid = self._read_param_grid(lines[4].split(': ')[1].strip(),'logg')
        self.n_teff = self.teff_grid.size
        self.n_logg = self.logg_grid.size

        self.model_list = np.array([model.strip() for model in lines[5:]])
        self.model_grid = self.model_list.reshape([self.n_teff,self.n_logg])

    def _read_param_grid(self,param_array,param):
        if param == 'teff':
            param_type = int
        else:
            param_type = float
        if self.regular_grid is True:
            min_param, max_param, n_param = np.array(param_array.split(),dtype=param_type)
            n_param = int(n_param)
            param_grid = np.linspace(min_param,max_param,n_param)
        else:
            param_grid = np.array(param_array.split(),dtype=param_type)

        return param_grid

def get_config():
    """
    Defines the configuraton for fitting the parameters of a given
    spectra. Reads optional arguments given in command line to change
    such configs.

    Optional Arguments:
    ------------------
    -s, --single: str
        Sets the filepath to a single spectra to be fitted in isolation.
    -d, --directory: str
        Sets the filepath to a directory wiht .spec files to be fitted.
        defalut: "../New_SDSS/"
    -g, --grid: str
        Sets the filepath to the grid file containing the informations
        for the model grid. The models must be on the same directory.
        default: "./441_Grid/grid.txt"
    -f, --flux_grid:
        A new grid of the line flux will be calculated for the models
        in the grid. It will be saved as "lines.npz".
    -l, --lines: str
        Gives a string containing the initials of the lines to be
        considered.
        default: "abgdez" [a(lpha)b(eta)g(amma)d(elta)e(psilon)z(eta)]
    -c, --color: str
        Sets the color to be used when color is needed to choose a
        fitted minima. Colors accepted: 'BP-RP, u-g, g-r, r-i, i-z'.
        default: "BP-RP"
    -p, --plot
        The chi squared surface of the fitting will be plotted.

    Returns:
    -------
    config: dict
        Contains the needed configurations for the program to run.
    """

    parser = argparse.ArgumentParser(description="Parameter Parser")
    parser.add_argument("-s",  "--single",type=str)
    parser.add_argument("-d",  "--directory",type=str)
    parser.add_argument("-g",  "--grid",type=str,default="./Koester_2010/grid.txt")
    parser.add_argument("-f",  "--flux_grid",action="store_true")
    parser.add_argument("-l",  "--lines",type=str,default="abgdez")
    parser.add_argument("-c",  "--color",type=str,default="BP-RP")
    parser.add_argument("-p",  "--plot",action="store_true")
    args = parser.parse_args()

    config = {'target_spectra'  : args.single,
              'spectra_dir'     : args.directory,
              'grid_filepath'   : args.grid,
              'new_flux_grid'   : args.flux_grid,
              'lines'           : get_lines(args.lines),
              'color'           : args.color,
              'plot'            : args.plot}

    return config

def get_lines(lines_config):
    """Reads lines input from command line and returns in a array"""

    line_map = {"a" : "alpha",
                "b" : "beta",
                "g" : "gamma",
                "d" : "delta",
                "e" : "epsilon",
                "z" : "zeta"}

    line_order = ['alpha','beta','gamma','delta','epsilon','zeta']

    lines = [line_map[char] for char in set(lines_config)]
    lines.sort(key=lambda x: line_order.index(x))

    return lines

def get_model_grids(lines):
    """
    Retrieves the grid containing the convolved normalized line flux
    for the model grid and the grid of splines for each accepted coolor
    If asked for in the command line, will call for the creation new
    grids.

    Parameters:
    ----------
    lines: list(str)
        Contains the lines that shall be used for the fitting process.

    Returns:
    -------
    flux_grid: np.ndarray([len(lines),n_teff,n_logg])
        Grid containing each line for each model present in the grid.
    color_grid: np.ndarray([5])
        Array containing each spline for the colors accepted by the
        code: BP-RP, u-g, g-r, r-i, i-z
    """

    if create_new_flux_grid is True : create_model_grids()

    flux_grid = np.ones([len(lines),n_teff,n_logg],dtype=object)

    try:
        line_grid = np.load(f'{grid_directory}/line_grid.npz', allow_pickle=True)
    except:
        print('Pre-calculated grids not found on the model grid directory')
        create_model_grids()
        line_grid = np.load(f'{grid_directory}/line_grid.npz', allow_pickle=True)
    wavelengths = line_grid['wavelengths']

    for index, line in enumerate(lines):
        line_index = get_line_index(line)
        line_fluxes = line_grid[line]
        line_wavelength = wavelengths[line_index][0]
        line_center = hlines[line]

        for i in range(n_teff):
            for j in range(n_logg):
                model_line = line_fluxes[i][j]
                line_spline = CubicSpline(line_wavelength - line_center,model_line)
                flux_grid[index,i,j] = line_spline

    color_grid = np.load(f'{grid_directory}/color_grid.npy',allow_pickle=True)

    return flux_grid, color_grid

def create_model_grids():
    """Call the creation of both flux and color grids for the models."""

    print('Calculating line fluxes grid')
    create_flux_grid()
    print('Calculating color splines grid')
    create_color_grid()

def create_flux_grid():
    """
    Create a grid containing the flux for all the normalized lines
    from alpha to zeta for the model grid.
    Saves it in a file named "line_grid.npz" in the grid directoy.
    """

    alpha_space, line_alpha     = calculate_line_flux('alpha')
    beta_space, line_beta       = calculate_line_flux('beta')
    gamma_space, line_gamma     = calculate_line_flux('gamma')
    delta_space, line_delta     = calculate_line_flux('delta')
    epsilon_space, line_epsilon = calculate_line_flux('epsilon')
    zeta_space, line_zeta       = calculate_line_flux('zeta')

    spectra_space = np.array([[alpha_space],[beta_space],
                              [gamma_space],[delta_space],
                              [epsilon_space],[zeta_space]],dtype=object)

    np.savez(f'{grid_directory}/line_grid', wavelengths = spectra_space,
             alpha   = line_alpha,
             beta    = line_beta,
             gamma   = line_gamma,
             delta   = line_delta,
             epsilon = line_epsilon,
             zeta    = line_zeta)

def calculate_line_flux(line):
    """
    Convolves the normalized flux for the given line for each model on
    the grid. Also retriving the wavelength space for each of them.

    Returns:
    -------
    line_wavelength: np.array
        Array containing the wavelengths values for the given line.
    line_grid: np.array(n_teff,nlogg)
        Matrix containing the convoluted flux of the line for the
        models.
    """

    line_grid = np.ones([n_teff,n_logg],dtype=object)

    for i in range(n_teff):
        for j in range(n_logg):
            model_spectra = get_grid_spectra(model_grid,i,j)
            convoluted_line = model_spectra.convolve_line(line,R)
            convoluted_line_flux = convoluted_line[1]
            line_grid[i,j] = convoluted_line_flux

    line_wavelength = convoluted_line[0]

    return line_wavelength, line_grid

def get_grid_spectra(model_grid, index_a, index_b):
    """
    Creates a ModelSpectrum object for the model in the specified
    indexes for the given model grid.

    Returns:
    -------
    model_spectra: ModelSpectrum
        ModelSpectrum object used for the analysis of the model.
    """

    model_name = model_grid[index_a, index_b]
    model_filepath = f'{grid_directory}/{model_name}.spec'
    if use_generic_model is True:
        model_data = pd.read_csv(model_filepath)
        model_data = model_data[(model_data.Wavelength>3000)&(model_data.Wavelength<11000)]
        model_data = model_data.reset_index()
    else:
        model_data = pd.read_csv(model_filepath,
                                 sep=r'\s{4,}',engine='python',names=['Wavelength','Flux'])

    model_spectra = ModelSpectrum(model_data,generic=use_generic_model)

    return model_spectra

def create_color_grid():
    """
    Create an array containing the splines for the calculation of color
    in the model grid. For each model grid, calculate BP-RP, u-g, g-r,
    r-i and i-z colors.
    Saves it in a file named "color_grid.npy" in the grid's directory.
    """

    colors = ['BP-RP','u-g','g-r','r-i','i-z']
    color_splines = np.ones([len(colors)],dtype=object)

    for index, color in enumerate(colors):
        filter_1, filter_2 = color.split('-')
        color_grid = calculate_color_grid(filter_1,filter_2)
        color_spline = RectBivariateSpline(teff_grid,logg_grid,color_grid)

        color_splines[index] = color_spline

    np.save(f'{grid_directory}/color_grid', color_splines)

def calculate_color_grid(filter_1, filter_2):
    """Calculates the color grid for two given filters."""
    color_grid = np.ones([n_teff,n_logg])

    for i in range(n_teff):
        for j in range(n_logg):
            model_spectra = get_grid_spectra(model_grid,i,j)
            magnitude_1 = calculate_spectra_magnitude(model_spectra,filter_1)
            magnitude_2 = calculate_spectra_magnitude(model_spectra,filter_2)
            color_index = magnitude_1 - magnitude_2
            color_grid[i,j] = color_index

    return color_grid

def calculate_spectra_magnitude(spectra, magnitude_filter):
    """
    Uses speclite.filters to calculate the magnitude of the model
    spectra. 7 magnitudes are available: BP and RP using gaiadr3
    filters and u, g, r, i, and z filteres of sdss2010.

    The magnitudes are first calculated in AB system, but turned into
    Vega system for the gaia filters or asin system for SDSS filters.
    """

    wavelength = spectra.wavelength
    flux = spectra.flux
    if magnitude_filter in ['BP','RP']:
        magnitude_band = sf.load_filter(f'gaiadr3-{magnitude_filter}')
        if magnitude_filter == 'BP':
            cte = 25.3385422158 - 25.3539555559
        else:
            cte = 24.7478955012 - 25.1039837393
    else:
        magnitude_band = sf.load_filter(f'sdss2010-{magnitude_filter}')
        if magnitude_filter == 'u':
            cte = 0.04
        elif magnitude_filter == 'z':
            cte = -0.02
        else:
            cte = 0
    pd_fl, pd_wv = magnitude_band.pad_spectrum(flux,wavelength,method='edge')
    magnitude = magnitude_band.get_ab_magnitude(pd_fl,pd_wv) + cte

    return magnitude

def get_line_index(line):
    """"Returns the index relative to the given line."""

    line_index = {'alpha'   : 0,
                  'beta'    : 1,
                  'gamma'   : 2,
                  'delta'   : 3,
                  'epsilon' : 4,
                  'zeta'    : 5}

    return line_index[line]

def fit_spectra(spectra,lines,flux_grid,color_data,plot=False,save=False,full=False):
    """
    Fit the observed spectra using a chi squared procedure and return
    the fitted parameters and respective errors. Balmer lines can be
    specified, from alpha to zeta. The calculation needs pre-calculated
    grids of flux and color data from the grid of models.

    Paramters:
    ---------
    spectra: SpectrumTools.Observation
        Observation object containing spectral data (wavelength, flux,
        ivar) and observed magnitudes.
    lines: list
        Lists the balmer lines used for fitting. Availabre from alpha to
        zeta. This list may not be completely used, as the code will
        discard lines with problematic normalization.
    flux_grid: np.ndarray([len(lines),n_teff,n_logg])
        Grid containing each line for each model present in the grid.
    color_data: (np.ndarray([5]),str)
        Tuple containing each spline for the colors accepted by the
        code: BP-RP, u-g, g-r, r-i, i-z; and the color being used to
        differentiate multiple minima found.

    Optional Arguments:
    ------------------
    plot:
        After the fitting procedure, a contour plot of the chi2 surface
        will show up. The fitted parameters will be represented by an
        'X' mark.
    save:
        Similar to plot, but instead saves the plot in the current
        direcotry for single fitting or inside a /output directory
        inside the spectra directory when done in bulk.
    full:
        Option for debugging purpose, the code will interpolate every
        possible point in the grid and also plot/save in full. This
        option may consume much more RAM then needed.

    Returns:
    -------
    Return varies depending on single or directory type of fitting.
    result: str
        If a directory of spectra is fitted, the code will write the
        results on a fitting.dat file inside model grid directory. So,
        fit_spectra returns a formatted string to be written on the
        file. The values used are described below.
    teff_fitted: int
        Effective temperature fitted for the spectra. Resolution of 1 K.
    e_teff: int
        Standard deviation of the fit procedure for the effective
        temperature. Resolution of 1 K.
    logg_fitted: float
        Surface gravity (log g) fitted for the spectra. Resolution of
        0.001 dex
    e_logg: float
        Standard deviation of the fit procedure for the surface
        gravity (log g). Resolution of 0.001 dex.
    color_fitted: float
        Interpolated color for the fitted parameters given color_data.
    """

    color_grid, color = color_data
    color_spline = color_grid[color_spline_index(color)]
    print(color)
    spectra_data, lines_data, abort = get_joint_flux(spectra,lines)
    if abort is True:
        print("More than half of the Lines couldn't be fitted, aborting")
        result = '{: >19}'.format(spectra.objid) + '  ERROR - Line Fitting\n'

        return result

    print('Interpolating Chi2')
    try:
        chi_grid = chi2_grid(spectra_data,lines_data,n_teff,n_logg,flux_grid)
    except:
        print("Chi2 surface couldn't be calculated, aborting")
        result = '{: >19}'.format(spectra.objid) + '  ERROR - Chi2 Surface\n'

        return result

    print('Finding Minima')
    regions = minima_regions(chi_grid)
    subgrid_interpolated, parameter_spaces, spline = interpolate_regions(regions,chi_grid,full)
    minima_values = minima_parameters(subgrid_interpolated,regions,parameter_spaces,color_spline)

    print('Finding Solution')
    teff_fitted, logg_fitted, color_fitted = find_solution(spectra,color,minima_values)

    if plot or save:
        teff_space, logg_space = parameter_spaces
        if full is True:
            downsampling_factor = 1
        else:
            downsampling_factor = 25
        plt.contour(teff_space[::downsampling_factor], logg_space[::downsampling_factor],
                    subgrid_interpolated.T[::downsampling_factor,::downsampling_factor],
                    levels=50,cmap='inferno')

        plt.plot([teff_fitted],[logg_fitted],marker='x')
        plt.xlabel("Teff")
        plt.ylabel("Logg")
        plt.title(r"Curvas de nível de $S$")
        if save:
            if config['spectra_dir'] is not None:
                plt.savefig(f'{config["spectra_dir"]}/output/{spectra.objid}',dpi=100)
            else:
                plt.savefig(f'{spectra.objid}',dpi=100)
        if plot:
            plt.show()
        else:
            plt.clf()

    print('Calculating error')
    e_teff, e_logg = calc_parameters_chi(subgrid_interpolated,spline,parameter_spaces,
                                         teff_fitted,logg_fitted)
    if config['target_spectra'] is not None:
        return teff_fitted, e_teff, logg_fitted, e_logg, color_fitted

    result = format_result(spectra,teff_fitted,e_teff,logg_fitted,e_logg,color_fitted)

    return result

def get_joint_flux(spectra, lines):
    """
    Get an array of the jointed fluxes of the lines on spectra. Checks
    if a normalization is problematic, discarding the line if needed.
    If at least half of the lines are problematic, signals the code to
    abort the fitting procedure. The non-discarded lines are also
    returned.
    """

    num_bad_lines = 0
    abort = False

    final_lines = []
    joint_flux = []
    joint_ivar = []
    line_space = np.ones(len(lines),dtype=object)
    for index, line in enumerate(lines):
        line_wavelength, line_flux, ivar, result = spectra.normalize_individual_line(line,
                                                                                     central=True)
        line_space[index] = line_wavelength
        if result is True:
            joint_flux.append(line_flux)
            joint_ivar.append(ivar)
            final_lines.append(line)
        else:
            num_bad_lines += 1
            if num_bad_lines >= 0.5 * len(lines):
                abort = True
                break

    joint_flux = np.concatenate(joint_flux)
    joint_ivar = np.concatenate(joint_ivar)
    spectra_data = (joint_flux, joint_ivar)
    lines_data = (final_lines, line_space)

    return spectra_data, lines_data, abort

def chi2_grid(spectra_data,lines_data,n_teff,n_logg,flux_grid):
    """
    Calculates the grid of chi square values between the spectra
    and each of the models in the grid.

    Returns:
    -------
    chi2_grid: np.array(n_teff,n_logg)
        Matrix containing the calculated chi2 for each model
    """

    spectra_flux, ivar = spectra_data
    lines, line_space = lines_data
    chi2_grid = np.ones([n_teff,n_logg])
    for i in range(n_teff):
        for j in range(n_logg):
            joint_model = get_joint_model(lines,flux_grid[:,i,j],line_space)
            chi2_grid[i,j] = chi2(spectra_flux, joint_model, ivar)

    return chi2_grid

def get_joint_model(lines,flux_grid,line_space):
    """
    Returns an array of the joint flux of the to be fitted lines.
    """

    joint_flux = []
    for index, line in enumerate(lines):
        line_index = get_line_index(line)
        line_wavelength = line_space[line_index]
        spline = flux_grid[line_index]
        line_flux = np.array(spline(line_wavelength))
        joint_flux.append(line_flux)

    joint_flux = np.concatenate(joint_flux)

    return joint_flux

def chi2(real_data, model_data, ivar):
    """
    Determines the chi squared test statistic between the line data
    from the oberved data and a model's data.
    """

    sum = 0
    for i in range(len(real_data)):
        sum += ivar[i]*((real_data[i]-model_data[i])**2)

    return sum

def minima_regions(grid):
    """
    Identifies regions of chi2 minima on the grid. The function does
    this by using a threshold value that multiples the minimum value
    of chi2. If only one minima region is founded, a smaller value is
    used. That's repeated between threshold values of 1.2 and 1.02 or
    until the grid is broken in two or more minima regions.

    When more regions are found or only one is identifiable with 1.02,
    the function returns a labeled grid that allows the different
    regions to be identified.
    """

    chi2_min = np.min(grid)
    threshold_scale = np.arange(1020,1201,1)[::-1]/1000
    for value in threshold_scale:
        threshold = chi2_min*value
        mask = grid <= threshold
        labeled_regions, n_regions = label(mask)
        if n_regions > 1:
            print('Multiple Regions of Minima Found')

            return labeled_regions

    threshold = chi2_min*threshold_scale[0]
    mask = grid <= threshold
    labeled_regions, n_regions = label(mask)

    return labeled_regions

def interpolate_regions(regions,grid,full=False):
    """
    Interpolates the parameter space containing all the minima regions.
    The resulting interpolated grid is rectangular with the lowest and
    highest values of effective temperature and surface gravity found
    on the regions.

    Returns:
    -------
    interpolated_regions: np.ndarray
        Interpolated grid of the regions of interest for the fitting
        procedure. The limits in teff and logg takes into account every
        minima region. Has resolutions of 1 K and 0.001 dex.
    parameter_spaces: tuple(np.ndarray,np.ndarray)
        Tuple containing teff_space and logg_space, arrays of the values
        of teff and logg considered in interpolation.
    grid_spline: scipy.interpolate._fitpack2.RectBivariateSpline
        Spline of chi2 used to create the interpolated grid. Returned so
        that it can be used for error calculations.
    """

    grid_spline = RectBivariateSpline(teff_grid,logg_grid,grid)
    if full is False:
        min_teff, max_teff, min_logg, max_logg = region_parameters(regions)
        teff_space = np.arange(min_teff,max_teff+1,1)
        logg_space = np.arange(min_logg*1000,max_logg*1000+1,1)/1000
    else:
        teff_space, logg_space = get_grid_spaces()
    interpolated_regions = grid_spline(teff_space, logg_space)
    parameter_spaces = (teff_space,logg_space)

    return interpolated_regions, parameter_spaces, grid_spline

def region_parameters(regions):
    """
    Returns a tuple with the minimum and maximum values of teff and logg
    found in the minima regions parameter spaces.
    """

    coords = find_objects(regions)
    region_parameters = np.ones([4])
    min_teff = teff_grid.min()
    max_teff = teff_grid.max()
    min_logg = logg_grid.min()
    max_logg = logg_grid.max()
    min_grid_teff = max_teff
    max_grid_teff = min_teff
    min_grid_logg = max_logg
    max_grid_logg = min_logg
    for index, region in enumerate(coords):
        teff_array = teff_grid[region[0]]
        logg_array = logg_grid[region[1]]
        teff_array, logg_array = check_parameters_arrays(teff_array,logg_array)
        min_teff_value = teff_array.min()
        if min_teff_value < min_grid_teff:
            min_grid_teff = min_teff_value
        max_teff_value = teff_array.max()
        if max_teff_value > max_grid_teff:
            max_grid_teff = max_teff_value
        min_logg_value = logg_array.min()
        if min_logg_value < min_grid_logg:
            min_grid_logg = min_logg_value
        max_logg_value = logg_array.max()
        if max_logg_value > max_grid_logg:
            max_grid_logg = max_logg_value

    region_parameters = min_grid_teff, max_grid_teff, min_grid_logg, max_grid_logg

    return region_parameters

def check_parameters_arrays(teff_array,logg_array):
    """
    Checks if the parameters arrays contains only one value. If that's
    the case, expand it to include other two values for such parameter.
    """

    if len(teff_array) < 2:
        min_teff = teff_grid[0]
        max_teff = teff_grid[-1]
        dteff = teff_grid[1] - min_teff
        teff = teff_array[0]
        if teff == min_teff:
            teff_array = np.array([teff_grid[0], teff_grid[1], teff_grid[2]])
        elif teff == max_teff:
            teff_array = np.array([teff_grid[-3], teff_grid[-2], teff_grid[-1]])
        else:
            teff_array = np.array([teff - dteff, teff, teff + dteff])
    if len(logg_array) < 2:
        min_logg = logg_grid[0]
        max_logg = logg_grid[-1]
        dlogg = logg_grid[1] - min_logg
        logg = logg_array[0]
        if logg == min_logg:
            logg_array = np.array([logg_grid[0], logg_grid[1], logg_grid[2]])
        elif logg == max_logg:
            logg_array = np.array([logg_grid[-3], logg_grid[-2], logg_grid[-1]])
        else:
            logg_array = np.array([logg - dlogg, logg, logg + dlogg])

    return teff_array, logg_array

def get_grid_spaces():
    """Returns the teff and logg values present in the model grid."""

    min_teff = np.min(teff_grid)
    max_teff = np.max(teff_grid)
    min_logg = np.min(logg_grid)
    max_logg = np.max(logg_grid)
    teff_space = np.arange(min_teff,max_teff+1,1)
    logg_space = np.arange(min_logg*1000,max_logg*1000+1,1)/1000

    return teff_space, logg_space

def minima_parameters(grid,labeled_regions,parameter_spaces,color_spline):
    """
    Finds the minimum value of chi2 in each minima region and returns
    its corresponding teff, logg and color values.

    Returns:
    -------
    fitted_parameters: np.ndarray([len(regions),3])
        Contains the fitted values of effective temperature, surface
        gravity and color for each of the minima regions.
    """

    teff_space, logg_space = parameter_spaces
    coords = find_objects(labeled_regions)
    n_regions = len(coords)
    fitted_parameters = np.ones([n_regions,3])
    for index, region in enumerate(coords):
        teff_array = teff_grid[region[0]]
        logg_array = logg_grid[region[1]]
        teff_fitted, logg_fitted = get_fitted_indexes(teff_array,logg_array,
                                                      teff_space,logg_space,
                                                      grid)

        color_fitted = color_spline(teff_fitted, logg_fitted)[0][0]
        fitted_parameters[index,:] = teff_fitted, logg_fitted, color_fitted

    return fitted_parameters

def get_fitted_indexes(teff_array, logg_array, teff_space, logg_space, interpolated_grid):
    """
    Returns the parameters indexes of teff and logg corresponding to the
    point of minimum value on a sub-grid of the interpolated grid. The
    constraints being set by teff_array and logg_array.
    """

    teff_array, logg_array = check_parameters_arrays(teff_array,logg_array)
    min_teff = teff_array.min()
    max_teff = teff_array.max()
    min_logg = logg_array.min()
    max_logg = logg_array.max()
    teff_indices = np.where((teff_space >= min_teff) & (teff_space <= max_teff))[0]
    logg_indices = np.where((logg_space >= min_logg) & (logg_space <= max_logg))[0]
    grid = interpolated_grid[np.ix_(teff_indices,logg_indices)]
    min_indexes = np.argmin(grid)
    index_teff, index_logg = np.unravel_index(min_indexes, grid.shape)
    teff_subspace = np.arange(min_teff,max_teff+1,1)
    logg_subspace = np.arange(min_logg*1000,max_logg*1000+1,1)/1000
    teff_fitted = teff_subspace[index_teff]
    logg_fitted = logg_subspace[index_logg]

    return teff_fitted, logg_fitted

def find_solution(spectra,color,parameter_collection):
    """
    Decides the better fitted solution by comparing the fitted colors
    with the actual observed color of the spectra.
    """

    try:
        colors = {'BP-RP' : spectra.bprp,
                  'u-g'   : spectra.ug,
                  'g-r'   : spectra.gr,
                  'r-i'   : spectra.ri,
                  'i-z'   : spectra.iz}
    except:
        colors = {'BP-RP' : spectra.bprp}
    spectra_color = colors[color]
    minima_index = np.argmin(abs(parameter_collection[:,2]-spectra_color))
    teff_fitted, logg_fitted, color_fitted = parameter_collection[minima_index,:]

    return teff_fitted, logg_fitted, color_fitted

def calc_parameters_chi(chi_interpolated,spline,parameter_spaces,teff_fitted,logg_fitted):
    """
    Uses the method on Zhang, 1986 to estimate the fitting error. Given
    the fitted values of teff and logg, a variation of the chi2 value
    found by fixing one parameter and varying the other, founding
    other minima value of chi2.

    This function specifically uses the mean between a variation above
    and bellow the fitted values.

    Returns:
    -------
    chi_teff: float
        Standard deviation of effective temperature from the fitting
        procedure.
    chi_logg: float
        Standard deviation of surface gravity from the fitting
        procedure.
    """

    teff_space, logg_space = parameter_spaces
    dteff = teff_fitted*0.05
    dlogg = logg_fitted*0.05
    min_teff = teff_fitted-dteff
    max_teff = teff_fitted+dteff
    min_logg = logg_fitted-dlogg
    max_logg = logg_fitted+dlogg

    S0 = np.min(chi_interpolated)
    S_fixed = np.ones(len(logg_space))
    for index, logg in enumerate(logg_space):
        S_fixed[index] = (spline.ev(max_teff,logg)+spline.ev(min_teff,logg))/2
    S = S_fixed.min()
    chi_teff = np.sqrt(((dteff)**2)/(S - S0))

    S_fixed = np.ones(len(teff_space))
    for index, teff in enumerate(teff_space):
        S_fixed[index] = (spline.ev(teff,max_logg)+spline.ev(teff,min_logg))/2
    S = S_fixed.min()
    chi_logg = np.sqrt(((dlogg)**2)/(S - S0))

    return chi_teff, chi_logg

def format_result(spectra,teff_fitted,e_teff,logg_fitted,e_logg,color_fitted):
    """Format a string to be written in fitting.dat file."""

    objid = spectra.objid
    teff = int(teff_fitted)
    e_teff = int(np.round(e_teff,0))
    logg = logg_fitted
    e_logg = np.round(e_logg,3)
    color = np.round(color_fitted,6)

    objid_str = '{: >19}'.format(objid)
    teff_str  = '  ' + '{: >6}'.format(teff) + '  ' + '{: >6}'.format(e_teff)
    logg_str  = '  ' + '{: <6.3f}'.format(logg) + '  ' + '{: >6.3f}'.format(e_logg)
    color_str = '  ' + '{: >9.6f}'.format(color)
    formatted_result = objid_str + teff_str + logg_str + color_str + "\n"

    return formatted_result

def color_spline_index(color):
    """Returns the corresponding in the spline for  agiven color."""

    colors = {'BP-RP' : 0,
              'u-g'   : 1,
              'g-r'   : 2,
              'r-i'   : 3,
              'i-z'   : 4}

    return colors[color]

if __name__ == "__main__":
    config = get_config()
    grid_path = config['grid_filepath']
    create_new_flux_grid = config['new_flux_grid']
    lines = config['lines']
    color = config['color']
    plot = config["plot"]

    R = 2000
    print('Reading Grid')
    grid = ModelGrid(grid_path)
    grid_name = grid.name
    grid_code = grid.code
    print(f'Using the grid: {grid_name} using {grid_code} models')
    grid_directory = grid.directory
    use_generic_model = not grid.tlusty_model
    teff_grid  = grid.teff_grid
    logg_grid  = grid.logg_grid
    model_grid = grid.model_grid
    n_teff = teff_grid.size
    n_logg = logg_grid.size

    if config['target_spectra'] is not None:
        print('Getting Spectra')
        spectra_filepath = config["target_spectra"]
        spectra = Observation(spectra_filepath)
        flux_grid, color_grid = get_model_grids(lines)
        color_data = (color_grid, color)

        print('Fitting')
        fit_result  = fit_spectra(spectra,lines,flux_grid,color_data,plot=plot)
        try:
            teff_fitted, e_teff, logg_fitted, e_logg, color_fitted = fit_result
        except:
            quit()

        pm = u"\u00b1"
        print(f"Teff  = {int(teff_fitted)} {pm} {int(np.round(e_teff,0))}")
        print(f"logg  = {logg_fitted} {pm} {np.round(e_logg,3)}")
        print(f"({color}) = {np.round(color_fitted,6)}")

        quit()

    spectra_directory = config['spectra_dir']
    spectra_list = glob.glob(f"{spectra_directory}/*.spec")
    spectra_list.sort()
    with open(f"{spectra_directory}/fitting.dat","w") as fit_file:
        fit_file.write(f'# ID                   Teff  e_Teff   logg   e_logg   ({color})\n')
        flux_grid, color_grid = get_model_grids(lines)
        color_data = (color_grid, color)

        for spectra_filepath in spectra_list:
            spectra = Observation(spectra_filepath)
            try:
                result = fit_spectra(spectra,lines,flux_grid,color_data,plot=plot,save=True)
                fit_file.write(result)

            except:
                fit_file.write('{: >19}'.format(spectra.objid) + '  ERROR - Fit Spectra\n')
        

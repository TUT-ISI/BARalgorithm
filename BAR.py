"""
Bayesian Aerosol Retrieval algorithm
version 1.0

For the most recent version of the codes, please see https://github.com/TUT-ISI/BARalgorithm

Reference:
Lipponen, A., Mielonen, T., Pitkänen, M. R. A., Levy, R. C., Sawyer, V. R., Romakkaniemi, S., Kolehmainen, V.,
and Arola, A.: Bayesian Aerosol Retrieval Algorithm for MODIS AOD retrieval over land, Atmos. Meas. Tech.,
https://doi.org/10.5194/amt-2017-359, accepted, 2018.

Contact information:
Antti Lipponen
Finnish Meteorological Institute
antti.lipponen@fmi.fi

----

Copyright 2018 Antti Lipponen / Finnish Meteorological Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os
import json
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset
import numpy as np
from scipy.sparse import diags, csr_matrix
import sys
from datetime import datetime
# BAR functions
from BARfunctions import loadGranule, prepareModels, loadSurfaceReflectancePrior, combinePolyLookupList, BARretrieve


def printUse():
    print('Use: "python BAR.py INPUT [OUTPUT] [-quantifyuncertainty] [-nospatialcorrelation] [-noapproximationerror]"\nExample: "python BAR.py MOD04_L2.A2015101.1155.006.2015104000600.hdf"')
    print('\nCommand line options:\n-quantifyuncertainty: Quantify the uncertainties related to retrieval and save the uncertainty estimates into the outputfile')
    print('-nospatialcorrelation: Do not use spatial correlation model')
    print('-noapproximationerror: Do not use approximation error model\n')


# Read granule to be retrieved
if len(sys.argv) < 2:
    print('\nMODIS M(O|Y)D04_L2 granule filename needed.')
    printUse()
    sys.exit(0)

dataFile = sys.argv[1]
inputFilePath, inputFileName = os.path.split(dataFile)
print('\nBayesian Aerosol Retrieval (BAR) algorithm version 1.0.\n\n  ---------------------\n  Reference:\n  Lipponen, A., Mielonen, T., Pitkänen, M. R. A., Levy, R. C., Sawyer, V. R., Romakkaniemi, S., Kolehmainen, V., and Arola, A.:\n  Bayesian Aerosol Retrieval Algorithm for MODIS AOD retrieval over land, Atmos. Meas. Tech., https://doi.org/10.5194/amt-2017-359, accepted, 2018.\n  ---------------------\n  Contact information:\n  Antti Lipponen, Finnish Meteorological Institute\n  antti.lipponen@fmi.fi\n  ---------------------\n')
print('  Retrieving MODIS granule file: {}'.format(inputFileName))

if len(sys.argv) >= 3 and sys.argv[2][0] != '-':
    outputFilePath, outputFileName = os.path.split(sys.argv[2])
else:
    outputFilePath, outputFileName = inputFilePath, 'BAR_' + '.'.join(os.path.split(dataFile)[1].split('.')[:-1]) + '.nc'

if not os.path.isfile(os.path.join(inputFilePath, inputFileName)):
    print('\nMODIS granule file "{}" not found.\n'.format(os.path.join(inputFilePath, inputFileName)))
    sys.exit(1)

# Define areas, prior models & settings
areas = {
    # East North America
    'ENA': {'min_lat': 25.0, 'max_lat': 90.0, 'min_lon': -100.0, 'max_lon': -30.0},
    # West North America
    'WNA': {'min_lat': 25.0, 'max_lat': 90.0, 'min_lon': -180.0, 'max_lon': -100.0},
    # Central and South America
    'CSA': {'min_lat': -90.0, 'max_lat': 25.0, 'min_lon': -180.0, 'max_lon': -30.0},
    # Europe
    'EUR': {'min_lat': 35.0, 'max_lat': 90.0, 'min_lon': -30.0, 'max_lon': 60.0},
    # North Africa and Middle East
    'NAME': {'min_lat': 0.0, 'max_lat': 35.0, 'min_lon': -30.0, 'max_lon': 60.0},
    # South Africa
    'SA': {'min_lat': -90.0, 'max_lat': 0.0, 'min_lon': -30.0, 'max_lon': 60.0},
    # Northeast Asia
    'NEA': {'min_lat': 35.0, 'max_lat': 90.0, 'min_lon': 60.0, 'max_lon': 180.0},
    # Southeast Asia
    'SEA': {'min_lat': 0.0, 'max_lat': 35.0, 'min_lon': 60.0, 'max_lon': 180.0},
    # Oceania
    'OCE': {'min_lat': -90.0, 'max_lat': 0.0, 'min_lon': 60.0, 'max_lon': 180.0}
}

settings = {
    'max_pixels_per_part': 3000,  # max number of pixels to be retrieved at once
    'BFGSftol': 1.0e-6,  # iteration parameter
    'BFGSgtol': 1.0e-6,  # iteration parameter
    'BFGSmaxiter': 5000,  # iteration parameter
    'N_processes': 4,  # number of processes for retrieval (if computer has 4 CPU cores, 4 should result in optimal performance)
    'surfacePriorYear': 2014,  # year of surface reflectance prior model
    'useApproximationErrorModel': True,  # should we use the (precomputed) approximation error model
    'useSpatialCorrelations': True,  # should we use the spatial correlation models for aerosol properties
    'quantifyUncertainty': False,  # should we quantify the uncertainties, BAR can provide pixel-level uncertainty estimates but by default these are not saved to save time and disc space
}

AODprior = {
    'range': 50.0,  # correlation length (km)
    'p': 1.5,  # p parameter (defines smoothness of the field)
    'sill': 0.10,  # variance for the spatially correlated component
    'nugget': 0.0025,  # local variance
}

FMFprior = {
    'range': 50.0,  # correlation length (km)
    'p': 1.5,  # p parameter (defines smoothness of the field)
    'sill': 0.25,  # variance for the spatially correlated component
    'nugget': 0.0100,  # local variance
}

surfacePriorModifier = {
    'meancoef': 1.0,  # surface prior model expected values are multiplied by this number, > 1.0 brighten surface reflectance assumptions, < 1.0 darken surface reflectance assumptions
    'stdcoef': 1.0,  # surface prior model standard deviations are multiplied by this number, > 1.0 adds uncertainty to surface prior, < 1.0 reduces uncertainty
}

# parse command line options
for option in sys.argv:
    if option[0] != '-':
        continue

    if option.lower() == '-quantifyuncertainty':
        settings['quantifyUncertainty'] = True
        print('  Uncertainty quantification enabled')
    elif option.lower() == '-nospatialcorrelation':
        settings['useSpatialCorrelations'] = False
        print('  Spatial correlation models disabled')
    elif option.lower() == '-noapproximationerror':
        settings['useApproximationErrorModel'] = False
        print('  Approximation error models disabled')
    elif option.lower()[:10] == '-aodprior=':
        print('  Using custom AOD prior parameters')
        try:
            l = json.loads(option[10:])
            for k, v in l.items():
                print('    {}: {}'.format(k, v))
                if k not in AODprior:
                    print('\n{} is not AOD prior parameter\n'.format(k))
                    sys.exit(1)
                AODprior[k] = v
        except:
            print('\nAOD prior parameter JSON is not valid.\n')
            sys.exit(1)
    elif option.lower()[:10] == '-fmfprior=':
        print('  Using custom FMF prior parameters')
        try:
            l = json.loads(option[10:])
            for k, v in l.items():
                print('    {}: {}'.format(k, v))
                if k not in FMFprior:
                    print('\n{} is not FMF prior parameter\n'.format(k))
                    sys.exit(1)
                FMFprior[k] = v
        except:
            print('\nFMF prior parameter JSON is not valid.\n')
            sys.exit(1)
    elif option.lower()[:14] == '-surfaceprior=':
        print('  Modifying surface reflectance prior')
        try:
            l = json.loads(option[14:])
            for k, v in l.items():
                print('    {}: {}'.format(k, v))
                if k not in surfacePriorModifier:
                    print('\n{} is not surface reflectance prior parameter\n'.format(k))
                    sys.exit(1)
                surfacePriorModifier[k] = v
        except:
            print('\nSurface reflectance prior modifier JSON is not valid.\n')
            sys.exit(1)
    else:
        print('\nUnknown command line option "{}"\n'.format(option))
        printUse()
        sys.exit(1)

# load approximation error model
try:
    with open(os.path.join('models', 'ApproximationError', 'approximationerror.json'), 'rt') as f:
        approximationErrorModel = json.load(f)
except:
    print('\nApproximation error file not found. This data should be included with the codes and data. Please check everything is correct.\n')
    sys.exit(1)

# load FMF prior model expected value (the file is not provided with the code, please see the readme for information how to download the file)
try:
    FMFpriorFile = os.path.join('models', 'FMF', 'gt_ff_0550nm.nc')
    f = Dataset(FMFpriorFile, 'r', format='NETCDF4')
    fmfMonth, fmfLat, fmfLon, fmfValues = np.arange(1, 13), f['lat'][:], f['lon'][:], f['aer_data'][:]
    f.close()
except:
    print('\nFMF prior model data file ({}) not found. This is an external file that needs to be downloaded before the code can be run.\nYou can download the file from: ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_ff_0550nm.nc\nPlease see README.txt for more information.\n'.format(FMFpriorFile))
    sys.exit(1)

# load AOD prior model expected value (the file is not provided with the code, please see the readme for information how to download the file)
try:
    AODpriorFile = os.path.join('models', 'AOD', 'gt_t_00550nm.nc')
    f = Dataset(AODpriorFile, 'r', format='NETCDF4')
    aodMonth, aodLat, aodLon, aodValues = np.arange(1, 13), f['lat'][:], f['lon'][:], f['aod'][:]
    f.close()
except:
    print('\nAOD prior model data file ({}) not found. This is an external file that needs to be downloaded before the code can be run.\nYou can download the file from: ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_t_00550nm.nc\nPlease see README.txt for more information.\n'.format(AODpriorFile))
    sys.exit(1)

# construct an interpolator for the expected values
AODFMFMeanIntpolator = {
    'AOD': RegularGridInterpolator((aodMonth, aodLat[::-1], aodLon), aodValues[:, ::-1, :]),
    'FMF': RegularGridInterpolator((fmfMonth, fmfLat[::-1], fmfLon), fmfValues[:, ::-1, :])
}

# load MODIS granule file
try:
    modisData = loadGranule(dataFile)
except:
    print('\nError loading datafile "{}". The file should be a valid MODIS MOD04_L2 or MYD04_L2 product file.\n'.format(dataFile))
    sys.exit(1)

granuleMonth, granuleYear = modisData['scantime'].month, modisData['scantime'].year
print('  Granule date and time: {}'.format(modisData['scantime'].strftime('%d %b %Y %H:%MZ (doy: %j)')))

try:
    granuleCenterPoint = [np.mean(modisData['lon']), np.mean(modisData['lat'])]
    print('  Granule center point: LAT: {:.2f}  LON: {:.2f}'.format(granuleCenterPoint[1], granuleCenterPoint[0]))
    area = list(filter(lambda x: x[1]['min_lat'] <= granuleCenterPoint[1] and
                x[1]['max_lat'] >= granuleCenterPoint[1] and
                x[1]['min_lon'] <= granuleCenterPoint[0] and
                x[1]['max_lon'] >= granuleCenterPoint[0], areas.items()))[0][0]
    print('  Area to be used for retrieval: {}'.format(area))
except:
    print('\nERROR in finding the area.\n')
    sys.exit(1)


# load surface reflectance prior model
try:
    surfacePrior = loadSurfaceReflectancePrior(granuleMonth, granuleYear, surfacePriorModifier, settings)
except:
    print('\nError loading surface prior.\n')
    sys.exit(1)

# prepare radiative transfer models
polynomialLookups, granuleMASK_full, allpartsData, gFull, aerosolmap = prepareModels(modisData, settings)
if len(allpartsData) > 1:
    print('  Retrieval will be run in {} parts'.format(len(allpartsData)))

savedata = {
    'BAR_AOD': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD466': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD644': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_2.5': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_5.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_16.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_84.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_95.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_AOD_percentile_97.5': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_2.5': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_5.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_16.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_84.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_95.0': -9.999 * np.ones(gFull.shape[0]),
    'BAR_FMF_percentile_97.5': -9.999 * np.ones(gFull.shape[0]),
    'BAR_rhos466': -9.999 * np.ones(gFull.shape[0]),
    'BAR_rhos553': -9.999 * np.ones(gFull.shape[0]),
    'BAR_rhos644': -9.999 * np.ones(gFull.shape[0]),
    'BAR_rhos211': -9.999 * np.ones(gFull.shape[0]),
    'lon': gFull[:, 0],
    'lat': gFull[:, 1],
    'DarkTarget_AOD': modisData['Corrected_Optical_Depth_Land553'],
    'DarkTarget_FMF': modisData['Optical_Depth_Ratio_Small_Land'],
}

if not settings['quantifyUncertainty']:
    savedata = {k: savedata[k] for k in savedata if k.find('percentile') < 0}

for iPart, partData in enumerate(allpartsData):
    if len(allpartsData) == 1:
        print('  Running retrieval...')
    else:
        print('  Running retrieval... (part {:d}/{:d})'.format(iPart + 1, len(allpartsData)))

    # preparing masks & coordinates
    MASKl, MASKf = partData['MASKl'], partData['MASKf']
    g = gFull[MASKf, :]

    # preparing surface reflectance prior for this granule
    indices = surfacePrior['kNNmean'].kneighbors(g, return_distance=False)
    indicesstd = surfacePrior['kNNstd'].kneighbors(g, return_distance=False)

    granuleSurfacePrior = {
        'mean466': surfacePrior['surfprior466'][indices].mean(axis=1),
        'mean550': surfacePrior['surfprior550'][indices].mean(axis=1),
        'mean644': surfacePrior['surfprior644'][indices].mean(axis=1),
        'mean211': surfacePrior['surfprior211'][indices].mean(axis=1),
        'std466': np.sqrt(np.clip(surfacePrior['surfprior466'][indices].std(axis=1), a_min=1.0e-6, a_max=1.0)**2 + np.mean(surfacePrior['surfprior466std'][indicesstd]**2, axis=1)),
        'std550': np.sqrt(np.clip(surfacePrior['surfprior550'][indices].std(axis=1), a_min=1.0e-6, a_max=1.0)**2 + np.mean(surfacePrior['surfprior550std'][indicesstd]**2, axis=1)),
        'std644': np.sqrt(np.clip(surfacePrior['surfprior644'][indices].std(axis=1), a_min=1.0e-6, a_max=1.0)**2 + np.mean(surfacePrior['surfprior644std'][indicesstd]**2, axis=1)),
        'std211': np.sqrt(np.clip(surfacePrior['surfprior211'][indices].std(axis=1), a_min=1.0e-6, a_max=1.0)**2 + np.mean(surfacePrior['surfprior211std'][indicesstd]**2, axis=1)),
    }
    Nvars = g.shape[0]

    # preparing observed data
    REFW466 = modisData['rho466'][MASKl]
    STDREFW466 = modisData['stdrho466'][MASKl]
    REFW553 = modisData['rho550'][MASKl]
    STDREFW553 = modisData['stdrho550'][MASKl]
    REFW644 = modisData['rho644'][MASKl]
    STDREFW644 = modisData['stdrho644'][MASKl]
    REFW2113 = modisData['rho211'][MASKl]
    STDREFW2113 = modisData['stdrho211'][MASKl]
    MeasData = np.concatenate((REFW466, REFW553, REFW644, REFW2113))
    MeasDataStd = np.concatenate((STDREFW466, STDREFW553, STDREFW644, STDREFW2113))

    # bounds for AOD & FMF (& surface reflectance)
    min_AOD, max_AOD = np.log(0.0 + 1.0), np.log(5.0 + 1.0)
    min_FMF, max_FMF = 0.0, 1.0
    bounds = np.array([[min_AOD, max_AOD], [min_FMF, max_FMF], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # load approximation error model
    if settings['useApproximationErrorModel']:
        print('    Using approximation error model {}/month {}'.format(area, granuleMonth))
        MeasNoiseVar = (1.0 / (1.0 + MeasData) * MeasDataStd)**2  # LOGSCALE RHOS
        Nwavelengths = 4
        icov_ii, icov_jj, icov_vv = np.zeros(Nwavelengths**2 * Nvars, dtype=int), np.zeros(Nwavelengths**2 * Nvars, dtype=int), np.zeros(Nwavelengths**2 * Nvars)
        for ii in range(Nvars):
            local_cov = np.array(np.array(approximationErrorModel[area][str(granuleMonth)]['cov']).reshape((Nwavelengths, Nwavelengths)) + diags([MeasNoiseVar[ii], MeasNoiseVar[ii + 1 * Nvars], MeasNoiseVar[ii + 2 * Nvars], MeasNoiseVar[ii + 3 * Nvars]], 0))
            local_icov = np.linalg.inv(local_cov)
            icov_ii[ii * Nwavelengths**2:(ii + 1) * Nwavelengths**2] = np.concatenate([[ii + 0 * Nvars] * Nwavelengths, [ii + 1 * Nvars] * Nwavelengths, [ii + 2 * Nvars] * Nwavelengths, [ii + 3 * Nvars] * Nwavelengths])
            icov_jj[ii * Nwavelengths**2:(ii + 1) * Nwavelengths**2] = np.concatenate([[ii + 0 * Nvars, ii + 1 * Nvars, ii + 2 * Nvars, ii + 3 * Nvars] * Nwavelengths])
            icov_vv[ii * Nwavelengths**2:(ii + 1) * Nwavelengths**2] = local_icov.ravel()

        MeasNoiseiCov = csr_matrix((icov_vv, (icov_ii, icov_jj)), shape=(Nwavelengths * Nvars, Nwavelengths * Nvars))
        MeasNoiseE = np.tile(np.array(approximationErrorModel[area][str(granuleMonth)]['E'])[:, np.newaxis], (1, Nvars)).ravel()
    else:
        MeasNoiseCov = diags((1.0 / (1.0 + MeasData) * MeasDataStd)**2, 0)  # LOGSCALE RHOS
        MeasNoiseiCov = diags(1.0 / MeasNoiseCov.diagonal(), 0)
        MeasNoiseE = 0.0
    # switch to log-scale measurement data (TOA reflectances)
    MeasData = np.log(MeasData + 1.0)

    # Initial values for retrieval
    AOD0 = AODFMFMeanIntpolator['AOD'](np.array([[granuleMonth, gg[1], gg[0]] for gg in np.clip(g, a_min=[-179.5, -89.5], a_max=[179.5, 89.5])]))
    FMF0 = AODFMFMeanIntpolator['FMF'](np.array([[granuleMonth, gg[1], gg[0]] for gg in np.clip(g, a_min=[-179.5, -89.5], a_max=[179.5, 89.5])]))

    # prepare the lookup-tables to final form
    polynomiallookupMatrices_fine = combinePolyLookupList(polynomialLookups[MASKl])
    polynomiallookupMatrices_coarse = combinePolyLookupList(polynomialLookups[MASKl], 4)
    BARresults = BARretrieve(g, AOD0, FMF0, AODprior, FMFprior, granuleSurfacePrior,
                             polynomiallookupMatrices_fine,
                             polynomiallookupMatrices_coarse,
                             MeasData, MeasNoiseiCov, MeasNoiseE, MASKf, bounds, settings)

    for k, v in BARresults.items():
        savedata[k][MASKf] = v

outputfile = os.path.join(outputFilePath, outputFileName)
print('  Saving results to {}'.format(outputfile))
nc = Dataset(outputfile, 'w')
nc.AODprior = str(AODprior).replace('\'', '')
nc.FMFprior = str(FMFprior).replace('\'', '')
nc.surfacePriorModifier = str(surfacePriorModifier).replace('\'', '')
nc.retrievalsettings = str(settings).replace('\'', '')
nc.inputfilename = inputFileName
nc.granuleTime = modisData['scantime'].strftime('%d-%m-%Y %H:%MZ')
nc.retrievalTime = datetime.utcnow().strftime('%d-%m-%Y %H:%MZ')
nX, nY = modisData['nX'], modisData['nY']
ncX = nc.createDimension('X', nX)
ncY = nc.createDimension('Y', nY)
var = nc.createVariable('latitude', 'f4', ('X', 'Y'))
var[:] = savedata['lat'].reshape((nX, nY))
var = nc.createVariable('longitude', 'f4', ('X', 'Y'))
var[:] = savedata['lon'].reshape((nX, nY))
for key in savedata.keys():
    if not (key[:3] == 'BAR' or key[:10] == 'DarkTarget'):
        continue
    var = nc.createVariable(key, 'f4', ('X', 'Y'))
    varToBeSaved = savedata[key]
    if key.find('AOD') >= 0:
        var.vmin = -0.1
        var.vmax = 5.0
    else:
        var.vmin = 0.0
        var.vmax = 1.0
    varToBeSaved[np.logical_or(varToBeSaved < var.vmin, varToBeSaved > var.vmax)] = np.nan
    var[:] = varToBeSaved
nc.close()
print('***** ALL DONE FOR THIS GRANULE *****\n\nThanks for using Bayesian Aerosol Retrieval!\n')

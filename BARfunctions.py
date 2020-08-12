"""
Bayesian Aerosol Retrieval algorithm function file
version 1.01 (12 August 2020)

For the most recent version of the codes, please see https://github.com/TUT-ISI/BARalgorithm

Reference:
Lipponen, A., Mielonen, T., Pitkänen, M. R. A., Levy, R. C., Sawyer, V. R., Romakkaniemi, S., Kolehmainen, V., and Arola, A.:
Bayesian Aerosol Retrieval Algorithm for MODIS AOD retrieval over land, Atmos. Meas. Tech., 11, 1529–1547, 2018.
https://doi.org/10.5194/amt-11-1529-2018.

Contact information:
Antti Lipponen
Finnish Meteorological Institute
antti.lipponen@fmi.fi

----

Copyright 2018-2020 Antti Lipponen / Finnish Meteorological Institute

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

import numpy as np
from pyhdf.SD import SD
from netCDF4 import Dataset
from datetime import datetime, timedelta
from multiprocessing import Pool  # use multiprocessing to make things faster
import copy
from scipy.interpolate import interp1d, RegularGridInterpolator
from imageio import imread
from sklearn.neighbors import KNeighborsRegressor
import os
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.linalg import block_diag
from scipy.stats import norm


def loadGranule(filename):
    dataFields = [
        'Latitude',
        'Longitude',
        'Mean_Reflectance_Land',
        'STD_Reflectance_Land',
        'Surface_Reflectance_Land',
        'Solar_Azimuth',
        'Sensor_Azimuth',
        'Solar_Zenith',
        'Sensor_Zenith',
        'Topographic_Altitude_Land',
        'Scan_Start_Time',
        'Optical_Depth_Ratio_Small_Land',
        'Corrected_Optical_Depth_Land',
        'Land_Ocean_Quality_Flag',
        'Aerosol_Type_Land',
    ]

    hdfData = {}
    hdfFile = SD(filename)
    for dataField in dataFields:
        q = hdfFile.select(dataField)
        qAttributes = q.attributes()
        qData = q.get()
        hdfData[dataField] = qData * qAttributes['scale_factor']
    hdfFile.end()

    nX, nY = hdfData['Latitude'].shape[0], hdfData['Latitude'].shape[1]
    Latitude, Longitude = hdfData['Latitude'].ravel(), hdfData['Longitude'].ravel()
    Mean_Reflectance_Land = hdfData['Mean_Reflectance_Land'].reshape((10, -1))
    STD_Reflectance_Land = hdfData['STD_Reflectance_Land'].reshape((10, -1))

    Solar_Azimuth = hdfData['Solar_Azimuth'].ravel()
    Sensor_Azimuth = hdfData['Sensor_Azimuth'].ravel()
    Solar_Zenith = hdfData['Solar_Zenith'].ravel()
    Sensor_Zenith = hdfData['Sensor_Zenith'].ravel()
    Topographic_Altitude_Land = hdfData['Topographic_Altitude_Land'].ravel()

    Scan_Start_Time = hdfData['Scan_Start_Time'][0, 0]

    Optical_Depth_Ratio_Small_Land = hdfData['Optical_Depth_Ratio_Small_Land'].ravel()
    Corrected_Optical_Depth_Land466 = hdfData['Corrected_Optical_Depth_Land'][0, :].ravel()
    Corrected_Optical_Depth_Land553 = hdfData['Corrected_Optical_Depth_Land'][1, :].ravel()
    Corrected_Optical_Depth_Land644 = hdfData['Corrected_Optical_Depth_Land'][2, :].ravel()
    Land_Ocean_Quality_Flag = hdfData['Land_Ocean_Quality_Flag'].astype(int).ravel()

    Aerosol_Type_Land = hdfData['Aerosol_Type_Land'].astype(int).ravel()

    MASK = np.logical_and(np.sum(Mean_Reflectance_Land > -0.1, axis=0) == 10, Corrected_Optical_Depth_Land553 > -0.1)

    theta0, phi0 = Solar_Zenith[MASK], Solar_Azimuth[MASK]
    theta, phi = Sensor_Zenith[MASK], Sensor_Azimuth[MASK]
    mhght = Topographic_Altitude_Land[MASK]

    rho466, rho553, rho644, rho211 = Mean_Reflectance_Land[0, MASK], Mean_Reflectance_Land[1, MASK], Mean_Reflectance_Land[2, MASK], Mean_Reflectance_Land[6, MASK]
    stdrho466, stdrho553, stdrho644, stdrho211 = STD_Reflectance_Land[0, MASK], STD_Reflectance_Land[1, MASK], STD_Reflectance_Land[2, MASK], STD_Reflectance_Land[6, MASK]

    Npixels = MASK.sum()
    scantime = datetime(1993, 1, 1) + timedelta(seconds=Scan_Start_Time)

    return {
        'Corrected_Optical_Depth_Land553': Corrected_Optical_Depth_Land553,
        'Corrected_Optical_Depth_Land466': Corrected_Optical_Depth_Land466,
        'Corrected_Optical_Depth_Land644': Corrected_Optical_Depth_Land644,
        'Land_Ocean_Quality_Flag': Land_Ocean_Quality_Flag,
        'Optical_Depth_Ratio_Small_Land': Optical_Depth_Ratio_Small_Land,
        'Aerosol_Type_Land': Aerosol_Type_Land[MASK] - 1,
        'MASK': MASK,
        'theta0': theta0,
        'phi0': phi0,
        'theta': theta,
        'phi': phi,
        'mhght': mhght,
        'lat': Latitude,
        'lon': Longitude,
        'rho466': rho466,
        'stdrho466': stdrho466,
        'rho550': rho553,
        'stdrho550': stdrho553,
        'rho644': rho644,
        'stdrho644': stdrho644,
        'rho211': rho211,
        'stdrho211': stdrho211,
        'Npixels': Npixels,
        'scantime': scantime,
        'nX': nX,
        'nY': nY
    }


def loadSurfaceReflectancePrior(granuleMonth, granuleYear, surfacePriorModifier, settings):
    print('  Loading surface reflectance prior model {}/{}'.format(granuleMonth, settings['surfacePriorYear']))
    surflats, surflons = np.meshgrid((np.arange(-90.0, 90.0, 0.05))[::-1] + 0.05 / 2, (np.arange(-180.0, 180.0, 0.05)) + 0.05 / 2)
    surflats, surflons = surflats.T.ravel(), surflons.T.ravel()

    surfacePriorYear = settings['surfacePriorYear']

    surfprior466 = 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band3_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel() + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band3_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()
    surfprior466std = np.sqrt(0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band3_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2 + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band3_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2)

    surfprior550 = 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band4_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel() + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band4_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()
    surfprior550std = np.sqrt(0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band4_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2 + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band4_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2)

    surfprior644 = 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band1_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel() + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band1_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()
    surfprior644std = np.sqrt(0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band1_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2 + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band1_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2)

    surfprior211 = 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band7_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel() + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band7_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()
    surfprior211std = np.sqrt(0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_BSA_Band7_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2 + 0.5 * (imread(os.path.join('models', 'SurfaceReflectance', str(surfacePriorYear), 'Albedo_WSA_Band7_std_{}_{:02d}.png'.format(str(surfacePriorYear), granuleMonth))) / 65535.0).ravel()**2)

    surfmask = (surfprior466 > 0.0)
    surfstdmask = (surfprior466std > 0.0)

    surfacePrior = {
        'surfprior466': surfprior466[surfmask] * surfacePriorModifier['meancoef'],
        'surfprior550': surfprior550[surfmask] * surfacePriorModifier['meancoef'],
        'surfprior644': surfprior644[surfmask] * surfacePriorModifier['meancoef'],
        'surfprior211': surfprior211[surfmask] * surfacePriorModifier['meancoef'],
        'lat': surflats[surfmask],
        'lon': surflons[surfmask],
        'surfprior466std': surfprior466std[surfstdmask] * surfacePriorModifier['stdcoef'],
        'surfprior550std': surfprior550std[surfstdmask] * surfacePriorModifier['stdcoef'],
        'surfprior644std': surfprior644std[surfstdmask] * surfacePriorModifier['stdcoef'],
        'surfprior211std': surfprior211std[surfstdmask] * surfacePriorModifier['stdcoef'],
        'latstd': surflats[surfstdmask],
        'lonstd': surflons[surfstdmask],
        'kNNmean': KNeighborsRegressor(n_neighbors=3, leaf_size=50000, n_jobs=-1).fit(np.hstack((surflons[surfmask][:, np.newaxis], surflats[surfmask][:, np.newaxis])), np.zeros_like(surfprior466[surfmask])[:, np.newaxis]),
        'kNNstd': KNeighborsRegressor(n_neighbors=3, leaf_size=50000, n_jobs=-1).fit(np.hstack((surflons[surfstdmask][:, np.newaxis], surflats[surfstdmask][:, np.newaxis])), np.zeros_like(surfprior466[surfstdmask])[:, np.newaxis]),
    }
    return surfacePrior


def prepareFcn(data):
    geometry, LUTdata = data['geometry'], data['LUTdata']
    lookups = interpolateToGeometry(geometry, LUTdata)
    return lookupToPolynomial(lookups)


def interpolateToGeometry(geometry, lutdata):

    theta, theta0, phi, phi0, mhght = geometry['theta'], geometry['theta0'], geometry['phi'], geometry['phi0'], geometry['mhght']

    lutInterpolatedData = copy.deepcopy(lutdata)
    del lutInterpolatedData['INT_NL0']
    del lutInterpolatedData['Fd_NL0']
    del lutInterpolatedData['SBAR_NL0']
    del lutInterpolatedData['T_NL0']
    del lutInterpolatedData['OPTH_NL0']
    lutInterpolatedData['OPTH_NL'] = lutdata['OPTH_NL0'].copy()

    mdphi = np.abs(phi0 - phi - 180.0)
    if mdphi > 360.0:
        mdphi = np.mod(mdphi, 360.0)
    if mdphi > 180.0:
        mdphi = 360.0 - mdphi

    NLTAU = len(lutdata['axisTau'])
    NLWAV = len(lutdata['axisWav'])
    NLAEROSOLMODEL = len(lutdata['axisAerosolModel'])

    coordsINT_NL0 = np.array(list(map(lambda x: x.ravel(), np.meshgrid(mdphi, theta, theta0, lutdata['axisTau'], lutdata['axisWav'], lutdata['axisAerosolModel'])))).T
    INT_NL_vector = RegularGridInterpolator((lutInterpolatedData['axisPhi'], lutInterpolatedData['axisTheta'], lutInterpolatedData['axisTheta0'], lutInterpolatedData['axisTau'], lutInterpolatedData['axisWav'], lutInterpolatedData['axisAerosolModel']), lutdata['INT_NL0'])(coordsINT_NL0)
    lutInterpolatedData['INT_NL'] = INT_NL_vector.reshape((NLTAU, NLWAV, NLAEROSOLMODEL))

    coordsSBAR_NL0 = np.array(list(map(lambda x: x.ravel(), np.meshgrid(theta0, lutdata['axisTau'], lutdata['axisWav'], lutdata['axisAerosolModel'])))).T
    SBAR_NL_vector = RegularGridInterpolator((lutInterpolatedData['axisTheta0'], lutInterpolatedData['axisTau'], lutInterpolatedData['axisWav'], lutInterpolatedData['axisAerosolModel']), lutdata['SBAR_NL0'])(coordsSBAR_NL0)
    lutInterpolatedData['SBAR_NL'] = SBAR_NL_vector.reshape((NLTAU, NLWAV, NLAEROSOLMODEL))

    coordsFd_NL0 = np.array(list(map(lambda x: x.ravel(), np.meshgrid(theta0, lutdata['axisTau'], lutdata['axisWav'], lutdata['axisAerosolModel'])))).T
    Fd_NL_vector = RegularGridInterpolator((lutInterpolatedData['axisTheta0'], lutInterpolatedData['axisTau'], lutInterpolatedData['axisWav'], lutInterpolatedData['axisAerosolModel']), lutdata['Fd_NL0'])(coordsFd_NL0)
    coordsT_NL0 = np.array(list(map(lambda x: x.ravel(), np.meshgrid(theta, theta0, lutdata['axisTau'], lutdata['axisWav'], lutdata['axisAerosolModel'])))).T
    T_NL_vector = RegularGridInterpolator((lutInterpolatedData['axisTheta'], lutInterpolatedData['axisTheta0'], lutInterpolatedData['axisTau'], lutInterpolatedData['axisWav'], lutInterpolatedData['axisAerosolModel']), lutdata['T_NL0'])(coordsT_NL0)

    lutInterpolatedData['FdT_NL'] = Fd_NL_vector.reshape((NLTAU, NLWAV, NLAEROSOLMODEL)) * T_NL_vector.reshape((NLTAU, NLWAV, NLAEROSOLMODEL))

    EQWAV_HGHT = [
        np.interp(mhght, lutInterpolatedData['axisHeight'], lutInterpolatedData['EQWAV'][0, :]),
        np.interp(mhght, lutInterpolatedData['axisHeight'], lutInterpolatedData['EQWAV'][1, :]),
        np.interp(mhght, lutInterpolatedData['axisHeight'], lutInterpolatedData['EQWAV'][2, :]),
    ]

    INT_NL_height = (lutInterpolatedData['INT_NL']).copy()
    SBAR_NL_height = (lutInterpolatedData['SBAR_NL']).copy()
    FdT_NL_height = (lutInterpolatedData['FdT_NL']).copy()
    OPTH_NL_height = (lutInterpolatedData['OPTH_NL']).copy()

    for iiWav in range(3):  # only for 466, 553 and 645nm
        INT_NL_height[:, iiWav, :] = interp1d(lutInterpolatedData['axisWav'], lutInterpolatedData['INT_NL'], axis=1, fill_value='extrapolate', bounds_error=False)(EQWAV_HGHT[iiWav])
        SBAR_NL_height[:, iiWav, :] = interp1d(lutInterpolatedData['axisWav'], lutInterpolatedData['SBAR_NL'], axis=1, fill_value='extrapolate', bounds_error=False)(EQWAV_HGHT[iiWav])
        FdT_NL_height[:, iiWav, :] = interp1d(lutInterpolatedData['axisWav'], lutInterpolatedData['FdT_NL'], axis=1, fill_value='extrapolate', bounds_error=False)(EQWAV_HGHT[iiWav])
        OPTH_NL_height[:, iiWav, :] = interp1d(lutInterpolatedData['axisWav'], lutInterpolatedData['OPTH_NL'], axis=1, fill_value='extrapolate', bounds_error=False)(EQWAV_HGHT[iiWav])

    return {
        'fine_aerosol_model': geometry['fine_aerosol_model'],
        'INT_NL': INT_NL_height,
        'SBAR_NL': SBAR_NL_height,
        'FdT_NL': FdT_NL_height,
        'OPTH_NL': OPTH_NL_height,
        'EXTNORM_NL': (lutInterpolatedData['EXTNORM_NL0']).copy()
    }


def lookupToPolynomial(lookupdata):

    thetas_INT_NL = np.zeros((5, 4, 6))
    thetas_SBAR_NL = np.zeros((5, 4, 6))
    thetas_FdT_NL = np.zeros((5, 4, 6))

    INT_NL = lookupdata['INT_NL']
    SBAR_NL = lookupdata['SBAR_NL']
    FdT_NL = lookupdata['FdT_NL']
    OPTH_NL = lookupdata['OPTH_NL']
    EXTNORM_NL = lookupdata['EXTNORM_NL']

    for ii in range(5):  # aerosol model
        x = np.log(OPTH_NL[:, 1, ii] + 1.0)  # AOD @ 550nm
        A = np.vstack([x**5, x**4, x**3, x**2, x, np.ones(len(x))]).T

        iHtHHt = np.linalg.inv(A.T.dot(A)).dot(A.T)

        for jj in range(4):
            y = np.log(INT_NL[:, jj, ii] + 1.0)
            thetas_INT_NL[ii, jj, :] = iHtHHt.dot(y)

            y = np.log(SBAR_NL[:, jj, ii] + 1.0)
            thetas_SBAR_NL[ii, jj, :] = iHtHHt.dot(y)

            y = np.log(FdT_NL[:, jj, ii] + 1.0)
            thetas_FdT_NL[ii, jj, :] = iHtHHt.dot(y)

    return {
        'thetas_INT_NL': thetas_INT_NL,
        'thetas_SBAR_NL': thetas_SBAR_NL,
        'thetas_FdT_NL': thetas_FdT_NL,
        'OPTH_NL': OPTH_NL,
        'EXTNORM_NL': EXTNORM_NL,
        'fine_aerosol_model': lookupdata['fine_aerosol_model'],
    }


def prepareModels(modisData, settings):

    Npixels = modisData['Npixels']
    month = modisData['scantime'].month

    print('  Number of pixels to be retrieved: {}'.format(Npixels))
    print('  Loading radiative transfer lookup-tables')

    # The LUT is based on NASA's Dark Target over land lookup tables (https://darktarget.gsfc.nasa.gov/reference/code)
    nc = Dataset('models/LUT/DarkTargetLUT.nc', 'r')
    LUTdata = {
        'aerosolmap': getattr(nc['LUTdata'], 'aerosolmapmonth{:02d}'.format(month)),
        'axisHeight': getattr(nc['LUTdata'], 'axisHeight'),
        'axisTheta': getattr(nc['LUTdata'], 'axisTheta'),
        'axisTheta0': getattr(nc['LUTdata'], 'axisTheta0'),
        'axisPhi': getattr(nc['LUTdata'], 'axisPhi'),
        'axisWav': getattr(nc['LUTdata'], 'axisWav'),
        'axisAerosolModel': getattr(nc['LUTdata'], 'axisAerosolModel'),
        'axisTau': getattr(nc['LUTdata'], 'axisTau')
    }
    N_Height, N_Theta, N_Theta0, N_Phi = len(LUTdata['axisHeight']), len(LUTdata['axisTheta']), len(LUTdata['axisTheta0']), len(LUTdata['axisPhi'])
    N_Wav, N_AerosolModel, N_Tau = len(LUTdata['axisWav']), len(LUTdata['axisAerosolModel']), len(LUTdata['axisTau'])
    LUTdata['EQWAV'] = getattr(nc['LUTdata'], 'EQWAV').reshape((3, N_Height))
    LUTdata['INT_NL0'] = getattr(nc['LUTdata'], 'INT_NL0').reshape((N_Phi, N_Theta, N_Theta0, N_Tau, N_Wav, N_AerosolModel))
    LUTdata['Fd_NL0'] = getattr(nc['LUTdata'], 'Fd_NL0').reshape((N_Theta0, N_Tau, N_Wav, N_AerosolModel))
    LUTdata['SBAR_NL0'] = getattr(nc['LUTdata'], 'SBAR_NL0').reshape((N_Theta0, N_Tau, N_Wav, N_AerosolModel))
    LUTdata['T_NL0'] = getattr(nc['LUTdata'], 'T_NL0').reshape((N_Theta, N_Theta0, N_Tau, N_Wav, N_AerosolModel))
    LUTdata['OPTH_NL0'] = getattr(nc['LUTdata'], 'OPTH_NL0').reshape((N_Tau, N_Wav, N_AerosolModel))
    LUTdata['EXTNORM_NL0'] = getattr(nc['LUTdata'], 'EXTNORM_NL0').reshape((N_Tau, N_Wav, N_AerosolModel))
    nc.close()

    print('  Preparing radiative transfer lookup-tables')
    lonsvec, latsvec = modisData['lon'].ravel(), modisData['lat'].ravel()
    gFull = np.hstack((lonsvec[:, np.newaxis], latsvec[:, np.newaxis]))

    MASK_FULL = modisData['MASK']
    MASK_FULL_true_pos = np.where(MASK_FULL)[0]

    N_parts = int(np.ceil(Npixels / float(settings['max_pixels_per_part'])))
    part_indices = np.linspace(0, Npixels - 1, N_parts + 1, dtype=int)

    partData = []
    for iPart in range(1, N_parts + 1):

        iStart, iEnd = MASK_FULL_true_pos[part_indices[iPart - 1]], MASK_FULL_true_pos[part_indices[iPart]]

        MASKl = np.zeros(Npixels, dtype=bool)
        if iPart < N_parts:
            MASKl[np.arange(part_indices[iPart - 1], part_indices[iPart])] = True
        else:
            MASKl[np.arange(part_indices[iPart - 1], part_indices[iPart] + 1)] = True

        MASKf = MASK_FULL.copy()
        MASKf[:iStart] = False
        if iPart < N_parts:
            MASKf[iEnd:] = False

        partData.append({
            'MASKl': MASKl,
            'MASKf': MASKf
        })

    data = []
    for ii in range(Npixels):
        geometry = {
            'theta': modisData['theta'][ii],
            'theta0': modisData['theta0'][ii],
            'phi': modisData['phi'][ii],
            'phi0': modisData['phi0'][ii],
            'mhght': modisData['mhght'][ii],
            'fine_aerosol_model': modisData['Aerosol_Type_Land'][ii],
        }
        data.append({'geometry': geometry, 'LUTdata': LUTdata})

    # prepare models in parallel
    p = Pool(settings['N_processes'])
    polylookups = np.array(p.map(prepareFcn, data))
    p.close()
    p.join()
    return polylookups, MASK_FULL, partData, gFull, LUTdata['aerosolmap']


def combinePolyLookupList(polylookuplist, aerosolmodelindex=None):

    POLYORDER = polylookuplist[0]['thetas_INT_NL'].shape[2]
    NLWAV = 4

    if aerosolmodelindex is not None:
        EXTNORM_NL_matrix = np.array(list(map(lambda x: x['EXTNORM_NL'][:, :, aerosolmodelindex], polylookuplist)))
        OPTH_NL_matrix = np.array(list(map(lambda x: x['OPTH_NL'][:, :, aerosolmodelindex], polylookuplist)))

        INT_NL_matrix = np.array(list(map(lambda x: x['thetas_INT_NL'][aerosolmodelindex, :, :], polylookuplist))).reshape((-1, POLYORDER))
        SBAR_NL_matrix = np.array(list(map(lambda x: x['thetas_SBAR_NL'][aerosolmodelindex, :, :], polylookuplist))).reshape((-1, POLYORDER))
        FdT_NL_matrix = np.array(list(map(lambda x: x['thetas_FdT_NL'][aerosolmodelindex, :, :], polylookuplist))).reshape((-1, POLYORDER))
    else:
        EXTNORM_NL_matrix = np.array(list(map(lambda x: x['EXTNORM_NL'][:, :, x['fine_aerosol_model']], polylookuplist)))
        OPTH_NL_matrix = np.array(list(map(lambda x: x['OPTH_NL'][:, :, x['fine_aerosol_model']], polylookuplist)))

        INT_NL_matrix = np.array(list(map(lambda x: x['thetas_INT_NL'][x['fine_aerosol_model'], :, :], polylookuplist))).reshape((-1, POLYORDER))
        SBAR_NL_matrix = np.array(list(map(lambda x: x['thetas_SBAR_NL'][x['fine_aerosol_model'], :, :], polylookuplist))).reshape((-1, POLYORDER))
        FdT_NL_matrix = np.array(list(map(lambda x: x['thetas_FdT_NL'][x['fine_aerosol_model'], :, :], polylookuplist))).reshape((-1, POLYORDER))
    INT_NL_matrix = np.vstack([INT_NL_matrix[0::NLWAV, :], INT_NL_matrix[1::NLWAV, :], INT_NL_matrix[2::NLWAV, :], INT_NL_matrix[3::NLWAV, :]])
    SBAR_NL_matrix = np.vstack([SBAR_NL_matrix[0::NLWAV, :], SBAR_NL_matrix[1::NLWAV, :], SBAR_NL_matrix[2::NLWAV, :], SBAR_NL_matrix[3::NLWAV, :]])
    FdT_NL_matrix = np.vstack([FdT_NL_matrix[0::NLWAV, :], FdT_NL_matrix[1::NLWAV, :], FdT_NL_matrix[2::NLWAV, :], FdT_NL_matrix[3::NLWAV, :]])

    return {'INT_NL': INT_NL_matrix, 'SBAR_NL': SBAR_NL_matrix, 'FdT_NL': FdT_NL_matrix, 'EXTNORM_NL': EXTNORM_NL_matrix, 'OPTH_NL': OPTH_NL_matrix}


def constructPriors(logaodPlus1, fmf, aodPrior, fmfPrior, surfPrior):
    priorAOD = {'mean': logaodPlus1,
                'sill': aodPrior['sill'],
                'range': aodPrior['range'],
                'nugget': aodPrior['nugget'],
                'p': aodPrior['p']
                }
    priorFMF = {'mean': fmf,
                'sill': fmfPrior['sill'],
                'range': fmfPrior['range'],
                'nugget': fmfPrior['nugget'],
                'p': fmfPrior['p']
                }
    priorRhos2113 = {
        'mean': surfPrior['mean211'],
        'std': surfPrior['std211'],
    }
    priorRhos644 = {
        'mean': surfPrior['mean644'],
        'std': surfPrior['std644'],
    }
    priorRhos553 = {
        'mean': surfPrior['mean550'],
        'std': surfPrior['std550'],
    }
    priorRhos466 = {
        'mean': surfPrior['mean466'],
        'std': surfPrior['std466'],
    }
    return priorAOD, priorFMF, priorRhos466, priorRhos553, priorRhos644, priorRhos2113


def covariance_function(d, pr_sill, pr_range, pr_nugget, pr_p=1.0):
    c = pr_sill * np.exp(-3.0 * (d / pr_range)**pr_p)
    c[d == 0] += pr_nugget
    return c


def dist(lat1, lon1, lat2, lon2):
    lat1R, lat2R, lon1R, lon2R = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
    dlon = lon2R - lon1R
    dlat = lat2R - lat1R
    R = 6378.1
    a = (np.sin(dlat / 2.0))**2 + np.cos(lat1R) * np.cos(lat2R) * (np.sin(dlon / 2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R * c
    return d


class InverseProblem(object):
    def __init__(self, gGranule, polylookupMats_fine, polylookupMats_coarse, priorLogAODPlus1, priorFMF,
                 priorRhos466, priorRhos553, priorRhos644, priorRhos2113,
                 observations, noiseiCov, noiseE, mask, settings):
        self.g = gGranule
        self.Ng = self.g.shape[0]
        self.polylookupMats_fine = polylookupMats_fine
        self.polylookupMats_coarse = polylookupMats_coarse
        self.simuData = None
        self.invnoiseCov = noiseiCov
        self.noiseE = noiseE
        self.obs = observations
        self.mask = mask
        self.priorLogAODPlus1 = priorLogAODPlus1.copy()
        self.priorFMF = priorFMF.copy()
        self.priorRhos466 = priorRhos466.copy()
        self.priorRhos553 = priorRhos553.copy()
        self.priorRhos644 = priorRhos644.copy()
        self.priorRhos2113 = priorRhos2113.copy()
        self.settings = settings
        self.Nmodels = mask.sum()
        self.priorLogAODPlus1['cov'] = np.zeros((self.Ng, self.Ng))
        self.priorFMF['cov'] = np.zeros((self.Ng, self.Ng))

        for ii in range(self.Ng):
            distanceinKm = dist(self.g[:, 1], self.g[:, 0], self.g[ii, 1], self.g[ii, 0])
            self.priorLogAODPlus1['cov'][ii, :] = covariance_function(distanceinKm, priorLogAODPlus1['sill'], priorLogAODPlus1['range'], priorLogAODPlus1['nugget'], priorLogAODPlus1['p'])
            self.priorFMF['cov'][ii, :] = covariance_function(distanceinKm, priorFMF['sill'], priorFMF['range'], priorFMF['nugget'], priorFMF['p'])

        self.priorRhos = {
            'iCov': diags(np.concatenate((1.0 / self.priorRhos466['std']**2, 1.0 / self.priorRhos553['std']**2,
                                          1.0 / self.priorRhos644['std']**2, 1.0 / self.priorRhos2113['std']**2)), 0),
            'mean': np.concatenate((self.priorRhos466['mean'], self.priorRhos553['mean'], self.priorRhos644['mean'], self.priorRhos2113['mean']))
        }

        print("    Inverting prior covariances")
        if settings['useSpatialCorrelations']:
            self.priorLogAODPlus1['iCov'] = np.linalg.inv(self.priorLogAODPlus1['cov'])
            self.priorFMF['iCov'] = np.linalg.inv(self.priorFMF['cov'])
        else:
            self.priorLogAODPlus1['iCov'] = diags(1.0 / self.priorLogAODPlus1['cov'].diagonal(), 0)
            self.priorFMF['iCov'] = diags(1.0 / self.priorFMF['cov'].diagonal(), 0)

    def functional_prior_val_and_Jac(self, priorX, X):
        dX = (X - priorX['mean'])[:, np.newaxis]
        iCovXdX = priorX['iCov'].dot(dX)
        return np.asarray(dX.T.dot(iCovXdX))[0, 0], 2.0 * np.asarray(iCovXdX).ravel()

    def functional_value_and_Jacobian_logscalemeas(self, x):
        logaodplus1, fmf, rho_s = x[0 * self.Ng:1 * self.Ng], x[1 * self.Ng:2 * self.Ng], x[2 * self.Ng:6 * self.Ng]
        fval_logaodplus1, J_fval_logaodplus1 = self.functional_prior_val_and_Jac(self.priorLogAODPlus1, logaodplus1)
        fval_fmf, J_fval_fmf = self.functional_prior_val_and_Jac(self.priorFMF, fmf)
        fval_rhos, J_fval_rhos = self.functional_prior_val_and_Jac(self.priorRhos, rho_s)
        rhostar_fine, J_rhostar_logaodplus1_fine, J_rhostar_rhos_fine = self.simulate(logaodplus1, rho_s, self.polylookupMats_fine, True)
        rhostar_coarse, J_rhostar_logaodplus1_coarse, J_rhostar_rhos_coarse = self.simulate(logaodplus1, rho_s, self.polylookupMats_coarse, True)
        rhostar = np.tile(fmf, (4,)) * rhostar_fine + (1.0 - np.tile(fmf, (4,))) * rhostar_coarse
        J_rhostar_logaodplus1 = np.tile(fmf, (4,)) * J_rhostar_logaodplus1_fine + (1.0 - np.tile(fmf, (4,))) * J_rhostar_logaodplus1_coarse
        J_rhostar_fmf = rhostar_fine - rhostar_coarse
        J_rhostar_logaodplus1 = 1.0 / (1.0 + rhostar) * J_rhostar_logaodplus1
        J_rhostar_fmf = 1.0 / (1.0 + rhostar) * J_rhostar_fmf
        d_rho_TOA_d_rhos = np.tile(fmf, (4,)) * J_rhostar_rhos_fine + (1.0 - np.tile(fmf, (4,))) * J_rhostar_rhos_coarse
        dObs = (np.log(rhostar + 1.0) + self.noiseE - self.obs)[:, np.newaxis]
        iNdObs = self.invnoiseCov.dot(dObs)
        func_val = (dObs.T.dot(iNdObs))[0, 0] + fval_logaodplus1 + fval_fmf + fval_rhos
        J = np.zeros(len(x))
        J[0 * self.Ng:1 * self.Ng] = (2.0 * iNdObs.ravel() * J_rhostar_logaodplus1).reshape((-1, self.Ng)).sum(axis=0) + J_fval_logaodplus1
        J[1 * self.Ng:2 * self.Ng] = (2.0 * iNdObs.ravel() * J_rhostar_fmf).reshape((-1, self.Ng)).sum(axis=0) + J_fval_fmf
        J[2 * self.Ng:6 * self.Ng] = (2.0 * iNdObs.ravel() * d_rho_TOA_d_rhos) + J_fval_rhos
        return func_val, J

    def Jacobian_aod_fmf_rhos(self, x):
        logaodplus1, fmf, rho_s = x[0 * self.Ng:1 * self.Ng], x[1 * self.Ng:2 * self.Ng], x[2 * self.Ng:6 * self.Ng]
        fval_logaodplus1, J_fval_logaodplus1 = self.functional_prior_val_and_Jac(self.priorLogAODPlus1, logaodplus1)
        fval_fmf, J_fval_fmf = self.functional_prior_val_and_Jac(self.priorFMF, fmf)
        fval_rhos, J_fval_rhos = self.functional_prior_val_and_Jac(self.priorRhos, rho_s)
        rhostar_fine, J_rhostar_logaodplus1_fine, J_rhostar_rhos_fine = self.simulate(logaodplus1, rho_s, self.polylookupMats_fine, True)
        rhostar_coarse, J_rhostar_logaodplus1_coarse, J_rhostar_rhos_coarse = self.simulate(logaodplus1, rho_s, self.polylookupMats_coarse, True)
        rhostar = np.tile(fmf, (4,)) * rhostar_fine + (1.0 - np.tile(fmf, (4,))) * rhostar_coarse
        J_rhostar_logaodplus1 = np.tile(fmf, (4,)) * J_rhostar_logaodplus1_fine + (1.0 - np.tile(fmf, (4,))) * J_rhostar_logaodplus1_coarse
        J_rhostar_fmf = rhostar_fine - rhostar_coarse
        J_rhostar_logaodplus1 = 1.0 / (1.0 + rhostar) * J_rhostar_logaodplus1
        J_rhostar_fmf = 1.0 / (1.0 + rhostar) * J_rhostar_fmf
        d_rho_TOA_d_rhos = np.tile(fmf, (4,)) * J_rhostar_rhos_fine + (1.0 - np.tile(fmf, (4,))) * J_rhostar_rhos_coarse
        return J_rhostar_logaodplus1, J_rhostar_fmf, d_rho_TOA_d_rhos

    def simulate(self, logaodplus1, rhos, lookupMatrices, derivatives=True):
        NLWAV = 4
        logaodplus1Matrix = np.hstack((logaodplus1[:, np.newaxis]**5,
                                       logaodplus1[:, np.newaxis]**4,
                                       logaodplus1[:, np.newaxis]**3,
                                       logaodplus1[:, np.newaxis]**2,
                                       logaodplus1[:, np.newaxis],
                                       np.ones_like(logaodplus1[:, np.newaxis])))

        INT_NL = np.exp((lookupMatrices['INT_NL'] * np.tile(logaodplus1Matrix, (NLWAV, 1))).sum(axis=1)) - 1.0
        SBAR_NL = np.exp((lookupMatrices['SBAR_NL'] * np.tile(logaodplus1Matrix, (NLWAV, 1))).sum(axis=1)) - 1.0
        FdT_NL = np.exp((lookupMatrices['FdT_NL'] * np.tile(logaodplus1Matrix, (NLWAV, 1))).sum(axis=1)) - 1.0
        rhostar = INT_NL + (FdT_NL * rhos) / (1.0 - SBAR_NL * rhos)
        if derivatives:
            logaodplus1stacked = np.tile(logaodplus1, (NLWAV,))

            d_INT_NL_d_logaodplus1 = (INT_NL + 1.0) * (5.0 * lookupMatrices['INT_NL'][:, 0] * logaodplus1stacked**4 +
                                                       4.0 * lookupMatrices['INT_NL'][:, 1] * logaodplus1stacked**3 +
                                                       3.0 * lookupMatrices['INT_NL'][:, 2] * logaodplus1stacked**2 +
                                                       2.0 * lookupMatrices['INT_NL'][:, 3] * logaodplus1stacked +
                                                       lookupMatrices['INT_NL'][:, 4])
            d_SBAR_NL_d_logaodplus1 = (SBAR_NL + 1.0) * (5.0 * lookupMatrices['SBAR_NL'][:, 0] * logaodplus1stacked**4 +
                                                         4.0 * lookupMatrices['SBAR_NL'][:, 1] * logaodplus1stacked**3 +
                                                         3.0 * lookupMatrices['SBAR_NL'][:, 2] * logaodplus1stacked**2 +
                                                         2.0 * lookupMatrices['SBAR_NL'][:, 3] * logaodplus1stacked +
                                                         lookupMatrices['SBAR_NL'][:, 4])
            d_FdT_NL_d_logaodplus1 = (FdT_NL + 1.0) * (5.0 * lookupMatrices['FdT_NL'][:, 0] * logaodplus1stacked**4 +
                                                       4.0 * lookupMatrices['FdT_NL'][:, 1] * logaodplus1stacked**3 +
                                                       3.0 * lookupMatrices['FdT_NL'][:, 2] * logaodplus1stacked**2 +
                                                       2.0 * lookupMatrices['FdT_NL'][:, 3] * logaodplus1stacked +
                                                       lookupMatrices['FdT_NL'][:, 4])

            J_rhostar_logaodplus1 = d_INT_NL_d_logaodplus1 + (d_FdT_NL_d_logaodplus1 * rhos * (1.0 - SBAR_NL * rhos) +
                                                              (FdT_NL * rhos) * (d_SBAR_NL_d_logaodplus1 * rhos)) / (1.0 - SBAR_NL * rhos)**2
            J_rhostar_rhos = (FdT_NL * (1.0 - SBAR_NL * rhos) + FdT_NL * rhos * SBAR_NL) / (1.0 - SBAR_NL * rhos)**2
            return rhostar, J_rhostar_logaodplus1, J_rhostar_rhos
        else:
            return rhostar


def BARretrieve(g, AOD0, FMF0, AODprior, FMFprior, surfPrior, polylookupMats_fine, polylookupMats_coarse, MeasData, MeasNoiseiCov, MeasNoiseE, MASK, bounds, settings):

    logAODPlus1 = np.log(AOD0 + 1.0)
    priorLogAODPlus1, priorFMF, priorRhos466, priorRhos553, priorRhos644, priorRhos2113 = constructPriors(logAODPlus1, FMF0, AODprior, FMFprior, surfPrior)

    # the inverse problem
    IP = InverseProblem(g, polylookupMats_fine, polylookupMats_coarse, priorLogAODPlus1, priorFMF, priorRhos466, priorRhos553, priorRhos644, priorRhos2113,
                        MeasData, MeasNoiseiCov, MeasNoiseE, MASK, settings)

    # AOD, ETA, rhos466, rhos644
    x0 = np.concatenate((priorLogAODPlus1['mean'], priorFMF['mean'], priorRhos466['mean'], priorRhos553['mean'], priorRhos644['mean'], priorRhos2113['mean']))

    BFGSbounds = np.vstack((np.tile(bounds[0, :], (IP.Ng, 1)),
                            np.tile(bounds[1, :], (IP.Ng, 1)),
                            np.tile(bounds[2, :], (IP.Ng, 1)),
                            np.tile(bounds[3, :], (IP.Ng, 1)),
                            np.tile(bounds[4, :], (IP.Ng, 1)),
                            np.tile(bounds[5, :], (IP.Ng, 1))))

    print('    Retrieving...', end='', flush=True)
    res = minimize(IP.functional_value_and_Jacobian_logscalemeas, x0, method='L-BFGS-B', bounds=BFGSbounds, jac=True,
                   options={
                       'ftol': settings['BFGSftol'],
                       'gtol': settings['BFGSgtol'],
                       'disp': False,
                       'maxcor': 10,
                       'maxls': 100,
                       'maxiter': settings['BFGSmaxiter']
                   })
    print('Done! (# iterations: {}, status message from L-BFGS-B: "{}")'.format(res.nit, res.message.decode('utf-8')))

    logaodPlus1, fmf, rhos466, rhos553, rhos644, rhos2113 = res.x[:IP.Ng], res.x[IP.Ng:2 * IP.Ng], res.x[2 * IP.Ng:3 * IP.Ng], res.x[3 * IP.Ng:4 * IP.Ng], res.x[4 * IP.Ng:5 * IP.Ng], res.x[5 * IP.Ng:6 * IP.Ng]
    aod550 = np.exp(logaodPlus1) - 1.0
    aod550_fine = fmf * aod550
    aod550_coarse = (1.0 - fmf) * aod550

    OPTH = polylookupMats_fine['OPTH_NL'][:, :, 1]  # 1 = AOD @ 550nm
    EXTNORM466f = polylookupMats_fine['EXTNORM_NL'][:, :, 0]
    EXTNORM644f = polylookupMats_fine['EXTNORM_NL'][:, :, 2]
    EXTNORM466c = polylookupMats_coarse['EXTNORM_NL'][:, :, 0]
    EXTNORM644c = polylookupMats_coarse['EXTNORM_NL'][:, :, 2]

    aod466, aod644 = -9.999 * np.ones_like(aod550), -9.999 * np.ones_like(aod550)
    for ii in range(len(aod550)):
        aod466_fine = np.interp(aod550_fine[ii], OPTH[ii, :], EXTNORM466f[ii, :]) * aod550_fine[ii]
        aod466_coarse = np.interp(aod550_coarse[ii], OPTH[ii, :], EXTNORM466c[ii, :]) * aod550_coarse[ii]
        aod466[ii] = aod466_fine + aod466_coarse
        aod644_fine = np.interp(aod550_fine[ii], OPTH[ii, :], EXTNORM644f[ii, :]) * aod550_fine[ii]
        aod644_coarse = np.interp(aod550_coarse[ii], OPTH[ii, :], EXTNORM644c[ii, :]) * aod550_coarse[ii]
        aod644[ii] = aod644_fine + aod644_coarse

    if settings['quantifyUncertainty']:
        print('    Quantifying uncertainties...', end='', flush=True)
        if IP.invnoiseCov is None:
            variance_X = np.concatenate((IP.priorLogAODPlus1['cov'].diagonal(), IP.priorFMF['cov'].diagonal()))
        else:
            J_AODPlus1, J_FMF, d_rho_TOA_d_rhos = IP.Jacobian_aod_fmf_rhos(res.x)
            J_AODPlus1 = J_AODPlus1.reshape((-1, IP.Ng))
            J_FMF = J_FMF.reshape((-1, IP.Ng))
            d_rho_TOA_d_rhos = d_rho_TOA_d_rhos.reshape((-1, IP.Ng))
            J = np.zeros((4 * IP.Ng, 6 * IP.Ng))
            for ii in range(IP.Ng):
                J[0 * IP.Ng + ii, 0 * IP.Ng + ii] = J_AODPlus1[0, ii]
                J[1 * IP.Ng + ii, 0 * IP.Ng + ii] = J_AODPlus1[1, ii]
                J[2 * IP.Ng + ii, 0 * IP.Ng + ii] = J_AODPlus1[2, ii]
                J[3 * IP.Ng + ii, 0 * IP.Ng + ii] = J_AODPlus1[3, ii]
                J[0 * IP.Ng + ii, 1 * IP.Ng + ii] = J_FMF[0, ii]
                J[1 * IP.Ng + ii, 1 * IP.Ng + ii] = J_FMF[1, ii]
                J[2 * IP.Ng + ii, 1 * IP.Ng + ii] = J_FMF[2, ii]
                J[3 * IP.Ng + ii, 1 * IP.Ng + ii] = J_FMF[3, ii]
                J[0 * IP.Ng + ii, 2 * IP.Ng + ii] = d_rho_TOA_d_rhos[0, ii]
                J[1 * IP.Ng + ii, 3 * IP.Ng + ii] = d_rho_TOA_d_rhos[1, ii]
                J[2 * IP.Ng + ii, 4 * IP.Ng + ii] = d_rho_TOA_d_rhos[2, ii]
                J[3 * IP.Ng + ii, 5 * IP.Ng + ii] = d_rho_TOA_d_rhos[3, ii]
            if settings['useSpatialCorrelations']:
                inv_Cov_LogAODPlus1FMF = (J.T.dot(IP.invnoiseCov.dot(J))) + np.linalg.inv(block_diag(IP.priorLogAODPlus1['cov'], IP.priorFMF['cov'], np.array(diags(np.concatenate((IP.priorRhos466['std']**2, IP.priorRhos553['std']**2, IP.priorRhos644['std']**2, IP.priorRhos2113['std']**2)), 0).todense())))
            else:
                inv_Cov_LogAODPlus1FMF = (J.T.dot(IP.invnoiseCov.dot(J))) + diags(np.concatenate((IP.priorLogAODPlus1['iCov'].diagonal(), IP.priorFMF['iCov'].diagonal(), IP.priorRhos466['std']**2, IP.priorRhos553['std']**2, IP.priorRhos644['std']**2, IP.priorRhos2113['std']**2)), 0)
            cov_LogAODPlus1FMF = np.linalg.inv(inv_Cov_LogAODPlus1FMF)
            variance_X = np.array(cov_LogAODPlus1FMF.diagonal()).ravel()

        std_logaodPlus1, std_fmf, std_rhos = np.sqrt(variance_X[:IP.Ng]), np.sqrt(variance_X[IP.Ng:2 * IP.Ng]), np.sqrt(variance_X[2 * IP.Ng:])
        print('Done!')

        AODinterval68 = np.clip(np.exp(norm.interval(0.68, loc=logaodPlus1, scale=std_logaodPlus1)) - 1.0, a_min=0.0, a_max=np.inf)
        AODinterval90 = np.clip(np.exp(norm.interval(0.90, loc=logaodPlus1, scale=std_logaodPlus1)) - 1.0, a_min=0.0, a_max=np.inf)
        AODinterval95 = np.clip(np.exp(norm.interval(0.95, loc=logaodPlus1, scale=std_logaodPlus1)) - 1.0, a_min=0.0, a_max=np.inf)
        FMFinterval68 = np.clip(norm.interval(0.68, loc=fmf, scale=std_fmf), a_min=0.0, a_max=1.0)
        FMFinterval90 = np.clip(norm.interval(0.90, loc=fmf, scale=std_fmf), a_min=0.0, a_max=1.0)
        FMFinterval95 = np.clip(norm.interval(0.95, loc=fmf, scale=std_fmf), a_min=0.0, a_max=1.0)
        BARresult = {
            'BAR_AOD': aod550,
            'BAR_logAODplus1': logaodPlus1,
            'BAR_logAODplus1_std': std_logaodPlus1,
            'BAR_AOD466': aod466,
            'BAR_AOD644': aod644,
            'BAR_AOD_percentile_2.5': AODinterval95[0],
            'BAR_AOD_percentile_5.0': AODinterval90[0],
            'BAR_AOD_percentile_16.0': AODinterval68[0],
            'BAR_AOD_percentile_84.0': AODinterval68[1],
            'BAR_AOD_percentile_95.0': AODinterval90[1],
            'BAR_AOD_percentile_97.5': AODinterval95[1],
            'BAR_FMF': fmf,
            'BAR_FMF_std': std_fmf,
            'BAR_FMF_percentile_2.5': FMFinterval68[0],
            'BAR_FMF_percentile_5.0': FMFinterval90[0],
            'BAR_FMF_percentile_16.0': FMFinterval95[0],
            'BAR_FMF_percentile_84.0': FMFinterval68[1],
            'BAR_FMF_percentile_95.0': FMFinterval90[1],
            'BAR_FMF_percentile_97.5': FMFinterval95[1],
            'BAR_rhos466': rhos466,
            'BAR_rhos553': rhos553,
            'BAR_rhos644': rhos644,
            'BAR_rhos211': rhos2113,
            'BAR_rhos466_std': std_rhos[0 * IP.Ng:1 * IP.Ng],
            'BAR_rhos553_std': std_rhos[1 * IP.Ng:2 * IP.Ng],
            'BAR_rhos644_std': std_rhos[2 * IP.Ng:3 * IP.Ng],
            'BAR_rhos211_std': std_rhos[3 * IP.Ng:4 * IP.Ng],
        }
    else:
        BARresult = {
            'BAR_AOD': aod550,
            'BAR_logAODplus1': logaodPlus1,
            'BAR_AOD466': aod466,
            'BAR_AOD644': aod644,
            'BAR_FMF': fmf,
            'BAR_rhos466': rhos466,
            'BAR_rhos553': rhos553,
            'BAR_rhos644': rhos644,
            'BAR_rhos211': rhos2113,
        }

    return BARresult

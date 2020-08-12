# BARalgorithm
Bayesian Aerosol Retrieval algorithm
version 1.01 (12 August 2020)

This experimental code and data is meant for testing purposes only.
As some of the models distributed with the code represent year 2014
it should be noted that it is possible that the algorithm does
not produce optimal estimates for data acquired long before or
after year 2014.
In all cases, the user should consult the author of the code
before drawing any conclusions. The code is licensed under the
MIT licence (found below).

For the most recent version of the codes, please see https://github.com/TUT-ISI/BARalgorithm

## MODELS

The models directory contains model files needed to run the code and some of the files are not available at Github.
To download the LUT model (models/LUT directory) and surface reflectance prior model (models/SurfaceReflectance/2014)
please download the code package from Zenodo (https://doi.org/10.5281/zenodo.1182939) and copy the
contents of these two directories of the Zenodo package to corresponding BARalgorithm folders.

---

**Reference:**
Lipponen, A., Mielonen, T., Pitkänen, M. R. A., Levy, R. C., Sawyer, V. R., Romakkaniemi, S., Kolehmainen, V., and Arola, A.:
Bayesian Aerosol Retrieval Algorithm for MODIS AOD retrieval over land, Atmos. Meas. Tech., 11, 1529–1547, 2018.
https://doi.org/10.5194/amt-11-1529-2018.

Contact information:
Antti Lipponen
Finnish Meteorological Institute
antti.lipponen@fmi.fi

---

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

---


## 1) Bayesian Aerosol Retrieval (BAR) algorithm

  Bayesian Aerosol Retrieval (BAR) algorithm is a statistical aerosol retrieval algorithm and this
  code runs BAR algorithm using MODIS data over land.
  All technical details can be found from the article published in the Atmospheric Measurement Techniques
  (https://doi.org/10.5194/amt-11-1529-2018). The BAR algorithm is strongly based on NASA's Dark Target
  aerosol retrieval algorithm (https://darktarget.gsfc.nasa.gov/) and, for example, the radiative transfer
  simulation lookup-tables (LUTs) are based on the Dark Target LUTs distributed with the Dark
  Target stand-alone land code (https://darktarget.gsfc.nasa.gov/reference/code).
  
  
## 2) Requirements to run the code
  
  The code is written for Python 3 and it requires some packages and datafiles to run.
  The code has been tested on Linux/Ubuntu environment and it may or may not run on other operating systems.
  The code depends on, at least, the following packages and libraries:
  * NumPy
  * Scipy
  * netcdf4
  * pyhdf
  * scikit-learn
  
  We recommend the Anaconda Python (https://www.anaconda.com/download) to run the code (Python version >= 3.6).
  Please see the next sections for the instructions to download the required external data (Models), and
  install all required packages and libraries (Installation).

## 3) Models

  To run the BAR algorithm, you need certain model data files, some of the data files are included with the code
  and some of them you need to download before you can start using the algorithm code. All data files for models
  are located in the 'models' directory.
  There are 5 different models in BAR algorithm:
    AOD
      This is the expected value for AOD prior model. The expected value is based on MAC-V2 aerosol climatology
      (https://doi.org/10.1002/jame.20035 or https://doi.org/10.5194/gmd-10-433-2017) and the filename should be 'gt_t_00550nm.nc'.
      You can download the file from:
        ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_t_00550nm.nc
      If you want to download it from the command line, type:
        wget -O "models/AOD/gt_t_00550nm.nc" "ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_t_00550nm.nc"

    FMF
      This is the expected value for FMF prior model. The expected value is based on MAC-V2 aerosol climatology
      (https://doi.org/10.1002/jame.20035 or https://doi.org/10.5194/gmd-10-433-2017) and the filename should be 'gt_ff_0550nm.nc'.
      You can download the file from:
        ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_ff_0550nm.nc
      If you want to download it from the command line, type:
        wget -O "models/FMF/gt_ff_0550nm.nc" "ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_ff_0550nm.nc"

    ApproximationError
      The approximation error model is based on 2014 AERONET data and the required file 'approximationerror.json' is included with the code.

    LUT
      LUT directory includes the radiative transfer model lookup-tables. The lookup-tables are included with the code
      and are based on NASA's Dark Target over land algorithm (https://darktarget.gsfc.nasa.gov/). The original lookup-tables
      are distributed with the Dark Target over land stand-alone code (https://darktarget.gsfc.nasa.gov/reference/code).

    SurfaceReflectance
      SurfaceReflectance directory includes the prior model files corresponding to year 2014 for MODIS bands 1, 3, 4, and 7.
      The surface reflectance prior models are based on MODIS MCD43C3 data products
      (https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mcd43c3).


## 4) Installation

  (These instructions are tested in Linux/Ubuntu and OS X environment)
  We recommend Anaconda Python and its environments to be used to run the BAR algorithm.
  To check if Anaconda is correctly installed type 'conda' and you should get information about the conda manager.
  
  Some of the packages may only be available the 'conda-forge' repository (https://conda-forge.org/).
  Start the installation by adding the conda-forge to your conda by typing:
    conda config --add channels conda-forge

  Creating an environment is recommended so you can install all the necessary packages and they won't
  disturb your regular/default Python environment. To create environment 'BARenv' for BAR and install the necessary
  Python packages type:
    conda create -n BARenv python=3.6 numpy scipy netcdf4 pyhdf scikit-learn imageio

  Now activate the 'BARenv' environment by typing:
    source activate BARenv

  If everything is OK you should now see the name of the environment in your command prompt. While the environment
  is active you can navigate to your BAR algorithm folder and start running the code.

  When you are done with your computations you can return back to your default Python configuration by
  deactivating the environment. The deactivation is done by typing:
    source deactivate


## 5) Use

  You can now run the retrieval by typing:
  python BAR.py INPUTFILENAME [OUTPUTFILENAME]
  
  INPUTFILENAME = path to MODIS MOD04_L2/MYD04_L2 granule hdf file
  OUTPUTFILENAME (optional) = the netCDF file to be used to save the results

  You can also give the following command line options:
    -quantifyuncertainty: Quantify the uncertainties related to retrieval and save the uncertainty estimates into the outputfile
    -nospatialcorrelation: Do not use spatial correlation model
    -noapproximationerror: Do not use approximation error model

  Also the AOD, FMF, and surface reflectance prior model parameters can be tuned from the command line using the commands -AODprior, -FMFprior, and -surfaceprior
  Examples:
    -AODprior='{"range": 250, "p": 2.0, "sill": 0.5, "nugget": 0.001}'
    -FMFprior='{"range": 75, "p": 1.25, "sill": 0.1, "nugget": 0.01}'
    -surfaceprior='{"meancoef": 1.05, "stdcoef": 1.5}'


## 6) Example

  In this example, a MODIS MOD04_L2 datafile is first downloaded and then
  the BAR algorithm is run to retrieve the AOD and FMF
  (one easy way to download single granules of data is to use the
  NASA Worldview https://worldview.earthdata.nasa.gov/).
  The resulting file will be in netCDF format and can be viewed
  for example with the Panoply Data Viewer (https://www.giss.nasa.gov/tools/panoply/).

  In this example case we will retrieve the aerosol properties over Finland
  in 2010 during a heavy smoke event from the fires in Russia.
  The example case can be viewed in NASA Worldview at: https://go.nasa.gov/2FfsNuG

  Download the data:
    wget --no-check-certificate -O "granuleFiles/MOD04_L2.A2010210.0950.006.2015046201421.hdf" "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/MOD04_L2/2010/210/MOD04_L2.A2010210.0950.006.2015046201421.hdf"

  Run the retrieval:
    source activate BARenv
    python BAR.py granuleFiles/MOD04_L2.A2010210.0950.006.2015046201421.hdf granuleFiles/Finland20100729.nc
    source deactivate

  Now the results are stored in the netCDF file 'granuleFiles/Finland20100729.nc'.

---

## Questions/Requests/Collaboration: antti.lipponen@fmi.fi

---

## Installation and test commands without explanation:

```
wget -O "models/AOD/gt_t_00550nm.nc" "ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_t_00550nm.nc"
wget -O "models/FMF/gt_ff_0550nm.nc" "ftp://ftp-projects.zmaw.de/aerocom/climatology/MACv2_2017/550nm_2005/gt_ff_0550nm.nc"
conda config --add channels conda-forge
conda create -n BARenv python=3.6 numpy scipy netcdf4 pyhdf scikit-learn imageio
source activate BARenv
wget --no-check-certificate -O "granuleFiles/MOD04_L2.A2010210.0950.006.2015046201421.hdf" "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/MOD04_L2/2010/210/MOD04_L2.A2010210.0950.006.2015046201421.hdf"
python BAR.py granuleFiles/MOD04_L2.A2010210.0950.006.2015046201421.hdf granuleFiles/Finland20100729.nc
source deactivate
```

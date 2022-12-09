import pandas as pd
import scipy
import math
import os
import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from functions import operations as op

def clasts_rasterize(ClastImageFilePath, ClastSizeListCSVFilePath, RasterFileWritingPath, field = "Clast_length", parameter="quantile", cellsize=1, percentile=0.5, plot=True, figuresize = (15,20), T=10):
    
#     Converts the clast information from vector type to raster type.
#     ClastImageFilePath: Path of the geotiff file used to realize the clast detection and measurement
#     ClastSizeListCSVFilePath: Path of the CSV file containing the list of previously detected and measured clasts
#     RasterFileWritingPath: Path to be used for writing the raster produced by the present function
#     field: field (i.e. clast dimension) to be considered for computation (default = "Clast_length"):
#         - Clast_length
#         - Clast_width
#         - Ellipse_major_axis
#         - Ellipse_minor_axis
#         - Equivalent_diameter
#         - Score
#         - Orientation
#         - Surface_area
#         - Clast_elongation
#         - Ellipse_elongation
#         - Clast_circularity
#         - Ellipse_circularity
#         - Van_Rijn_dimensionless_diameter:              Dimensionless grain size calculated on the equivalent diameter #Van Rijn (1993)
#         - Soulsby_critical_shields:                     Empirical critical Shields parameter calculated on the equivalent diameter #Soulsby (1997)
#         - Shields_critical_shear_stress:                Empirical shear stress calculated on the equivalent diameter #Shields (1936) 
#         - Shields_critical_shear_velocity:              Empirical shear velocity calculated on the equivalent diameter #Shields (1936) 
#         - Shields_critical_grain_reynolds_number:       Empirical grain Reynold's number calculated on the equivalent diameter #Shields (1936) 
#         - Hjulstrom_deposition_velocity:                Empirical current velocity (m/s) related to the clast deposition according to Hjulström's diagram #DOI: 10.1115/OMAE2013-10524
#         - Hjulstrom_erosion_velocity:                   Empirical current velocity (m/s) required in order to mobilize the clast according to Hjulström's diagram #DOI: 10.1115/OMAE2013-10524
#         - Leroux_wave_orbital_velocity:                 Empirical wave orbital velocity (m/s) as a function of the wave period T required to mobilize the clast according to Le Roux's formula #DOI: 10.1016/S0037-0738(01)00105-1
#     parameter: Parameter to be computed for each cell: 
#         - "quantile": returns the quantile valued for the threshold specified by the "percentile" keyword 
#         - "density": returns the density of objects per cell size unit
#         - "average": returns the average value for each cell
#         - "std": returns the standard deviation for each cell
#         - "kurtosis": returns the kurtosis size for each cell
#         - "skewness": returns the skewness value for each cell
#         - "sorting": returns the sorting value for each cell
#     cellsize: Wanted output raster cell size (same unit as the geotiff file used to realize the clast detection and measurement
#     percentile: Percentile to be used for computing the quantile of each cell (default = 0.5, i.e. median)
#     plot: Switch for displaying the produced maps (default = True)
#     figuresize: Size of the displayed figure (default = (10,10))
#     T: Wave period for estimating the wave orbital velocity of Le roux (default = 10)
    

    clasts = pd.read_csv(ClastSizeListCSVFilePath)
    local_clasts = clasts.copy()

    #Error prevention
    local_clasts = local_clasts[local_clasts['Clast_length'].isnull()==False]
    local_clasts = local_clasts[local_clasts['Clast_length']>0]
    local_clasts = local_clasts[local_clasts['Clast_width']>0]
    local_clasts = local_clasts.reset_index(drop=True)

    #Orientation
    if str.lower(field) == "orientation":
        local_clasts['u'], local_clasts['v'] = op.pol2cart(np.ones(np.shape(local_clasts)[0]), np.deg2rad(local_clasts['Orientation']))

    #Elongation
    if str.lower(field) == "clast_elongation":
        local_clasts['Clast_elongation'] = local_clasts['Clast_width']/local_clasts['Clast_length']

    if str.lower(field) == "ellipse_elongation":
        local_clasts['Ellipse_elongation'] = local_clasts['Ellipse_minor_axis']/local_clasts['Ellipse_major_axis']

    #Equivalent diameter
    if str.lower(field) == "equivalent_diameter" or str.lower(field) == "hjulstrom_deposition_velocity" or str.lower(field) == "hjulstrom_erosion_velocity" or str.lower(field) == "leroux_wave_orbital_velocity" or str.lower(field) == "van_rijn_dimensionless_diameter" or str.lower(field) == "soulsby_critical_shields" or str.lower(field) == "shields_critical_shear_stress" or str.lower(field) == "shields_critical_shear_velocity" or str.lower(field) == "shields_critical_grain_reynolds_number" or str.lower(field) == "clast_circularity" or str.lower(field) == "ellipse_circularity":
        local_clasts['Equivalent_diameter'] = 2*np.sqrt(local_clasts['Surface_area']/np.pi)

    #Circularity
    if str.lower(field) == "clast_circularity":
        local_clasts['Clast_circularity'] = local_clasts['Equivalent_diameter']/local_clasts['Clast_length']

    if str.lower(field) == "ellipse_circularity":
        local_clasts['Ellipse_circularity'] = local_clasts['Equivalent_diameter']/local_clasts['Ellipse_major_axis']

    #Sediment sorting
    if str.lower(parameter) == "sorting":
        local_clasts['Clast_length'] = -np.log2(local_clasts['Clast_length']/0.001)
        local_clasts['Clast_width'] = -np.log2(local_clasts['Clast_width']/0.001)
        local_clasts['Ellipse_major_axis'] = -np.log2(local_clasts['Ellipse_major_axis']/0.001)
        local_clasts['Ellipse_minor_axis'] = -np.log2(local_clasts['Ellipse_minor_axis']/0.001)
        if str.lower(field) == "equivalent_diameter":
            local_clasts['Equivalent_diameter'] = -np.log2(local_clasts['Equivalent_diameter']/0.001)

    #Sediment motion thresholds
    if str.lower(field) == "van_rijn_dimensionless_diameter" or str.lower(field) == "soulsby_critical_shields" or str.lower(field) == "shields_critical_shear_stress" or str.lower(field) == "shields_critical_shear_velocity" or str.lower(field) == "shields_critical_grain_reynolds_number":
        rho_water = 1.025
        rho_sed = 2.600
        Rd = rho_sed / rho_water
        g = 9.81
        mu = 1.07e-3
        nu = mu / rho_water

        local_clasts['Van_Rijn_dimensionless_diameter'] = local_clasts['Equivalent_diameter']*((g*(Rd-1))/(nu**2))**(1/3)

        local_clasts['Soulsby_critical_shields'] = (0.24/local_clasts['Van_Rijn_dimensionless_diameter'])+(0.055*(1-np.exp(-0.02*local_clasts['Van_Rijn_dimensionless_diameter'])))
        local_clasts['Soulsby_critical_shields'][local_clasts['Van_Rijn_dimensionless_diameter']<=5] = (0.30/(1+(1.2*local_clasts['Van_Rijn_dimensionless_diameter'])))+(0.055*(1-np.exp(-0.02*local_clasts['Van_Rijn_dimensionless_diameter'])))

        local_clasts['Shields_critical_shear_stress'] = local_clasts['Soulsby_critical_shields'] * (rho_sed - rho_water) * g * local_clasts['Equivalent_diameter']

        local_clasts['Shields_critical_shear_velocity'] = (local_clasts['Shields_critical_shear_stress'] / rho_water) ** (1/2)

        local_clasts['Shields_critical_grain_reynolds_number'] = (local_clasts['Shields_critical_shear_velocity']*local_clasts['Equivalent_diameter']) / nu

    if str.lower(field) == "hjulstrom_deposition_velocity": 
        local_clasts['Hjulstrom_deposition_velocity'] = 77*(local_clasts['Equivalent_diameter']/(1+(24*local_clasts['Equivalent_diameter'])))

    if str.lower(field) == "hjulstrom_erosion_velocity":
        rho_water = 1.025
        rho_sed = 2.600
        Rd = rho_sed / rho_water
        g = 9.81
        mu = 1.07e-3
        nu = mu / rho_water
        local_clasts['Hjulstrom_erosion_velocity'] = 1.5*((nu/local_clasts['Equivalent_diameter'])**0.8) + 0.85*((nu/local_clasts['Equivalent_diameter'])**0.35) + 9.5*((Rd*g*local_clasts['Equivalent_diameter'])/(1+(2.25*Rd*g*local_clasts['Equivalent_diameter'])))

    if str.lower(field) == "leroux_wave_orbital_velocity":
        rho_water = 1.025
        rho_sed = 2.600
        g = 981
        D = local_clasts['Equivalent_diameter']*100
        rho_gam = rho_sed - rho_water
        mu = 1.07e-2
        Dd = D*(((rho_water*g*rho_gam)/(mu**2)))**(1/3)
        Wd = np.zeros(np.shape(Dd))
        for i in range(0,np.shape(Dd)[0]-1):
            if (Dd[i]<1.2538):
                Wd[i] = (0.2354*Dd[i])**2
            elif (Dd[i] < 2.9074):
                Wd[i] = (0.208*Dd[i]-0.0652)**(3/2)
            elif (Dd[i] < 22.9866):
                Wd[i] = (0.2636*Dd[i]-0.37)
            elif (Dd[i] < 134.92150):
                Wd[i] = (0.8255*Dd[i]-5.4)**(2/3)
            else:
                Wd[i] = (2.531*Dd[i]+160)**(1/2)
        Wd[Wd==0] = np.nan
        theta_wl = (0.0246*Wd)**(-0.55)
        local_clasts['Leroux_wave_orbital_velocity'] = ((theta_wl*g*D*rho_gam)/(((np.pi*T)/(rho_water*mu))**0.5))/100;

    #Convolution prep
    local_clasts['y'] = clasts['y']-np.min(clasts['y'])
    local_clasts['x'] = clasts['x']-np.min(clasts['x'])
    n_rows = math.ceil(np.max(local_clasts['y'])/cellsize)
    n_cols = math.ceil(np.max(local_clasts['x'])/cellsize)
    p = np.zeros((n_rows,n_cols))
    p1 = np.zeros((n_rows,n_cols))
    p2 = np.zeros((n_rows,n_cols))
    if str.lower(parameter) == "distribution":
        p = np.zeros((n_rows,n_cols, 100))
        p1 = np.zeros((n_rows,n_cols, 100))
        p2 = np.zeros((n_rows,n_cols, 100))

    #Convolution
    for m in tqdm(range(0, n_rows)):
        for n in range(0, n_cols):
            crop = local_clasts[(local_clasts['y']>=m*cellsize) & 
                                 (local_clasts['y']<(m+1)*cellsize) & 
                                 (local_clasts['x']>=n*cellsize) & 
                                 (local_clasts['x']<(n+1)*cellsize)]
            if str.lower(field) == "orientation":
                if np.shape(crop)[0]>0:
                    if str.lower(parameter) == "quantile":
                        p1[m,n] = np.nanquantile(crop['u'], percentile)
                        p2[m,n] = np.nanquantile(crop['v'], percentile)
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "density":
                        p[m,n] = (np.shape(crop[field])[0]+1)/(cellsize**2)
                    if str.lower(parameter) == "average":
                        p1[m,n] = np.nanmean(crop['u'])
                        p2[m,n] = np.nanmean(crop['v'])
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "mode":
                        p1[m,n] = stats.mode(crop['u'], nan_policy='omit')
                        p2[m,n] = stats.mode(crop['v'], nan_policy='omit')
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "kurtosis":
                        p1[m,n] = scipy.stats.kurtosis(crop['u'], nan_policy = 'omit')
                        p2[m,n] = scipy.stats.kurtosis(crop['v'], nan_policy = 'omit')
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "skewness":
                        p1[m,n] = scipy.stats.skew(crop['u'], nan_policy = 'omit')
                        p2[m,n] = scipy.stats.skew(crop['v'], nan_policy = 'omit')
                        p[m,n] = np.rad2deg(cop.art2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "std":
                        p1[m,n] = np.nanstd(crop['u'])
                        p2[m,n] = np.nanstd(crop['v'])
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
                    if str.lower(parameter) == "distribution":
                        for l in range(1,100):
                            p1[m,n,l-1] = np.nanquantile(crop['u'], l/100)
                            p2[m,n,l-1] = np.nanquantile(crop['v'], l/100)
                            p[m,n, l-1] = np.rad2deg(op.cart2pol(p1[m,n,l-1], p2[m,n,l-1])[1])
                    if str.lower(parameter) == "sorting":
                        p1[m,n] = ((np.nanquantile(crop['u'], 0.84)-np.nanquantile(crop['u'], 0.16))/4) + ((np.nanquantile(crop['u'], 0.95)-np.nanquantile(crop['u'], 0.05))/6.6)
                        p2[m,n] = ((np.nanquantile(crop['v'], 0.84)-np.nanquantile(crop['v'], 0.16))/4) + ((np.nanquantile(crop['v'], 0.95)-np.nanquantile(crop['v'], 0.05))/6.6)
                        p[m,n] = np.rad2deg(op.cart2pol(p1[m,n], p2[m,n])[1])
            else:
                if np.shape(crop)[0]>0:
                    if str.lower(parameter) == "quantile":
                        p[m,n] = np.nanquantile(crop[field], percentile)
                    if str.lower(parameter) == "density":
                        p[m,n] = (np.shape(crop[field])[0]+1)/(cellsize**2)
                    if str.lower(parameter) == "average":
                        p[m,n] = np.nanmean(crop[field])
                    if str.lower(parameter) == "mode":
                        p[m,n] = stats.mode(crop[field], nan_policy='omit')
                    if str.lower(parameter) == "kurtosis":
                        p[m,n] = scipy.stats.kurtosis(crop[field], nan_policy = 'omit')
                    if str.lower(parameter) == "skewness":
                        p[m,n] = scipy.stats.skew(crop[field], nan_policy = 'omit')
                    if str.lower(parameter) == "std":
                        p[m,n] = np.nanstd(crop[field])
                    if str.lower(parameter) == "distribution":
                        for l in range(1,100):
                            p[m,n,l-1] = np.nanquantile(crop[field], l/100)
                    if str.lower(parameter) == "sorting":
                        p[m,n] = ((np.nanquantile(crop[field], 0.84)-np.nanquantile(crop[field], 0.16))/4) + ((np.nanquantile(crop[field], 0.95)-np.nanquantile(crop[field], 0.05))/6.6)
    print('Saving...')
    raster = gdal.Open(ClastImageFilePath, gdal.GA_ReadOnly)
    geotransform = raster.GetGeoTransform()

    driver = gdal.GetDriverByName("GTiff")
    arr_out = p.copy()


    if str.lower(parameter) == "distribution":
        if np.shape(np.shape(arr_out))==(2,):
            n_band = 1
        else:
            n_band = np.shape(arr_out)[2]
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows, n_band-1, gdal.GDT_Float64)
        outdata.SetGeoTransform([np.min(clasts['x']), cellsize, 0, np.min(clasts['y']), 0, cellsize ])
        outdata.SetProjection(raster.GetProjection())
        for i in range(1,n_band):
            outdata.GetRasterBand(i).WriteArray(arr_out[:,:,i-1])
            outdata.GetRasterBand(i).SetNoDataValue(0)
            outdata.GetRasterBand(i).SetDescription('D'+str(i))

    else:
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform([np.min(clasts['x']), cellsize, 0, np.min(clasts['y']), 0, cellsize ])
        outdata.SetProjection(raster.GetProjection())
        outdata.GetRasterBand(1).WriteArray(arr_out)
        outdata.GetRasterBand(1).SetNoDataValue(0)

    outdata.FlushCache() 
    outdata = None
    ds=None

    del(outdata)
    
    if plot==True:
        image = np.dstack([raster.GetRasterBand(1).ReadAsArray(), raster.GetRasterBand(2).ReadAsArray(),raster.GetRasterBand(3).ReadAsArray()] )
        fig = plt.figure(figsize = figuresize)
        ax1 = fig.add_subplot(2,1,1)
        ax1.imshow(image, interpolation='none')
        ax1.set_title("Ortho-image")
        ax2 = fig.add_subplot(2,1,2) 
        pos = ax2.imshow(np.flipud(p), interpolation='none')
        if parameter=="quantile":
            ax2.set_title("D"+str(int(percentile*100))+" map")
        else:
            ax2.set_title(parameter+" map")
        fig.colorbar(pos, ax=ax2)
    print('File saved: '+RasterFileWritingPath)
    return p
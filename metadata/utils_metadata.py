import utm
import scipy
import math
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Determines return value units.
    return c * r

def from_lat_lon_to_utm_and_epsg(lat, lon):
    """https://github.com/Turbo87/utm"""
    # Note lat lon can be also NumPy arrays.
    # In this case zone letter and number will be calculate from first entry.
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    southern_hemisphere_TrueFalse = True if zone_letter < 'N' else False
    epsg_code = 32600 + zone_number + southern_hemisphere_TrueFalse * 100
    return (easting, northing, zone_number, zone_letter, epsg_code)

def gaussian_filter_with_nans(U, sigma, trunc=4.0):
    # Since the reprojection into utm leads to distortions (=nans) we need to take care of this during filtering
    # From David in https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    WW[WW == 0] = np.nan
    filtered_U = VV / WW
    return filtered_U

def get_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # (0.11764706, 0.56470588, 1.0)] dodgerblue
    colors = [(1, 1, 1), (0.0, 0.0, .8)]  # White to electric blue
    cmap_name = 'white_electric_blue'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    return cm

def calc_volume_glacier(points_thickness, area=0):
    # A potential drawback of this method is that I am randomly sampling in epsg:4326. In a utm projection
    # such sampling does not turn out to be uniform. Returned volume in km3.
    N = len(points_thickness)
    volume = np.sum(points_thickness) * 0.001 * area / N
    return volume

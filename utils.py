import os
import random
import math
from random import seed

#import fiona
#import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import numpy as np
import rioxarray
import xarray as xr

from shapely.geometry import Point, Polygon
from tqdm import tqdm
from scipy.special import binom
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import find_peaks
from PIL import Image, ImageDraw
from skimage.morphology import erosion
#from shapely.geometry import mapping
#from shapely.wkt import loads
from pyproj import Transformer


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self._calc_points(self.r)

    def _calc_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = self._bezier(self.p,self.numpoints)

    def _bezier(self, points, num=200):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(self.bernstein(N - 1, i, t), points[i])
        return curve

class GlacierMaskGenerator():
    """
    Create a mask from a bezier curve
    """
    def __init__(self, height, width, channels, offset=[0,0]):

        self.height = height
        self.width = width
        self.channels = channels
        center_x = width // 2
        center_y = height // 2

        self.offset = np.array(offset) + np.array([center_x, center_y])

    def _get_curve(self, points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def _ccw_sort(self, p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def _get_bezier_curve(self, a, sharp=0.2, smooth=0):
        p = np.arctan(smooth)/np.pi+.5
        a = self._ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = self._get_curve(a, r=sharp, method="var")
        x,y = c.T
        return x,y, a

    def _get_random_points(self, n, scale=0.8, rec=0):
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(self._ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= .7/n) or rec>=200:
            return a*scale
        else:
            return self._get_random_points(n=n, scale=scale, rec=rec+1)

    def _generate_mask(self, poly):
        img = np.zeros((self.height, self.width, self.channels), np.uint8)
        minx, miny, maxx, maxy = poly.bounds
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
        box_patch = [[x,y] for x in range(minx,maxx+1) for y in range(miny,maxy+1)]

        for pb in box_patch:
            pt = Point(pb[0],pb[1])
            if(poly.contains(pt)):
                img[pb[0], pb[1]] = 1

        return img

    def sample(self, min_scale, max_scale, n, sharp, smooth, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        scale = int(random.triangular(min_scale, max_scale, min_scale))
        scale_offset = np.array([scale//2, scale//2])
        a = self._get_random_points(n=n, scale=scale) + self.offset - scale_offset
        x,y, _ = self._get_bezier_curve(a, sharp=sharp, smooth=smooth)
        mask = self._generate_mask(Polygon(zip(x,y)))
        return 1 - mask

class MaskGenerator():
    """
    Create a mask from a box and brush stroke
    """
    def __init__(self, height, width, box_h, box_w):
        self.min_num_vertex = 4
        self.max_num_vertex = 12
        self.mean_angle = 2*math.pi / 5
        self.angle_range = 2*math.pi / 15
        self.min_width = 12
        self.max_width = 40

        self.height = height
        self.width = width
        self.box_h = box_h
        self.box_w = box_w

    def sample(self):
        box = self.random_bbox()
        box_mask = self.bbox2mask(box)
        irregular = self.brush_stroke_mask()
        mask = np.logical_or(irregular, box_mask).squeeze()
        return 1 - mask[..., np.newaxis]

    def brush_stroke_mask(self):
        """Generate brush stroke mask \\
        (Algorithm 1) from `Generative Image Inpainting with Contextual Attention`(Yu et al., 2019) \\
        Returns:
           output with shape [1, 1, H, W]

        """

        H = self.height
        W = self.width

        average_radius = np.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)
            angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(
                        2*np.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)),
                        int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        return mask

    def random_bbox(self):
        """Generate a random tlhw.

        Returns:
            tuple: (top, left, height, width)

        """
        img_height = self.height
        img_width = self.width
        maxt = img_height - 0 - 128
        maxl = img_width - 0 - 128
        t = np.random.randint(0, maxt)
        l = np.random.randint(0, maxl)

        return (t, l, self.box_h, self.box_w)

    def bbox2mask(self, bbox):
        """Generate mask tensor from bbox.

        Args:
            bbox: tuple, (top, left, height, width)

        Returns:
            output with shape [1, 1, H, W]

        """
        img_height = self.height
        img_width = self.width
        mask = np.zeros((1, 1, img_height, img_width))
        h = np.random.randint(32 // 2 + 1)
        w = np.random.randint(32 // 2 + 1)
        mask[:, :, bbox[0]+h: bbox[0]+bbox[2]-h,
            bbox[1]+w: bbox[1]+bbox[3]-w] = 1.
        return mask

def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in km^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = np.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*np.sin(np.radians(f))
        zp = 1 + e*np.sin(np.radians(f))
        area_list.append(
            np.pi * b**2 * (
                np.log(zp/zm) / (2*e) +
                np.sin(np.radians(f)) / (zp*zm)))

    am2 = pixel_size / 360. * (area_list[0] - area_list[1]) # area in m2
    return 1e-6 * am2

def bbox(coords_list):
    box = []
    for i in (0,1):
        res = sorted(coords_list, key=lambda x:x[i])
        box.append((res[0][i], res[-1][i]))
    ret = [box[0][0], box[0][1], box[1][0], box[1][1]]
    return ret


def bounding_box(img, label):
    print('Entered bbox function')
    a = np.where(img == label)
    if a[0].size > 0:
        box = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    else:
        box = 0, 0, 0, 0
    return box



def get_glacier_metrics(DEM, mask):
    # get all glacier pixels
    gp = DEM[np.where(mask == 1)]
    # remove invalid pixels
    gp = gp[gp >= 0]
    # sort all values
    gp = np.sort(gp)
    return np.min(gp), np.mean(gp), np.median(gp), np.max(gp)


def find_maximum(image, shape, sigma=5, elevation=500, iter=1):
    maxima_patch_r = np.zeros_like(image)
    maxima_patch_c = np.zeros_like(image)
    patch = gaussian_filter(image, sigma=sigma)

    for r in range(shape):
        row = patch[r,:]
        peaks, _ = find_peaks(row, height=np.max(patch)-elevation)
        maxima_patch_r[r,peaks] = 1
    for c in range(shape):
        col = patch[:, c]
        peaks, _ = find_peaks(col, height=np.max(patch)-elevation)
        maxima_patch_c[peaks, c] = 1

    maxima_patch = maxima_patch_r + maxima_patch_c
    maxima_patch[maxima_patch > 0] = 1
    return binary_dilation(maxima_patch, iterations=iter)


# We use latitude, the horizonal line of every tile is slightly smaller than the one below it
# when above equator. 
def calc_volume(rgiid, centers, lat, shape, diff):
    idx = centers.loc[centers['RGIId'].isin([rgiid])]['rows'].values[0]
    latitude_array = lat[idx-int(shape/2):idx+int(shape/2)]
    latitude_matrix = np.zeros((shape, shape))
    for i in range(shape):
        latitude_matrix[:,i] = (np.cos(np.radians(latitude_array[i]))*30)*30

    result = np.multiply(latitude_matrix, diff)
    return np.sum(result)


def get_distance_mask(patch):
    new_patch = np.ones_like(patch).astype(float)
    new_patch = np.negative(new_patch)
    #np.seterr(invalid='ignore')

    fill_value = 0
    while True:
        if np.sum(patch) == 0:   
            break

        patch_erosion = erosion(patch)
        patch -= patch_erosion
        new_patch[np.where(patch == 1)] = fill_value
        patch = patch_erosion
        fill_value += 1

    new_patch[new_patch == 0] = 0.01
    new_patch = new_patch/(fill_value-1)
    return new_patch


def get_minmax_latlon(dem_path, img=None):
    """
    Consider change to output corners only
    and create seperate function to extract min and max.
    """
    dataset = rio.open(dem_path, 'r')
    if img is None:
        img = dataset.read(1)
        height, width = img.shape
    else:
        img = img
        height, width = img.shape

    # get corner lon and lat from DEM GeoTIFF
    topRight = rio.transform.xy(dataset.transform, 0, width-1, offset='center')
    topLeft = rio.transform.xy(dataset.transform, 0, 0, offset='center')
    bottomRight = rio.transform.xy(dataset.transform, height-1, width-1, offset='center')
    bottomLeft = rio.transform.xy(dataset.transform, height-1, 0, offset='center')

    # transform output to useful format
    max_lat = max(topRight[1], topLeft[1], bottomRight[1], bottomLeft[1])
    min_lat = min(topRight[1], topLeft[1], bottomRight[1], bottomLeft[1])
    max_lon = max(topRight[0], topLeft[0], bottomRight[0], bottomLeft[0])
    min_lon = min(topRight[0], topLeft[0], bottomRight[0], bottomLeft[0])

    return max_lat, min_lat, max_lon, min_lon

def get_minmax_latlon_nico(dem_path, img=None):

    dataset = rioxarray.open_rasterio(dem_path)
    min_lon, min_lat, max_lon, max_lat = dataset.rio.bounds()
    x_res, y_res = dataset.rio.resolution()

    min_lat += x_res / 2.
    max_lat -= x_res / 2.
    min_lon += x_res / 2.
    max_lon -= x_res / 2.

    return max_lat, min_lat, max_lon, min_lon

def contains_glacier(dem_paths, glaciers, add=0):
    # table to store filepath, glacier presence (True/False) and list of glacier ids in tile
    columns = ['filepath', 'contains_glacier', 'RGIId']
    df = pd.DataFrame(columns = columns)

    for i in range(len(dem_paths)):
        contains_glacier = True
        print(dem_paths[i])
        max_lat, min_lat, max_lon, min_lon = get_minmax_latlon_nico(dem_paths[i])

        # locate all glaciers within tile
        current_glaciers = glaciers.loc[glaciers['CenLon'] >= min_lon - add]
        current_glaciers = current_glaciers.loc[current_glaciers['CenLon'] <= max_lon + add]
        current_glaciers = current_glaciers.loc[current_glaciers['CenLat'] >= min_lat - add]
        current_glaciers = current_glaciers.loc[current_glaciers['CenLat'] <= max_lat + add]

        # check if above eliminated all
        if current_glaciers.empty:
            contains_glacier = False

        # add results to output table
        current = pd.DataFrame(
            data=[[dem_paths[i], contains_glacier, current_glaciers['RGIId'].tolist()]],
            columns = columns
        )

        df = pd.concat([df, current], ignore_index = True)

    return df


def contains_glacier_(dem_paths, glaciers, add=0):
    # table to store filepath, glacier presence (True/False) and list of glacier ids in tile
    columns = ['filepath', 'contains_glacier', 'RGIId']
    df = pd.DataFrame(columns = columns)

    contains_glacier = True
    max_lat, min_lat, max_lon, min_lon = get_minmax_latlon(dem_paths)

    # locate all glaciers within tile
    current_glaciers = glaciers.loc[glaciers['CenLon'] >= min_lon - add]
    current_glaciers = current_glaciers.loc[current_glaciers['CenLon'] <= max_lon + add]
    current_glaciers = current_glaciers.loc[current_glaciers['CenLat'] >= min_lat - add]
    current_glaciers = current_glaciers.loc[current_glaciers['CenLat'] <= max_lat + add]

    # check if above eliminated all
    if current_glaciers.empty:
        contains_glacier = False

    # add results to output table
    current = pd.DataFrame(
        data=[[dem_paths, contains_glacier, current_glaciers['RGIId'].tolist()]],
        columns = columns
    )

    df = pd.concat([df, current], ignore_index = True)

    return df

def rasterio_clip(dem_path, polygon_set, epsg):
    mask = rioxarray.open_rasterio(dem_path)
    mask = xr.zeros_like(mask)
    mask.rio.write_nodata(1., inplace=True)

    # clip all glaciers at once
    mask = mask.rio.clip(polygon_set['geometry'].to_list(), epsg, drop=False, invert=True, all_touched=False)

    # if you prefer the loop over single glaciers
    # for glacier in tqdm(range(len(polygon_set)), leave=False):
    #    geom = polygon_set['geometry'][glacier]
    #    mask = mask.rio.clip([geom], epsg, drop=False, invert=True, all_touched=False)

    return mask


def coords_to_xy(dem_path, glaciers, crs_from=4326, crs_to=4326):
    """
    This function return the indexes corresponding to the glacier center geographic coordinates,
    as well as their names. I modified this function replacing rio with rioxarray and in particular
    using get_loc, see
    https://stackoverflow.com/questions/61457310/how-can-i-find-the-indices-equivalent-to-a-specific-selection-of-xarray
    """
    dataset = rioxarray.open_rasterio(dem_path)

    coords = []
    for glacier in range(len(glaciers)):
        lon = glaciers['CenLon'][glacier]
        lat = glaciers['CenLat'][glacier]

        transformer = Transformer.from_crs(crs_from, crs_to) # necessary ?
        lat, lon = transformer.transform(lat, lon) # necessary ?


        ##rows, cols = rio.transform.rowcol(dataset.transform, x, y)
        # non mi piace questo .index. Puo ritornare valori negativi, che non capisco.
        # rows, cols = dataset.index(lon, lat) # Get the (row, col) index of the pixel containing (x, y).

        dims = dataset.dims #  <-- ('band', 'y', 'x')
        rows = dataset.indexes[dims[1]].get_loc(lat,  method="nearest")
        cols = dataset.indexes[dims[2]].get_loc(lon,  method="nearest")
        coords.append([rows, cols])

    RGI = glaciers['RGIId'].to_list()

    # coords will be a ndarray of shape (len(glaciers), 2)
    return np.array(coords), RGI
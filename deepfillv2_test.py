import argparse
import os
import time
import rioxarray
import xarray as xr
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import haversine
import Deepfillv2.libs.misc as misc
from Deepfillv2.libs.data import ImageDataset_box, ImageDataset_segmented, get_transforms

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


parser = argparse.ArgumentParser(description='Test inpainting')
parser.add_argument("--image", type=str,
                    default="/home/nico/PycharmProjects/skynet/code/dataset/test/RGI_11_size_256/images/",
                    help="input folder with image files")
parser.add_argument("--mask", type=str,
                    default="/home/nico/PycharmProjects/skynet/code/dataset/test/RGI_11_size_256/masks/",
                    help="input folder with mask files")
parser.add_argument("--fullmask", type=str,
                    default="/home/nico/PycharmProjects/skynet/code/dataset/test/RGI_11_size_256/masks_full/",
                    help="input folder with full mask files")
parser.add_argument("--out", type=str,
                    default="/home/nico/PycharmProjects/skynet/code/dataset/output/RGI_11_size_256_test/",
                    help="path to saved results")
parser.add_argument("--checkpoint", type=str,
                    default="/home/nico/PycharmProjects/skynet/code/Deepfillv2/callbacks/checkpoints/box_model/states_it500.pth",
                    help="path to the checkpoint file")
parser.add_argument("--tfmodel", action='store_true',
                    default=False, help="use model from models_tf.py?")
parser.add_argument('--burned', default=True, type=str2bool, const=True,
                    nargs='?', help='Run all burned glaciers')
parser.add_argument('--all',  default=False, type=str2bool, const=True, 
                    nargs='?', help='Run all glaciers in input folder')
parser.add_argument('--config', type=str, default="Deepfillv2/configs/train.yaml", help="Path to yaml config file")


def main():
    # --------------------------------------------------------------------------------------- #
    #                                         Config                                          #
    # --------------------------------------------------------------------------------------- #
    args = parser.parse_args()
    config = misc.get_config(args.config)  # some stuff in train.yaml may be useful
    # --------------------------------------------------------------------------------------- #
    #                                         Model                                           #
    # --------------------------------------------------------------------------------------- #

    # What is Deepfillv2.libs.networks_tf as compared to Deepfillv2.libs.networks ?
    if args.tfmodel:
        from Deepfillv2.libs.networks_tf import Generator
    else:
        from Deepfillv2.libs.networks_radio3 import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # set up network
    # generator = Generator(cnum_in=4, cnum=48, return_flow=False).to(device) # OLD
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in + 3, cnum_out=1, cnum=48, return_flow=False).to(device)
    generator_state_dict = torch.load(args.checkpoint)['G']
    generator.load_state_dict(generator_state_dict)
    generator.eval() # maffe added this: eval mode

    # --------------------------------------------------------------------------------------- #
    #                             Setup the glacier list                                      #
    # --------------------------------------------------------------------------------------- #

    RGI_burned = ['RGI60-11.00562', 'RGI60-11.00590', 'RGI60-11.00603', 'RGI60-11.00638', 'RGI60-11.00647',
                  'RGI60-11.00689', 'RGI60-11.00695', 'RGI60-11.00846', 'RGI60-11.00950', 'RGI60-11.01024', 
                  'RGI60-11.01041', 'RGI60-11.01067', 'RGI60-11.01144', 'RGI60-11.01199', 'RGI60-11.01296', 
                  'RGI60-11.01344', 'RGI60-11.01367', 'RGI60-11.01376', 'RGI60-11.01473', 'RGI60-11.01509', 
                  'RGI60-11.01576', 'RGI60-11.01604', 'RGI60-11.01698', 'RGI60-11.01776', 'RGI60-11.01786', 
                  'RGI60-11.01790', 'RGI60-11.01791', 'RGI60-11.01806', 'RGI60-11.01813', 'RGI60-11.01840', 
                  'RGI60-11.01857', 'RGI60-11.01894', 'RGI60-11.01928', 'RGI60-11.01962', 'RGI60-11.01986', 
                  'RGI60-11.02006', 'RGI60-11.02024', 'RGI60-11.02027', 'RGI60-11.02244', 'RGI60-11.02249', 
                  'RGI60-11.02261', 'RGI60-11.02448', 'RGI60-11.02490', 'RGI60-11.02507', 'RGI60-11.02549', 
                  'RGI60-11.02558', 'RGI60-11.02583', 'RGI60-11.02584', 'RGI60-11.02596', 'RGI60-11.02600', 
                  'RGI60-11.02624', 'RGI60-11.02673', 'RGI60-11.02679', 'RGI60-11.02704', 'RGI60-11.02709', 
                  'RGI60-11.02715', 'RGI60-11.02740', 'RGI60-11.02745', 'RGI60-11.02755', 'RGI60-11.02774', 
                  'RGI60-11.02775', 'RGI60-11.02787', 'RGI60-11.02796', 'RGI60-11.02864', 'RGI60-11.02884', 
                  'RGI60-11.02890', 'RGI60-11.02909', 'RGI60-11.03249']

    if args.burned:
        list_of_glaciers = [gl + '.tif' for gl in RGI_burned]
        print(f"We burn: {len(list_of_glaciers)} images.")
        if os.path.isfile(args.image + list_of_glaciers[0]) is True:
            pass
        else:
            print("Oops! Burned glaciers not found. Check input folder and try again. Bye bye")
            exit()

    elif args.all:
        list_of_glaciers = os.listdir(args.image)
        print(f"We process: {len(os.listdir(args.image))} images and {len(os.listdir(args.mask))} masks.")

    else:
        print("Invalid option, choices: --burned true and/or --all true")

    # --------------------------------------------------------------------------------------- #
    #                                 Main Inference loop                                     #
    # --------------------------------------------------------------------------------------- #
    # for csv
    names = []
    d1 = [] # size of mask in %
    d2 = [] # size of full mask in %
    means = []
    negatives = []
    areas = []
    volumes = []

    # MAIN LOOP OVER ALL GLACIERS
    for imgfile in tqdm(list_of_glaciers):
        tqdm.write(imgfile)

        # import dem
        dem = rioxarray.open_rasterio(args.image+imgfile)
        dem_values = dem.values.squeeze().astype(np.float32)

        # calculate things I need
        ris_ang = dem.rio.resolution()[0]
        lon_c = (0.5 * (dem.coords['x'][-1] + dem.coords['x'][0])).to_numpy()
        lat_c = (0.5 * (dem.coords['y'][-1] + dem.coords['y'][0])).to_numpy()
        ris_metre_lon = haversine(lon_c, lat_c, lon_c+ris_ang, lat_c) * 1000  # m
        ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c+ ris_ang) * 1000  # m

        # import mask
        mask_rs = rioxarray.open_rasterio(args.mask + imgfile.replace('.tif', '_mask.tif'))
        mask_values = mask_rs.values.squeeze()
        mask_full_rs = rioxarray.open_rasterio(args.fullmask + imgfile.replace('.tif', '_mask_full.tif'))
        mask_full_values = mask_full_rs.values.squeeze()
        # NB the mask should be the full mask to account for all other neighbouring glaciers
        mask = torch.from_numpy(mask_full_values).unsqueeze_(dim=0).to(dtype=torch.float32)

        # calculate slopes
        slope_lat, slope_lon = torch.gradient(torch.from_numpy(dem_values), spacing=[ris_metre_lat, ris_metre_lon], dim=(0, 1))  # (256, 256)

        # normalize to [-1, 1]
        img_max = 9000. # np.amax(dem_values)
        img_min = 0.0 #np.amin(dem_values)
        img = (dem_values - img_min) / (img_max - img_min)
        img = img * 2. - 1.

        img = torch.from_numpy(img).unsqueeze_(dim=0).to(dtype=torch.float32) # (1, 256, 256)

        # normalize slopes
        slope_lat = torch.clip(slope_lat, min=-10., max=10.)
        slope_lon = torch.clip(slope_lon, min=-10., max=10.)
        # slope_lat = (slope_lat-torch.amin(slope_lat))/(torch.amax(slope_lat)-torch.amin(slope_lat))
        # slope_lon = (slope_lon-torch.amin(slope_lon))/(torch.amax(slope_lon)-torch.amin(slope_lon))
        min_slope_lat, max_slope_lat = -10., 10.
        min_slope_lon, max_slope_lon = -10., 10.
        slope_lat = (slope_lat - min_slope_lat) / (max_slope_lat - min_slope_lat)
        slope_lon = (slope_lon - min_slope_lon) / (max_slope_lon - min_slope_lon)
        slope_lat.mul_(2).sub_(1).unsqueeze_(0).unsqueeze_(1) # (1, 1, 256, 256)
        slope_lon.mul_(2).sub_(1).unsqueeze_(0).unsqueeze_(1) # (1, 1, 256, 256)

        show_input_examples = True
        if show_input_examples:
            img_slope_lat = slope_lat[0,0,:,:].numpy()
            img_slope_lon = slope_lon[0,0,:,:].numpy()
            fig, axes = plt.subplots(nrows=1, ncols=3)
            im0 = axes[0].imshow(img[0,:,:].numpy(), cmap='terrain')
            colorbar0 = fig.colorbar(im0, ax=axes[0], shrink=.3)
            axes[0].set_title('DEM')
            im1 = axes[1].imshow(img_slope_lat, cmap='Greys')
            colorbar1 = fig.colorbar(im1, ax=axes[1], shrink=.3)
            axes[1].set_title('Slope Latitude')
            im2 = axes[2].imshow(img_slope_lon, cmap='Greys')
            colorbar2 = fig.colorbar(im2, ax=axes[2], shrink=.3)
            axes[2].set_title('Slope Longitude')

            fig.tight_layout()
            plt.show()


        _, h, w = img.shape
        grid = 8

        # in case the shape is not multiple of 8, we take the closest (//) 8* multiple
        # e.g. if the image is (513, 513), this results in an (512, 512) image
        # we also add one extra dimension at the beginning
        img = img[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0) # (1, 1, 256, 256)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0) # (1, 1, 256, 256)

        img = img.to(device)                # (1, 1, 256, 256)
        slope_lat = slope_lat.to(device)    # (1, 1, 256, 256)
        slope_lon = slope_lon.to(device)    # (1, 1, 256, 256)
        mask = mask.to(device)              # (1, 1, 256, 256)

        # NB QUESTO COMANDO E' IMPORTANTE!
        #img = torch.cat([img[:, 0:1, :, :], slope_lat, slope_lon], axis=1)

        img_masked = img * (1.-mask)  # masked image
        ones_x = torch.ones_like(mask) # (1, 1, 256, 256) this is useless so far
        x = torch.cat([img_masked, slope_lat, slope_lon, ones_x*mask], dim=1)  # (1, 4, 256, 256)
        #x = torch.cat([img_masked, ones_x * mask], dim=1)  # (1, 4, 256, 256)

        with torch.no_grad():
            _, x_stage2 = generator(x, mask) # x_stage2 (1, 1, 256, 256) will have values in [-1, 1]

        # complete image
        image_inpainted = img * (1.-mask) + x_stage2 * mask # (1, 1, 256, 256)
        # denormalize image
        image_denorm = misc.pt_to_image_denorm(image_inpainted, min=0.0, max=9000.).squeeze().cpu().numpy() # (256, 256)

        # ground truth - prediction
        icethick = dem_values - image_denorm # ndarray (256, 256)
        icethick = np.where((mask_values > 0.), icethick, np.nan) # Glacier (central!) ice thickness calculation
        tqdm.write(f"Glacier total thickness: {np.nansum(icethick):.1f} m")

        # stats for csv
        names.append(imgfile)
        d1.append(mask_values.sum()/np.prod(dem_values.shape))
        d2.append(mask_full_values.sum()/np.prod(dem_values.shape))
        means.append(np.nanmean(icethick))
        areas.append(mask_values.sum() * ris_metre_lon * ris_metre_lat * 1e-6) # km2
        volumes.append(np.nansum(icethick) * ris_metre_lon * ris_metre_lat * 1e-9) # km3
        negatives.append(np.nansum(icethick <= 0)/mask_values.sum())
        tqdm.write(f"Glacier bounds: {dem.rio.bounds()}")
        tqdm.write(f"Glacier area: {mask_values.sum() * ris_metre_lon * ris_metre_lat * 1e-6} km2")
        tqdm.write(f"Glacier volume: {np.nansum(icethick) * ris_metre_lon * ris_metre_lat * 1e-9} km3")

        show2d = True
        if show2d:
            mask_to_show = np.where((mask_full_values > 0.), mask_full_values, np.nan)
            fig, axes = plt.subplots(2,3, figsize=(11,6))
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

            N = dem_values.shape[0]
            y_lon_dem = dem_values[int(N / 2), :]
            y_lat_dem = dem_values[:, int(N / 2)]
            y_lon_bed = image_denorm[int(N / 2), :]
            y_lat_bed = image_denorm[:, int(N / 2)]
            y_lon_mask = mask_full_values[int(N / 2), :]  # 1 mask, 0 non mask
            y_lat_mask = mask_full_values[:, int(N / 2)]
            y_lon_mask = np.where(y_lon_mask == 0, np.nan, y_lon_mask)
            y_lat_mask = np.where(y_lat_mask == 0, np.nan, y_lat_mask)

            im1 = ax1.imshow(dem_values, cmap='terrain')
            im1_1 = ax1.imshow(mask_to_show, alpha=.3, cmap='gray')
            im2 = ax2.imshow(image_denorm, cmap='terrain')
            vmin, vmax = abs(np.nanmin(icethick)), abs(np.nanmax(icethick))
            v = max(vmin, vmax)
            im3 = ax3.imshow(icethick, vmin=-v, vmax=v, cmap='bwr_r')
            im4 = ax4.scatter(range(N), y_lon_dem, s=5, c='k', label='dem along lon')
            im4_1 = ax4.plot(range(N), y_lon_bed*y_lon_mask, c='r', label='bed along lon')
            im5 = ax5.scatter(range(N), y_lat_dem, s=5, c='k', label='dem along lat')
            im5_1 = ax5.plot(range(N), y_lat_bed*y_lat_mask, c='r', label='bed along lat')

            ice_farinotti = rioxarray.open_rasterio('/home/nico/PycharmProjects/skynet/Extra_Data/Farinotti/composite_thickness_RGI60-11/RGI60-11/'
                                                    +imgfile.replace('.tif', '_thickness.tif'))
            ice_farinotti = ice_farinotti.values.squeeze()
            im6 = ax6.imshow(ice_farinotti, vmin=-v, vmax=v, cmap='bwr_r')

            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='H (m)')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='H (m)')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='th (m)')
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='th (m)')

            ax4.legend()
            ax5.legend()
            ax1.title.set_text(f'{imgfile[:-4]} - DEM')
            ax2.title.set_text('Inpainted')
            ax3.title.set_text('Ice thickness')
            ax6.title.set_text('Ice thickness Farinotti')

            plt.tight_layout()
            #plt.savefig(args.out + imgfile.replace('.tif', '_res.png'))
            plt.show()
            plt.close()



        # Create Dataset to save
        ds_tosave = xr.Dataset({'dem': dem.squeeze(),
                               'mask': (('y', 'x'), mask_values),
                               'inp': (('y', 'x'), image_denorm),
                               'icethick': (('y', 'x'), icethick)
                               })

        save = False
        if save:
            ds_tosave.rio.to_raster(args.out + imgfile.replace('.tif', '_res.tif'))


    results = pd.DataFrame({'glacier': names, 'd1': d1, 'd2': d2, 'mean (m)': means, 'area (km2)':areas, 'vol (km3)': volumes, 'NP': negatives})
    results.to_csv(args.out+'results.csv', index=False)


if __name__ == '__main__':
    main()

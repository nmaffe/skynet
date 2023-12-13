import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from skimage.segmentation import watershed
from skimage.morphology import local_minima, label
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage import io, color, feature
from skimage import measure

import torch
import torchvision as tv
import torchvision.transforms as T
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM

import Deepfillv2.libs.losses as losses#gan_losses
import Deepfillv2.libs.misc as misc
from Deepfillv2.libs.networks_radio3 import Generator, Discriminator
from Deepfillv2.libs.data import ImageDataset_box, ImageDataset_segmented, get_transforms
from Deepfillv2.libs.losses import *
from Deepfillv2.libs.custom_metrics import RMSE_MAE

parser = argparse.ArgumentParser()
mask_modes = ["box", "segmented"]
parser.add_argument('--config', type=str,default="Deepfillv2/configs/train.yaml", help="Path to yaml config file")
parser.add_argument('--mask', type=str, default="box", help="mask used for training (box, segmented, otsu)")


def training_loop(generator,        # generator network
                  discriminator,    # discriminator network
                  g_optimizer,      # generator optimizer
                  d_optimizer,      # discriminator optimizer
                  gan_loss_g,       # generator gan loss function
                  gan_loss_d,       # discriminator gan loss function
                  ae_loss,          # autoencoder loss
                  train_dataloader, # training dataloader
                  val_dataloader,   # val dataloader
                  last_n_iter,      # last iteration
                  writer,           # tensorboard writer
                  config,            # Config object
                  args
                  ):

    # visto che non abbiamo passato il devide alla funzione training_loop qui lo ridefinisco
    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')

    losses, losses_val = {}, {} # at each training loop its values will be overwritten
    metrics, metrics_val = {}, {}

    # initialize dict for logging
    losses_log = {'d_loss': [],
                  'g_loss': [],
                  'g_loss_adv': [],
                  'ae_loss': [],
                  'ae_loss1': [],
                  'ae_loss2': [],
                  'scaling_loss': [],
                  }
    losses_log_val = {'d_loss': [],
                  'g_loss': [],
                  'g_loss_adv': [],
                  'ae_loss': [],
                  'ae_loss1': [],
                  'ae_loss2': [],
                  'scaling_loss': [],
                  }

    metrics_log = {'ssim': [], 'psnr': [], 'rmse': [], 'mae': []}
    metrics_log_val = {'ssim': [], 'psnr': [], 'rmse': [], 'mae': []}

    # training loop
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader) # initialize the iterator over the dataloader
    val_iter = iter(val_dataloader)
    time0 = time.time()
    print('-'*50)
    for n_iter in tqdm(range(init_n_iter, config.max_iters), leave=True, desc='TRAINING'):

        # print(f"iter: {n_iter}/{config.max_iters}")

        # --------------------------------------------------------------------------------------- #
        #                                  Training loop                                          #
        # --------------------------------------------------------------------------------------- #
        generator.train()
        discriminator.train()
        # load batch of raw data and masks
        while True:
            try:
                if args.mask == "box":
                    batch_real, slope_lat, slope_lon, batch_mins, batch_maxs, batch_ris_lon, batch_ris_lat, batch_bounds = next(train_iter) # fetch batch_real=(N, 3, 256, 256)
                    mask = misc.create_box_brush_mask(config).to(torch.float32)  # (1,1,256,256)
                elif args.mask == "segmented":
                    batch_real, mask = next(train_iter)
                    mask = mask.to(device).to(torch.float32)
                elif args.mask == "otsu":
                    batch_real, slope_lat, slope_lon, batch_mins, batch_maxs, \
                                batch_ris_lon, batch_ris_lat, batch_bounds = next(train_iter)
                    masks = np.ones_like(batch_real) # (N, 1, 256, 256)
                    for j, dem in enumerate(batch_real.numpy().squeeze()):
                        #print(f"train {j} Min {np.min(dem)} Max {np.max(dem)}")
                        thresh = threshold_otsu(dem)
                        binary_otsu = dem <= thresh
                        #print('train', j, thresh, np.sum(binary_otsu) / (256 ** 2))
                        while np.sum(binary_otsu)/(256 ** 2)>0.5:  # reduce threshold if mask is too big
                            thresh = thresh - 0.01
                            binary_otsu = dem <= thresh
                            #print('train', j, thresh, np.sum(binary_otsu)/(256 ** 2))
                            if (thresh<np.min(dem)): # if threshold too small we go for box mask
                                binary_otsu = misc.create_box_brush_mask(config).squeeze().numpy()
                                #print(f"train {j} box {np.sum(binary_otsu)/(256 ** 2)}")
                                continue # exit while loop
                        masks[j] = np.expand_dims(binary_otsu, axis=0)
                    mask = torch.from_numpy(masks)
                    mask = mask.to(device).to(torch.float32)
                elif args.mask == "minmax":
                    batch_real, slope_lat, slope_lon, batch_mins, batch_maxs, \
                            batch_ris_lon, batch_ris_lat, batch_bounds = next(train_iter)
                    masks = np.ones_like(batch_real)  # (N, 1, 256, 256)
                    for j, dem in enumerate(batch_real.numpy().squeeze()):
                        dem_min, dem_max = np.min(dem), np.max(dem)
                        # Note: rare case in which data preproccesing random crop yields a constant region.
                        # In such case we decide for a box mask.
                        if (np.min(dem)==np.max(dem)):
                            binary_random = misc.create_box_brush_mask(config).squeeze().numpy()
                        # if minimum and maximum of dem are well defined and different
                        else:
                            r1, r2 = np.sort(np.random.uniform(dem_min, dem_max, size=2))
                            binary_random = ((dem >= r1) & (dem <= r2))
                            area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                            while (area_mask > 0.5): # if too big try to decrease it. Works in ca. 80% cases
                                r1, r2 = np.sort(np.random.uniform(dem_min, dem_max, size=2))
                                binary_random = ((dem >= r1) & (dem <= r2))
                                area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                            if (area_mask < 0.05): # if decreased result is too small just move to box mask (20% cases)
                                binary_random = misc.create_box_brush_mask(config).squeeze().numpy()
                                area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        #ax1.imshow(dem, cmap='terrain')
                        #ax2.imshow(binary_random)
                        #ax3.hist(dem.flatten(), bins=np.arange(np.min(dem), np.max(dem), 0.01))
                        #plt.show()
                        masks[j] = np.expand_dims(binary_random, axis=0)
                    mask = torch.from_numpy(masks)
                    mask = mask.to(device).to(torch.float32)
                break
            except StopIteration:
                train_iter = iter(train_dataloader)

        show_input_examples = False
        if show_input_examples:
            img = batch_real[0,0].numpy()
            img_slope_lat = slope_lat[0,0].numpy()
            img_slope_lon = slope_lon[0,0].numpy()

            fig, axes = plt.subplots(nrows=1, ncols=3)
            im0 = axes[0].imshow(img, cmap='terrain')
            colorbar0 = fig.colorbar(im0, ax=axes[0], shrink=.3)
            axes[0].set_title('DEM')
            im1 = axes[1].imshow(img_slope_lat, cmap='terrain')
            colorbar1 = fig.colorbar(im1, ax=axes[1], shrink=.3)
            axes[1].set_title('Slope Latitude')
            im2 = axes[2].imshow(img_slope_lon, cmap='terrain')
            colorbar2 = fig.colorbar(im2, ax=axes[2], shrink=.3)
            axes[2].set_title('Slope Longitude')

            fig.tight_layout()
            plt.show()

        batch_real = batch_real.to(device)  # (N,3,256,256)
        slope_lat = slope_lat.to(device)    # (N,1,256,256)
        slope_lon = slope_lon.to(device)    # (N,1,256,256)
        batch_mins = batch_mins.to(device) # (N,)
        batch_maxs = batch_maxs.to(device) # (N,)
        batch_ris_lon = batch_ris_lon.to(device) # (N,)
        batch_ris_lat = batch_ris_lat.to(device) # (N,)
        #print('mask:', mask.shape)

        # NB QUESTO COMANDO E' IMPORTANTE!
        # batch_real = torch.cat([batch_real[:,0:1,:,:], slope_lat, slope_lon], axis=1)

        # prepare input for generator
        batch_incomplete = batch_real * (1. - mask) # (N,3,256,256)
        #batch_incomplete = torch.cat([batch_real[:,0:1,:,:], slope_lat, slope_lon], axis=1) * (1. - mask) # (N,3,256,256)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(device) # (N,1,256,256)

        x = torch.cat([batch_incomplete, slope_lat, slope_lon, ones_x*mask], axis=1)      # (N,6,256,256)
        #x = torch.cat([batch_incomplete, ones_x*mask], axis=1)      # (N,4,256,256)
        #x = torch.cat([batch_incomplete], axis=1)      # (N,4,256,256)
        #print('x: ', x.shape)

        # generate inpainted images
        x1, x2 = generator(x, mask)     # sia x1 che x2 sono (N,3,256,256)
        batch_predicted = x2            # this is the output of the fine generator
        #print('x1', x1.shape)
        #print('x2', x2.shape)
        #input('wait after generator created x1, x2')

        check_x2 = False
        if (check_x2 and n_iter%500 == 0):
            x_inspect = x.cpu().numpy()
            x2_inspect = x2.detach().cpu().numpy()
            fig, axes = plt.subplots(nrows=1, ncols=2)
            im0 = axes[0].imshow(x_inspect[0, 0, :, :], cmap='terrain')
            colorbar0 = fig.colorbar(im0, ax=axes[0], shrink=.3)
            axes[0].set_title('DEM')
            im1 = axes[1].imshow(x2_inspect[0, 0, :, :], cmap='terrain')
            colorbar1 = fig.colorbar(im1, ax=axes[1], shrink=.3)
            axes[1].set_title('Reconstructed DEM')
            #im2 = axes[2].imshow(x2_inspect[0, 2, :, :], cmap='terrain')
            #colorbar2 = fig.colorbar(im2, ax=axes[2], shrink=.3)
            #axes[2].set_title('Slope Longitude')
            #im3 = axes[3].imshow(x_inspect[0, 0, :, :], cmap='terrain')
            #colorbar3 = fig.colorbar(im3, ax=axes[3], shrink=.3)
            #axes[3].set_title('input')

            fig.tight_layout()
            plt.show()

        # use the fine generator prediction inside the mask while keeping the original image elsewhere
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask) # (N,3,256,256)

        # D training steps:
        #batch_real_mask = torch.cat((batch_real, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N,4,256,256)
        #batch_filled_mask = torch.cat((batch_complete.detach(), torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N,4,256,256)
        batch_real_mask = torch.cat((batch_real, mask), dim=1) # (N,4,256,256)
        batch_filled_mask = torch.cat((batch_complete.detach(), mask), dim=1) # (N,4,256,256)
        # oss: batch_filled_mask e batch_real_filled avranno requires_grad=False, quindi saranno staccati dal graph. Perche ?
        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask), dim=0) # (2*N,4,256,256)
        # we apply the discriminator to the whole batch containing both real and generated (and completed) images
        d_real_gen = discriminator(batch_real_filled) # (2*N, 4096)
        # we extract the separate outputs for the real/generated images
        d_real, d_gen = torch.split(d_real_gen, config.batch_size) # (N, 4096), # (N, 4096)
        # todo: perche non posso direttamente calcolare d_real = discriminator(batch_real_mask), d_gen = discriminator(batch_filled_mask) ?
        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # update D parameters
        d_optimizer.zero_grad()
        losses['d_loss'].backward()
        d_optimizer.step()

        # G training steps:
        losses['ae_loss1'] = config.ae_loss_alpha * ae_loss(batch_real, x1, penalty=1.0)
        losses['ae_loss2'] = config.ae_loss_alpha * ae_loss(batch_real, x2, penalty=1.0)
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_predicted # perche usare un altra variabile quando avrei sia batch_predicted che x2 ?
        batch_gen = torch.cat((batch_gen, mask), dim=1) # (N, 4, 256, 256)
        #batch_gen = torch.cat((batch_gen, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N, 4, 256, 256)

        # apply the discriminator to the generated (not completed) images
        d_gen = discriminator(batch_gen) # (N, 4096)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss_adv'] = g_loss
        losses['scaling_loss'] = config.power_law_alpha * loss_power_law(dem=batch_real[:, 0, :, :],
                                                                    bed=batch_complete[:, 0, :, :],
                                                                    mask=mask[:, 0, :, :],
                                                                    c=config.power_law_c,
                                                                    gamma=config.power_law_gamma,
                                                                    mins=batch_mins, maxs=batch_maxs, ris_lon=batch_ris_lon, ris_lat=batch_ris_lat)
        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']
        if config.power_law_loss:
            losses['g_loss'] += losses['scaling_loss']

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()

        # calculate similarity metrics
        ssim = SSIM(batch_real, batch_predicted).detach()
        psnr = PSNR(batch_real, batch_predicted).detach()
        rmse, mae = RMSE_MAE(batch_real, batch_complete, mins=batch_mins, maxs=batch_maxs, mask=mask)
        metrics['ssim'] = ssim
        metrics['psnr'] = psnr
        metrics['rmse'] = rmse
        metrics['mae'] = mae

        # LOGGING TRAIN LOSSES AND METRICS
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())

        for k in metrics_log.keys():
            metrics_log[k].append(metrics[k].item())

        # --------------------------------------------------------------------------------------- #
        #                                  Validation loop                                        #
        #       (Note that some of the objects of the training loop will be overwritten)          #
        # --------------------------------------------------------------------------------------- #
        generator.eval()
        discriminator.eval()

        # load batch of raw data and masks
        while True:
            try:
                if args.mask == "box":
                    batch_real_val, slope_lat_val, slope_lon_val, batch_mins_val, batch_maxs_val, batch_ris_lon_val, batch_ris_lat_val, batch_bounds_val = next(val_iter)  # fetch batch_real=(batch_size, 3, 256, 256)
                    mask_val = misc.create_box_brush_mask(config).to(torch.float32)  # (1,1,256,256)

                elif args.mask == "segmented":
                    batch_real_val, mask_val = next(val_iter)
                    mask_val = mask_val.to(device).to(torch.float32)

                elif args.mask == "otsu":
                    batch_real_val, slope_lat_val, slope_lon_val, batch_mins_val, batch_maxs_val, batch_ris_lon_val, \
                            batch_ris_lat_val, batch_bounds_val = next(val_iter)  # batch_real_val (N, 3, 256, 256)
                    masks_val = np.ones_like(batch_real_val)
                    for j, dem in enumerate(batch_real_val.numpy().squeeze()):
                        thresh = threshold_otsu(dem)
                        binary_otsu = dem <= thresh
                        while np.sum(binary_otsu)/(256 ** 2)>0.5:
                            thresh = thresh - 0.01
                            binary_otsu = dem <= thresh
                            if (thresh < np.min(dem)):  # if threshold too small we go for box mask
                                binary_otsu = misc.create_box_brush_mask(config).squeeze().numpy()
                                continue  # exit while loop
                        masks_val[j] = np.expand_dims(binary_otsu, axis=0)
                    mask_val = torch.from_numpy(masks_val)
                    mask_val = mask_val.to(device).to(torch.float32)

                elif args.mask == "minmax":
                    batch_real_val, slope_lat_val, slope_lon_val, batch_mins_val, batch_maxs_val, batch_ris_lon_val, \
                            batch_ris_lat_val, batch_bounds_val = next(val_iter)  # batch_real_val (N, 3, 256, 256)
                    masks_val = np.ones_like(batch_real_val)
                    for j, dem in enumerate(batch_real_val.numpy().squeeze()):
                        dem_min, dem_max = np.min(dem), np.max(dem)
                        if (np.min(dem) == np.max(dem)): # Box mask
                            binary_random = misc.create_box_brush_mask(config).squeeze().numpy()
                        else: # Try minmax mask
                            r1, r2 = np.sort(np.random.uniform(dem_min, dem_max, size=2))
                            binary_random = ((dem >= r1) & (dem <= r2))
                            area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                            while (area_mask > 0.5):
                                r1, r2 = np.sort(np.random.uniform(dem_min, dem_max, size=2))
                                binary_random = ((dem >= r1) & (dem <= r2))
                                area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                            if (area_mask < 0.05): # Box mask
                                binary_random = misc.create_box_brush_mask(config).squeeze().numpy()
                                area_mask = np.sum(binary_random) / (binary_random.shape[0] * binary_random.shape[1])
                        masks_val[j] = np.expand_dims(binary_random, axis=0)
                    mask_val = torch.from_numpy(masks_val)
                    mask_val = mask_val.to(device).to(torch.float32)
                break
            except StopIteration:
                val_iter = iter(val_dataloader)

        batch_real_val = batch_real_val.to(device)
        slope_lat_val = slope_lat_val.to(device)
        slope_lon_val = slope_lon_val.to(device)
        batch_mins_val = batch_mins_val.to(device)
        batch_maxs_val = batch_maxs_val.to(device)
        batch_ris_lon_val = batch_ris_lon_val.to(device)
        batch_ris_lat_val = batch_ris_lat_val.to(device)
        #print('batch_real_val:', batch_real_val.shape)

        # NB QUESTO COMANDO E' IMPORTANTE!
        # batch_real_val = torch.cat([batch_real_val[:, 0:1, :, :], slope_lat_val, slope_lon_val], axis=1)

        batch_incomplete_val = batch_real_val * (1. - mask_val)  # (batch_size,3,256,256)
        #batch_incomplete = torch.cat([batch_real_val[:,0:1,:,:], slope_lat_val, slope_lon_val], axis=1) * (1. - mask_val) # (N,3,256,256)
        ones_x = torch.ones_like(batch_incomplete_val)[:, 0:1, :, :].to(device)  # (batch_size,1,256,256)
        #x = torch.cat([batch_incomplete, ones_x, ones_x * mask_val], axis=1)  # (batch_size,5,256,256)
        x = torch.cat([batch_incomplete_val, slope_lat_val, slope_lon_val, ones_x * mask_val], axis=1)  # (batch_size,6,256,256)
        #x = torch.cat([batch_incomplete_val, ones_x * mask_val], axis=1)  # (batch_size,6,256,256)
        #x = torch.cat([batch_incomplete_val], axis=1)  # (batch_size,6,256,256)

        # generate inpainted images
        x1_val, x2_val = generator(x, mask_val)  # sia x1 che x2 sono (batch_size,3,256,256)
        batch_predicted = x2_val  # this is the output of the fine generator

        # use the fine generator prediction inside the mask while keeping the original image elsewhere
        batch_complete_val = batch_predicted * mask_val + batch_incomplete_val * (1. - mask_val)  # (batch_size,3,256,256)

        # D training steps:
        #batch_real_mask = torch.cat((batch_real_val, torch.tile(mask_val, [config.batch_size_val, 1, 1, 1])),
        #                            dim=1)  # (batch_size,4,256,256)
        #batch_filled_mask = torch.cat((batch_complete_val.detach(), torch.tile(mask_val, [config.batch_size_val, 1, 1, 1])),
        #                              dim=1)  # (batch_size,4,256,256)
        batch_real_mask = torch.cat((batch_real_val, mask_val), dim=1)
        batch_filled_mask = torch.cat((batch_complete_val.detach(), mask_val), dim=1)
        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))  # (2*batch_size,4,256,256)

        # we apply the discriminator to the whole batch containing both real and generated (and completed) images
        d_real_gen = discriminator(batch_real_filled)  # (32, 4096)
        # we extract the separate outputs for the real/generated images
        d_real, d_gen = torch.split(d_real_gen, config.batch_size_val)  # (16, 4096), # (16, 4096)

        d_loss = gan_loss_d(d_real, d_gen)
        losses_val['d_loss'] = d_loss

        # G training steps:
        losses_val['ae_loss1'] = config.ae_loss_alpha * ae_loss(batch_real_val, x1_val, penalty=1.0)
        losses_val['ae_loss2'] = config.ae_loss_alpha * ae_loss(batch_real_val, x2_val, penalty=1.0)
        losses_val['ae_loss'] = losses_val['ae_loss1'] + losses_val['ae_loss2']

        batch_gen = batch_predicted  # perche usare un altra variabile quando avrei sia batch_predicted che x2 ?
        #batch_gen = torch.cat((batch_gen, torch.tile(mask_val, [config.batch_size_val, 1, 1, 1])), dim=1)  # (N, 4, 256, 256)
        batch_gen = torch.cat((batch_gen,mask_val),dim=1)

        # apply the discriminator to the generated (not completed) images
        d_gen = discriminator(batch_gen)  # (batch_size, 4096)

        g_loss = gan_loss_g(d_gen)
        losses_val['g_loss'] = g_loss
        losses_val['g_loss'] = config.gan_loss_alpha * losses_val['g_loss']
        losses_val['g_loss_adv'] = g_loss
        losses_val['scaling_loss'] = config.power_law_alpha * loss_power_law(dem=batch_real_val[:, 0, :, :],
                                                                            bed=batch_complete_val[:, 0, :, :],
                                                                            mask=mask_val[:, 0, :, :],
                                                                            c=config.power_law_c,
                                                                            gamma=config.power_law_gamma,
                                                                            mins=batch_mins_val, maxs=batch_maxs_val, ris_lon=batch_ris_lon_val,
                                                                            ris_lat=batch_ris_lat_val)
        if config.ae_loss:
            losses_val['g_loss'] += losses_val['ae_loss']
        if config.power_law_loss:
            losses_val['g_loss'] += losses_val['scaling_loss']

        # calculate similarity metrics
        ssim = SSIM(batch_real_val, batch_complete_val).detach()
        psnr = PSNR(batch_real_val, batch_complete_val).detach()
        rmse, mae = RMSE_MAE(batch_real_val, batch_complete_val, mins=batch_mins_val, maxs=batch_maxs_val, mask=mask_val)
        metrics_val['ssim'] = ssim
        metrics_val['psnr'] = psnr
        metrics_val['rmse'] = rmse
        metrics_val['mae'] = mae

        # LOGGING VAL LOSSES AND METRICS
        for k in losses_log_val.keys():
            losses_log_val[k].append(losses_val[k].item())

        for k in metrics_log_val.keys():
            metrics_log_val[k].append(metrics_val[k].item())

        # --------------------------------------------------------------------------------------- #
        #                  Write to console, tensorboard, saving model                            #
        # --------------------------------------------------------------------------------------- #
        # (tensorboard) logging
        if (n_iter%config.print_iter==0):
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()

            # write loss terms to console and tensorboard
            for (k1, loss_log1), (k2, loss_log2) in zip(losses_log.items(), losses_log_val.items()):
                loss_log_mean_train = sum(loss_log1) / len(loss_log1)  # mean of the loss of previous print_iter (default=100) iterations
                loss_log_mean_val = sum(loss_log2) / len(loss_log2)  # mean of the loss of previous print_iter (default=100) iterations
                print(f"Train {k1}: {loss_log_mean_train:.4f} | Val {k2}: {loss_log_mean_val:.4f}")
                if config.tb_logging:
                    writer.add_scalar(f"tr_losses/{k1}", loss_log_mean_train, global_step=n_iter)
                    writer.add_scalar(f"val_losses/{k2}", loss_log_mean_val, global_step=n_iter)
                losses_log[k1].clear()  # every print_iter (default=100) iterations I clean all losses_log values.
                losses_log_val[k2].clear()  # every print_iter (default=100) iterations I clean all losses_log values.

            # write metrics terms to console and tensorboard
            for (k1, metric_log), (k2, metric_log_val) in zip(metrics_log.items(), metrics_log_val.items()):
                metric_log_mean_train = sum(metric_log)/len(metric_log)
                metric_log_mean_val = sum(metric_log_val)/len(metric_log_val)
                print(f"Train {k1}: {metric_log_mean_train:.4f} | Val {k2}: {metric_log_mean_val:.4f}")
                if config.tb_logging:
                    writer.add_scalar(f"tr_metrics/{k1}", metric_log_mean_train, global_step=n_iter)
                    writer.add_scalar(f"val_metrics/{k2}", metric_log_mean_val, global_step=n_iter)
                metrics_log[k1].clear()
                metrics_log_val[k2].clear()


        # save example (train) image grids to tensorboard
        if (config.tb_logging and config.save_imgs_to_tb_iter and n_iter%config.save_imgs_to_tb_iter==0):
            # ognuno dei 3 tensori di viz_images has shape (batch_size, 3, 256, 256)
            #todo: this code to save images to tensorboard needs fixing
            #viz_images = [misc.pt_to_image(batch_complete), misc.pt_to_image(x1), misc.pt_to_image(x2)]
            viz_images = [misc.pt_to_image_denorm(batch_complete, min=batch_mins, max=batch_maxs),
                          misc.pt_to_image_denorm(x1, min=batch_mins, max=batch_maxs),
                          misc.pt_to_image_denorm(x2, min=batch_mins, max=batch_maxs)]
            # ognuno dei 3 tensori di img_grids has shape (3, 1292, 518)
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            #misc.show_grid(img_grids)
            writer.add_image("Inpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 1", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 2", img_grids[2], global_step=n_iter, dataformats="CHW")

        # save example (train) image grids to disk
        if (config.save_imgs_to_dics_iter and n_iter%config.save_imgs_to_dics_iter==0):
            #viz_images = [misc.pt_to_image(batch_real), misc.pt_to_image(batch_complete)]
            viz_images = [misc.pt_to_image_denorm(batch_real, min=batch_mins, max=batch_maxs).cpu(),
                          misc.pt_to_image_denorm(batch_complete, min=batch_mins, max=batch_maxs).cpu()]
            # img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            # tv.utils.save_image(img_grids, f"{config.checkpoint_dir}/images/iter_{n_iter}.png", nrow=2)

            # Extract some denormalized images we want to save
            viz_images_real = viz_images[0][:config.viz_max_out,:,:,:].numpy() # [:config.viz_max_out, 256, 256]
            viz_images_compl = viz_images[1][:config.viz_max_out,:,:,:].numpy() # [:config.viz_max_out, 256, 256]

            # Save them
            fig, axes = plt.subplots(nrows=config.viz_max_out//2, ncols=4, sharex=True, sharey=True, figsize=(6,7.5))
            nrows = config.viz_max_out//2  #=5
            for i in range(2*nrows):
                real = viz_images_real[i,0,:,:]
                compl = viz_images_compl[i,0,:,:]
                mask_region = mask[i,0,:,:].cpu().numpy()
                mask_contours = measure.find_contours(mask_region, 0.0) # Find contours at a constant value of 0
                vmin, vmax = min(real.min(), compl.min()), max(real.max(), compl.max())
                #print(i, i%5, 2*(i//5))
                imreal = axes[i%nrows, 2*(i//nrows)].imshow(real, vmin=vmin, vmax=vmax, cmap='terrain')
                #print(i, i%5, 2*(i//5)+1)
                imcompl = axes[i%nrows, 2*(i//nrows)+1].imshow(compl, vmin=vmin, vmax=vmax, cmap='terrain')
                for contour in mask_contours:
                    axes[i%nrows, 2*(i//nrows)+1].plot(contour[:, 1], contour[:, 0], c='k', lw=1)
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)
            plt.savefig(f"{config.checkpoint_dir}/images/iter_{n_iter}.png")
            plt.close(fig)
            #plt.show()

        # save model at the (almost) last training iteration
        if (config.save_checkpoint_iter and n_iter%config.save_checkpoint_iter==0 and n_iter>init_n_iter):
            misc.save_states("states.pth", generator, discriminator, g_optimizer, d_optimizer, n_iter, config)

        # save model versions during training
        if (config.save_cp_backup_iter and n_iter%config.save_cp_backup_iter==0 and n_iter>init_n_iter):
            misc.save_states(f"states_{n_iter}.pth", generator, discriminator, g_optimizer, d_optimizer, n_iter, config)


def main():
    args = parser.parse_args()
    config = misc.get_config(args.config) # config e' una class che contiene tutti i valori del file train.yaml


    # set random seed
    if config.random_seed != False:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")


    # transforms
    transforms_train = get_transforms(config, data='train')
    transforms_val = get_transforms(config, data='val')

    # dataloading
    if args.mask in ("box", "otsu", "minmax"):
        train_dataset = ImageDataset_box(config.dataset_train_path,
                                        img_shape=config.img_shapes[:2],
                                        scan_subdirs=config.scan_subdirs,
                                        transforms=transforms_train)
        val_dataset = ImageDataset_box(config.dataset_val_path,
                                         img_shape=config.img_shapes[:2],
                                         scan_subdirs=config.scan_subdirs,
                                         transforms=transforms_val)
    elif args.mask == "segmented":
        train_dataset = ImageDataset_segmented(config.dataset_train_path,
                                        img_shape=config.img_shapes[:2],
                                        random_crop=config.random_crop,
                                        scan_subdirs=config.scan_subdirs,
                                        transforms=transforms)
    else:
        print(f'Invalid mask option: {args.mask}')
        exit()


    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=config.batch_size_val,
                                                   shuffle=False,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')
    print(f'Working with: {device}')

    
    # construct networks
    # NB: se cambio cnum_in del generator devo farlo anche al discriminator credo ! -> No, non credo
    #generator = Generator(cnum_in=6, cnum=48, return_flow=False)
    #discriminator = Discriminator(cnum_in=4, cnum=64)
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in+3, cnum_out=1, cnum=48, return_flow=False)
    #generator = Generator(cnum_in=cnum_in+3, cnum_out=cnum_in, cnum=24, return_flow=False)
    discriminator = Discriminator(cnum_in=2, cnum=64)
    #discriminator = Discriminator(cnum_in=cnum_in + 1, cnum=32)

    # push models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

    # losses
    if config.gan_loss == 'hinge':
        gan_loss_d, gan_loss_g = losses.hinge_loss_d, losses.hinge_loss_g
    elif config.gan_loss == 'ls':
        gan_loss_d, gan_loss_g = losses.ls_loss_d, losses.ls_loss_g
    elif config.gan_loss == 'wasserstein':
        gan_loss_d, gan_loss_g = losses.wasserstein_loss_d, losses.wasserstein_loss_g
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    if config.ae_loss == 'l1': ae_loss = losses.loss_l1
    elif config.ae_loss == 'l1l2': ae_loss = losses.loss_l1_l2
    else: raise NotImplementedError(f"Unsupported loss: {config.ae_loss}")

    # decide weather resume from existing checkpoint or train from skratch
    # if train from skratch last_n_iter will be -1, otherwise we load its value from state_dicts['n_iter']
    last_n_iter = -1
    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore)
        generator.load_state_dict(state_dicts['G'])
        discriminator.load_state_dict(state_dicts['D'])
        if 'G_optim' in state_dicts.keys():
            g_optimizer.load_state_dict(state_dicts['G_optim'])
        if 'D_optim' in state_dicts.keys():
            d_optimizer.load_state_dict(state_dicts['D_optim'])
        if 'n_iter' in state_dicts.keys():
            last_n_iter = state_dicts['n_iter']
        print(f"Loaded models from: {config.model_restore}!")


    # start tensorboard logging
    if config.tb_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)
    else: writer = None


    # start training
    training_loop(generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_loss_g,
                  gan_loss_d,
                  ae_loss,
                  train_dataloader,
                  val_dataloader,
                  last_n_iter,
                  writer,
                  config,
                  args)


if __name__ == '__main__':
    main()

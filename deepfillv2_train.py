import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision as tv
import torchvision.transforms as T
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM

import Deepfillv2.libs.losses as gan_losses
import Deepfillv2.libs.misc as misc
from Deepfillv2.libs.networks import Generator, Discriminator
from Deepfillv2.libs.data import ImageDataset_box, ImageDataset_segmented, get_transforms
from Deepfillv2.libs.losses import *

parser = argparse.ArgumentParser()
mask_modes = ["box", "segmented"]
parser.add_argument('--config', type=str,default="/home/user/Documents/icenet/skynet/Deepfillv2/configs/train.yaml", help="Path to yaml config file")
parser.add_argument('--mask', type=str, default="box", help="mask used for training (box, segmented)")


def training_loop(generator,        # generator network
                  discriminator,    # discriminator network
                  g_optimizer,      # generator optimizer
                  d_optimizer,      # discriminator optimizer
                  gan_loss_g,       # generator gan loss function
                  gan_loss_d,       # discriminator gan loss function
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
          #        'scaling_loss': [],
                  }
    losses_log_val = {'d_loss': [],
                  'g_loss': [],
                  'g_loss_adv': [],
                  'ae_loss': [],
                  'ae_loss1': [],
                  'ae_loss2': [],
         #         'scaling_loss': [],
                  }

#    metrics_log = {'ssim': [], 'psnr': []}
#    metrics_log_val = {'ssim': [], 'psnr': []}
    metrics_log = {'d_real': [], 'd_fake': []}
    metrics_log_val = {'d_real': [], 'd_fake': []}

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
                    bbox = misc.random_bbox(config)  # restituisce valori random (top, left, height, width) di un box quadrato.
                    regular_mask = misc.bbox2mask(config, bbox).to(device)  # (1, 1, 256, 256)
                    irregular_mask = misc.brush_stroke_mask(config).to(device)  # (1,1,256,256)
                    mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)  # (1,1,256,256)
                elif args.mask == "segmented":
                    batch_real, mask = next(train_iter)
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
            #input('wait')

        batch_real = batch_real.to(device)  # (N,3,256,256)
        slope_lat = slope_lat.to(device)    # (N,1,256,256)
        slope_lon = slope_lon.to(device)    # (N,1,256,256)
        batch_mins = batch_mins.to(device) # (N,)
        batch_maxs = batch_maxs.to(device) # (N,)
        batch_ris_lon = batch_ris_lon.to(device) # (N,)
        batch_ris_lat = batch_ris_lat.to(device) # (N,)

        # NB QUESTO COMANDO E' IMPORTANTE!
        #batch_real = torch.cat([batch_real[:,0:1,:,:], slope_lat, slope_lon], axis=1)

        # prepare input for generator
        batch_incomplete = batch_real*(1.-mask) # (N,3,256,256)
        #batch_incomplete = torch.cat([batch_real[:,0:1,:,:], slope_lat, slope_lon], axis=1) * (1. - mask) # (N,3,256,256)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(device) # (N,1,256,256)

        #x = torch.cat([batch_incomplete, slope_lat, slope_lon, ones_x*mask], axis=1)      # (N,6,256,256)
        x = torch.cat([batch_incomplete, ones_x*mask], axis=1)      # (N,4,256,256)

        # generate inpainted images
        x1, x2 = generator(x, mask)     # sia x1 che x2 sono (N,3,256,256)
        batch_predicted = x2            # this is the output of the fine generator

        check_x2 = False
        #print plot every 500
        #if n_iter%500 == 0: check_x2 = True
        if check_x2:
            x_inspect = x.cpu().numpy()
            x2_inspect = x2.detach().cpu().numpy()
            fig, axes = plt.subplots(nrows=1, ncols=4)
            im0 = axes[0].imshow(x2_inspect[0, 0, :, :], cmap='terrain')
            colorbar0 = fig.colorbar(im0, ax=axes[0], shrink=.3)
            axes[0].set_title('DEM')
            im1 = axes[1].imshow(x2_inspect[0, 1, :, :], cmap='terrain')
            colorbar1 = fig.colorbar(im1, ax=axes[1], shrink=.3)
            axes[1].set_title('Slope Latitude')
            im2 = axes[2].imshow(x2_inspect[0, 2, :, :], cmap='terrain')
            colorbar2 = fig.colorbar(im2, ax=axes[2], shrink=.3)
            axes[2].set_title('Slope Longitude')
            im3 = axes[3].imshow(x_inspect[0, 0, :, :], cmap='terrain')
            colorbar3 = fig.colorbar(im3, ax=axes[3], shrink=.3)
            axes[3].set_title('input')

            fig.tight_layout()
            plt.show()

        # use the fine generator prediction inside the mask while keeping the original image elsewhere
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask) # (N,3,256,256)

        # D training steps:
        batch_real_mask = torch.cat((batch_real, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N,4,256,256)
        batch_filled_mask = torch.cat((batch_complete.detach(), torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N,4,256,256)
        # oss: batch_filled_mask e batch_real_filled avranno requires_grad=False, quindi saranno staccati dal graph. Perche ?
        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask)) # (2*N,4,256,256)

        # we apply the discriminator to the whole batch containing both real and generated (and completed) images
        d_real_gen = discriminator(batch_real_filled) # (2*N, 4096)
        # we extract the separate outputs for the real/generated images
        d_real, d_gen = torch.split(d_real_gen, config.batch_size) # (N, 4096), # (N, 4096)
        metrics['d_real'] = torch.mean(d_real)
        metrics['d_fake'] = torch.mean(d_gen)

        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # update D parameters
#        d_optimizer.zero_grad()  test con parte adversarial spenta
#        losses['d_loss'].backward()
#        d_optimizer.step()
#
        # G training steps:
        losses['ae_loss1'] = config.l1_loss_alpha * loss_l1(batch_real, x1, penalty=1.0)
        losses['ae_loss2'] = config.l1_loss_alpha * loss_l1(batch_real, x2, penalty=1.0)
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_predicted # perche usare un altra variabile quando avrei sia batch_predicted che x2 ?
        batch_gen = torch.cat((batch_gen, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (N, 4, 256, 256)

        # apply the discriminator to the generated (not completed) images
        d_gen = discriminator(batch_gen) # (N, 4096)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss_adv'] = g_loss
#        losses['scaling_loss'] = config.power_law_alpha * loss_power_law(dem=batch_real[:, 0, :, :],
                                                                   # bed=x2[:, 0, :, :],
                                                                   # mask=mask[:, 0, :, :],
                                                                   # c=config.power_law_c,
                                                                   # gamma=config.power_law_gamma,
                                                                   # mins=0.0, maxs=9000., ris_lon=batch_ris_lon, ris_lat=batch_ris_lat)
        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']
 #       if config.power_law_loss:
 #           losses['g_loss'] += losses['scaling_loss']

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()

        # calculate similarity metrics
#        ssim = SSIM(batch_real, batch_predicted).detach()
#        psnr = PSNR(batch_real, batch_predicted).detach()
#        metrics['ssim'] = ssim
#        metrics['psnr'] = psnr

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
                    bbox = misc.random_bbox(config)  # restituisce valori random (top, left, height, width) di un box quadrato.
                    regular_mask = misc.bbox2mask(config, bbox).to(device)  # (1, 1, 256, 256)
                    irregular_mask = misc.brush_stroke_mask(config).to(device)  # (1,1,256,256)
                    mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)  # (1,1,256,256)
                elif args.mask == "segmented":
                    batch_real_val, mask = next(val_iter)
                    mask = mask.to(device).to(torch.float32)
                break
            except StopIteration:
                # print(f'Exausted val iterator at it: {n_iter}')
                val_iter = iter(val_dataloader)

        batch_real_val = batch_real_val.to(device)
        slope_lat_val = slope_lat_val.to(device)
        slope_lon_val = slope_lon_val.to(device)
        batch_mins_val = batch_mins_val.to(device)
        batch_maxs_val = batch_maxs_val.to(device)
        batch_ris_lon_val = batch_ris_lon_val.to(device)
        batch_ris_lat_val = batch_ris_lat_val.to(device)

        # NB QUESTO COMANDO E' IMPORTANTE!
        #batch_real_val = torch.cat([batch_real_val[:, 0:1, :, :], slope_lat_val, slope_lon_val], axis=1)

        batch_incomplete = batch_real_val * (1. - mask)  # (batch_size,3,256,256)
        #batch_incomplete = torch.cat([batch_real_val[:,0:1,:,:], slope_lat_val, slope_lon_val], axis=1) * (1. - mask) # (N,3,256,256)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(device)  # (batch_size,1,256,256)
        #x = torch.cat([batch_incomplete, ones_x, ones_x * mask], axis=1)  # (batch_size,5,256,256)
        x = torch.cat([batch_incomplete, ones_x * mask], axis=1)  # (batch_size,6,256,256)

        # generate inpainted images
        x1_val, x2_val = generator(x, mask)  # sia x1 che x2 sono (batch_size,3,256,256)
        batch_predicted = x2_val  # this is the output of the fine generator

        # use the fine generator prediction inside the mask while keeping the original image elsewhere
        batch_complete_val = batch_predicted * mask + batch_incomplete * (1. - mask)  # (batch_size,3,256,256)

        # D training steps:
        batch_real_mask = torch.cat((batch_real_val, torch.tile(mask, [config.batch_size_val, 1, 1, 1])),
                                    dim=1)  # (batch_size,4,256,256)
        batch_filled_mask = torch.cat((batch_complete_val.detach(), torch.tile(mask, [config.batch_size_val, 1, 1, 1])),
                                      dim=1)  # (batch_size,4,256,256)
        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))  # (2*batch_size,4,256,256)

        # we apply the discriminator to the whole batch containing both real and generated (and completed) images
        d_real_gen = discriminator(batch_real_filled)  # (32, 4096)
        # we extract the separate outputs for the real/generated images
        d_real, d_gen = torch.split(d_real_gen, config.batch_size_val)  # (16, 4096), # (16, 4096)
        metrics_val['d_real'] = torch.mean(d_real)
        metrics_val['d_fake'] = torch.mean(d_gen)
        

        d_loss = gan_loss_d(d_real, d_gen)
        losses_val['d_loss'] = d_loss

        # G training steps:
        losses_val['ae_loss1'] = config.l1_loss_alpha * loss_l1(batch_real_val, x1_val, penalty=1.0)
        losses_val['ae_loss2'] = config.l1_loss_alpha * loss_l1(batch_real_val, x2_val, penalty=1.0)
        losses_val['ae_loss'] = losses_val['ae_loss1'] + losses_val['ae_loss2']

        batch_gen = batch_predicted  # perche usare un altra variabile quando avrei sia batch_predicted che x2 ?
        batch_gen = torch.cat((batch_gen, torch.tile(mask, [config.batch_size_val, 1, 1, 1])),
                              dim=1)  # (batch_size, 4, 256, 256)

        # apply the discriminator to the generated (not completed) images
        d_gen = discriminator(batch_gen)  # (batch_size, 4096)

        g_loss = gan_loss_g(d_gen)
        losses_val['g_loss'] = g_loss
        losses_val['g_loss'] = config.gan_loss_alpha * losses_val['g_loss']
        losses_val['g_loss_adv'] = g_loss
     #   losses_val['scaling_loss'] = config.power_law_alpha * loss_power_law(dem=batch_real_val[:, 0, :, :],
     #                                                                       bed=x2_val[:, 0, :, :],
     #                                                                       mask=mask[:, 0, :, :],
     #                                                                       c=config.power_law_c,
     #                                                                       gamma=config.power_law_gamma,
     #                                                                       mins=batch_mins_val, maxs=batch_maxs_val, ris_lon=batch_ris_lon_val,
     #                                                                       ris_lat=batch_ris_lat_val)
        if config.ae_loss:
            losses_val['g_loss'] += losses_val['ae_loss']
     #   if config.power_law_loss:
     #       losses_val['g_loss'] += losses_val['scaling_loss']

        # calculate similarity metrics
#        ssim = SSIM(batch_real_val, batch_predicted).detach()
#        psnr = PSNR(batch_real_val, batch_predicted).detach()
#        metrics_val['ssim'] = ssim
#        metrics_val['psnr'] = psnr

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
            viz_images = [misc.pt_to_image(batch_complete), misc.pt_to_image(x1), misc.pt_to_image(x2)]
            # ognuno dei 3 tensori di img_grids has shape (3, 1292, 518)
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            #misc.show_grid(img_grids)
            writer.add_image("Inpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 1", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 2", img_grids[2], global_step=n_iter, dataformats="CHW")

        # save example (train) image grids to disk
        if (config.save_imgs_to_dics_iter and n_iter%config.save_imgs_to_dics_iter==0):
            viz_images = [misc.pt_to_image(batch_real), misc.pt_to_image(batch_complete)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            tv.utils.save_image(img_grids, f"{config.checkpoint_dir}/images/iter_{n_iter}.png", nrow=2)

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
    if args.mask == "box":
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
    # NB: se cambio cnum_in del generator devo farlo anche al discriminator credo !
    #generator = Generator(cnum_in=4, cnum=48, return_flow=False)
    #discriminator = Discriminator(cnum_in=4, cnum=64)
    # construct networks
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in+2, cnum_out=cnum_in, cnum=48, return_flow=False)
    discriminator = Discriminator(cnum_in=cnum_in+1, cnum=64)

    # push models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))
 #   g_optimizer = torch.optim.SGD(generator.parameters(), lr=config.d_lr,momentum=0.9)
    # losses
    if config.gan_loss == 'hinge':
        gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g
    elif config.gan_loss == 'ls':
        gan_loss_d, gan_loss_g = gan_losses.ls_loss_d, gan_losses.ls_loss_g
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")


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
                  train_dataloader,
                  val_dataloader,
                  last_n_iter,
                  writer,
                  config,
                  args)


if __name__ == '__main__':
    main()

import os

import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch
import torchvision as tv
import torchvision.transforms as T
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM


import Deepfillv2.libs.losses as gan_losses
import Deepfillv2.libs.misc as misc
from Deepfillv2.libs.networks import Generator, Discriminator
from Deepfillv2.libs.data import ImageDataset_box, ImageDataset_segmented

print('pippo')
print('pippo2')
print('pippo3')
print('pippo4')
print('pippo5')

parser = argparse.ArgumentParser()
mask_modes = ["box", "segmented"]
parser.add_argument('--config', type=str,default="Deepfillv2/configs/train.yaml", help="Path to yaml config file")
parser.add_argument('--mask', type=str, default="box", help="mask used for training (box, segmented)")


def training_loop(generator,        # generator network
                  discriminator,    # discriminator network
                  g_optimizer,      # generator optimizer
                  d_optimizer,      # discriminator optimizer
                  gan_loss_g,       # generator gan loss function
                  gan_loss_d,       # discriminator gan loss function
                  train_dataloader, # training dataloader
                  last_n_iter,      # last iteration
                  writer,           # tensorboard writer
                  config,            # Config object
                  args
                  ):


    # visto che non abbiamo passato il devide alla funzione training_loop qui lo ridefinisco
    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')

    losses = {} # at each training loop its values will be overwritten
    metrics = {}

    generator.train() # sets the mode to train
    discriminator.train() # sets the mode to train

    # initialize dict for logging
    losses_log = {'d_loss': [],
                  'g_loss': [],
                  'ae_loss': [],
                  'ae_loss1': [],
                  'ae_loss2': [],
                  }

    metrics_log = {'ssim': [],
                   'psnr': [],}

    # training loop
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)
    time0 = time.time()
    for n_iter in range(init_n_iter, config.max_iters):

        print(f"iter: {n_iter}/{config.max_iters}")

        # load batch of raw data
        try:
            if args.mask == "box":
                batch_real = next(train_iter) # fetch batch_real=(batch_size, 3, 256, 256)
            elif args.mask == "segmented":    
                batch_real, mask = next(train_iter)
        except: # how are errors possibly triggered ?
            if args.mask == "box":
                train_iter = iter(train_dataloader) # why define again an iterator ?
                batch_real = next(train_iter)
            elif args.mask == "segmented":
                train_iter = iter(train_dataloader)
                batch_real, mask = next(train_iter)   

        batch_real = batch_real.to(device)

        if args.mask == "box":

            # create mask
            bbox = misc.random_bbox(config) # restituisce valori random (top, left, height, width) di un box quadrato.
            regular_mask = misc.bbox2mask(config, bbox).to(device) # (1, 1, 256, 256)
            irregular_mask = misc.brush_stroke_mask(config).to(device) # (1,1,256,256)
            mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32) # (1,1,256,256)

            # prepare input for generator
            batch_incomplete = batch_real*(1.-mask)                             # (batch_size,3,256,256)
            ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(device) # (batch_size,1,256,256)
            # TODO usare mappa dello slope al posto di ones_x
            x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)      # (batch_size,5,256,256)

            # generate inpainted images
            x1, x2 = generator(x, mask)     # sia x1 che x2 sono (batch_size,3,256,256)
            batch_predicted = x2            # this is the output of the fine generator

            # use the fine generator prediction inside the mask while keeping the original image elsewhere
            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask) # (batch_size,3,256,256)

            # D training steps:
            batch_real_mask = torch.cat((batch_real, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (batch_size,4,256,256)
            batch_filled_mask = torch.cat((batch_complete.detach(), torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (batch_size,4,256,256)
            # oss: batch_filled_mask e batch_real_filled avranno requires_grad=False, quindi saranno staccati dal graph. Perche ?
            batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask)) # (2*batch_size,4,256,256)

            # we apply the discriminator to the whole batch containing both real and generated (and completed) images
            d_real_gen = discriminator(batch_real_filled) # (32, 4096)
            # we extract the separate outputs for the real/generated images
            d_real, d_gen = torch.split(d_real_gen, config.batch_size) # (16, 4096), # (16, 4096)

            d_loss = gan_loss_d(d_real, d_gen)
            losses['d_loss'] = d_loss

            # update D parameters
            d_optimizer.zero_grad()
            losses['d_loss'].backward()
            d_optimizer.step()

            # G training steps:
            losses['ae_loss1'] = config.l1_loss_alpha * torch.mean((torch.abs(batch_real - x1)))
            losses['ae_loss2'] = config.l1_loss_alpha * torch.mean((torch.abs(batch_real - x2)))
            losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']


            batch_gen = batch_predicted # perche usare un altra variabile quando avrei sia batch_predicted che x2 ?
            batch_gen = torch.cat((batch_gen, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1) # (batch_size, 4, 256, 256)


        elif args.mask == "segmented":
            mask = mask.to(device).to(torch.float32)

            # prepare input for generator
            batch_incomplete = batch_real*(1.-mask)
            ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(device)
            x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)

            # generate inpainted images
            x1, x2 = generator(x, mask)
            batch_predicted = x2

            # apply mask and complete image
            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

            # D training steps:
            batch_real_mask = torch.cat((batch_real, mask), dim=1)#torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1)
            #print(batch_real_mask.cpu().numpy().shape)
            batch_filled_mask = torch.cat((batch_complete.detach(), mask),dim=1)#torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1)
            #print(batch_filled_mask.cpu().numpy().shape)

            batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))
            #print(batch_real_filled.cpu().numpy().shape)

            d_real_gen = discriminator(batch_real_filled)
            d_real, d_gen = torch.split(d_real_gen, config.batch_size)

            d_loss = gan_loss_d(d_real, d_gen)
            losses['d_loss'] = d_loss

            # update D parameters
            d_optimizer.zero_grad()
            losses['d_loss'].backward()
            d_optimizer.step()

            # G training steps:
            losses['ae_loss1'] = config.l1_loss_alpha * torch.mean((torch.abs(batch_real - x1)))
            losses['ae_loss2'] = config.l1_loss_alpha * torch.mean((torch.abs(batch_real - x2)))
            losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

            batch_gen = batch_predicted
            batch_gen = torch.cat((batch_gen, mask), dim=1)#torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1)
            #print(batch_gen.detach().cpu().numpy().shape)


        # apply the discriminator to the generated (not completed) images
        d_gen = discriminator(batch_gen) # (batch_size, 4096)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        #print(losses)
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']

        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()

        # calculate similarity metrics
        ssim = SSIM(batch_real, batch_predicted).detach()
        psnr = PSNR(batch_real, batch_predicted).detach()
        metrics['ssim'] = ssim
        metrics['psnr'] = psnr

        # LOGGING LOSSES AND METRICS
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())

        for k in metrics_log.keys():
            metrics_log[k].append(metrics[k].item())

        # (tensorboard) logging
        if (n_iter%config.print_iter==0):
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()

            # write loss terms to console and tensorboard
            for k, loss_log in losses_log.items():
                loss_log_mean = sum(loss_log)/len(loss_log) # mean of the loss of previous print_iter (default=100) iterations
                print(f"{k}: {loss_log_mean:.4f}")
                if config.tb_logging:
                    writer.add_scalar(f"losses/{k}", loss_log_mean, global_step=n_iter)
                losses_log[k].clear() # every print_iter (default=100) iterations I clean all losses_log values.

            # write metrics terms to console and tensorboard
            for k, metric_log in metrics_log.items():
                metric_log_mean = sum(metric_log)/len(metric_log)
                print(f"{k}: {metric_log_mean:.4f}")
                if config.tb_logging:
                    writer.add_scalar(f"metrics/{k}", metric_log_mean, global_step=n_iter)
                metrics_log[k].clear()



        # save example image grids to tensorboard
        if (config.tb_logging and config.save_imgs_to_tb_iter and n_iter%config.save_imgs_to_tb_iter==0):
            # ognuno dei 3 tensori di viz_images has shape (batch_size, 3, 256, 256)
            viz_images = [misc.pt_to_image(batch_complete), misc.pt_to_image(x1), misc.pt_to_image(x2)]
            # ognuno dei 3 tensori di img_grids has shape (3, 1292, 518)
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            #misc.show_grid(img_grids)
            writer.add_image("Inpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 1", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image("Stage 2", img_grids[2], global_step=n_iter, dataformats="CHW")

        # save example image grids to disk
        if (config.save_imgs_to_dics_iter and n_iter%config.save_imgs_to_dics_iter==0):
            viz_images = [misc.pt_to_image(batch_real), misc.pt_to_image(batch_complete)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2) for images in viz_images]
            tv.utils.save_image(img_grids, f"{config.checkpoint_dir}/images/iter_{n_iter}.png", nrow=2)

        # save model at the (almost) last training iteration
        if (n_iter%config.save_checkpoint_iter==0 and n_iter>init_n_iter):
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
    # ha senso un random horizonal flip ? probabilmente si
    transforms = [T.RandomHorizontalFlip(0.0)] if config.random_horizontal_flip else None


    # dataloading
    if args.mask == "box":
        train_dataset = ImageDataset_box(config.dataset_path,
                                        img_shape=config.img_shapes[:2],
                                        random_crop=config.random_crop,
                                        scan_subdirs=config.scan_subdirs,
                                        transforms=transforms)
    elif args.mask == "segmented":
        train_dataset = ImageDataset_segmented(config.dataset_path,
                                        img_shape=config.img_shapes[:2],
                                        random_crop=config.random_crop,
                                        scan_subdirs=config.scan_subdirs,
                                        transforms=transforms)
    else:
        print("Invalid mask option: {}".format(args.mask))
        exit()


    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')
    print(f'Working with: {device}')

    
    # construct networks
    generator = Generator(cnum_in=5, cnum=48, return_flow=False)
    discriminator = Discriminator(cnum_in=4, cnum=64)

    # push models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

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


    # start training
    training_loop(generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_loss_g,
                  gan_loss_d,
                  train_dataloader,
                  last_n_iter,
                  writer,
                  config,
                  args)


if __name__ == '__main__':
    main()

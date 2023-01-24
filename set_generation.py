import numpy as np
import random
import cv2
import xarray as xr
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt



def create_train_images_small(image, mask, patch_size, max_iter, threshold=None, mode=None, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size


    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    while True:

        for _ in range(max_iter):
            append = True
            rng = random.seed(random_state)
            i_s = random.randint(0, i_h - p_h + 1)
            j_s = random.randint(0, i_w - p_w + 1)
            patch = image[i_s:i_s+p_h, j_s:j_s+p_w]
            mask_patch = mask[i_s:i_s+p_h, j_s:j_s+p_w]

            if invalid_value in patch:
                append = False
            elif np.sum(mask_patch) != 0:
                append = False

            # mode = max:     max of image must be above threshold
            # mode = average: average of image must be above threshold
            elif mode == 'max':
                max = np.max(patch)
                if max < threshold:
                    append = False
            elif mode == 'average':
                avg = np.mean(patch)
                if avg < threshold:
                    append = False

            if append and patch.size == size:
                patches.append(patch)


        if len(patches) > 20:
            break

    return np.array(patches)


def flow_train_images(image, mask, patch_size, max_iter, threshold=None, mode=None, random_state=None, invalid_value=-32767.):
    flow = create_train_images_small(image, mask, patch_size, max_iter, threshold, mode, random_state, invalid_value)
    return flow[..., np.newaxis]


def create_test_images_from_glacier_center(image, mask, patch_size, coords_frame, create_blank=False, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    RGI = []
    masks = []

    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()


    # loop sugli indici dei centri del ghiacciai (r:lat, c:lon)
    for r, c, id in zip(rows, cols, RGIId):

        r = r - int(p_h/2) if r >= int(p_h/2) else r
        c = c - int(p_w/2) if c >= int(p_w/2) else c

        append = True
        patch = image[r:r+p_h, c:c+p_w]
        mask_patch = mask[r:r+p_h, c:c+p_w]


        # extract glacier at center from mask_patch
        # only return the singular glacier mask
        center_value = mask_patch[int(p_h/2), int(p_w/2)]
        # if glacier is very small, look for it in surrounding pixels

        single_mask = np.zeros_like(mask_patch)

        if center_value == 0:
            mid_right = mask_patch[int(p_h/2),int(p_w/2):]
            mid_left  = mask_patch[int(p_h/2),:int(p_w/2)+1][::-1]
            mid_bot   = mask_patch[int(p_h/2):,int(p_w/2)]
            mid_top   = mask_patch[:int(p_h/2)+1,int(p_w/2)][::-1]

            if np.sum(mid_right) != 0 and np.sum(mid_left) != 0 and np.sum(mid_bot) != 0 and np.sum(mid_top) != 0:
                mask_value = np.array([                     # creo array fatto dal primo valore diverso da zero per ognuno delle 4 direzioni
                    mid_right[np.nonzero(mid_right)][0],    # primo valore diverso da zero
                    mid_left[np.nonzero(mid_left)][0],      # primo valore diverso da zero
                    mid_bot[np.nonzero(mid_bot)][0],        # primo valore diverso da zero
                    mid_top[np.nonzero(mid_top)][0]         # primo valore diverso da zero
                ])
                if np.all(mask_value == mask_value[0]):     # se tutti i 4 valori sono identici (e saranno tutti 1)
                    mask_coords = np.nonzero(mask_patch == mask_value[0])   # prendo tutti i pixel con questo valore (1) in tutta la mask_patch
                    single_mask[mask_coords] = 1
                else:
                    if not create_blank:
                        append = False

            else:
                if not create_blank:
                    append = False

        else:
            # center_value e' 1.
            # single_mask e' uguale a mask_patch.
            mask_coords = np.nonzero(mask_patch == center_value)
            single_mask[mask_coords] = 1

        # fig, axs = plt.subplots(3, 1, figsize=(4,10))
        # im0 = axs[0].imshow(patch, cmap='terrain')
        # axs[0].title.set_text('patch')
        # im1 = axs[1].imshow(mask_patch)
        # axs[1].title.set_text('mask_patch')
        # im2 = axs[2].imshow(single_mask)
        # axs[2].title.set_text('single_mask')
        # axs[2].set_xlabel(id)
        # plt.show()


        #    # 3x3
        #    center_matrix = mask_patch[int(p_h/2)-1:int(p_h/2)+2,
        #                               int(p_w/2)-1:int(p_w/2)+2].flatten()
        #    values, counts = np.unique(center_matrix, return_counts=True)
        #    if np.sum(values) != 0:
        #        center_value = center_matrix[np.nonzero(center_matrix)[0][0]]
        #    else:
        #        append = False

        if invalid_value in patch:
            append = False

        # at low resolution, some glaciers are not visible
        # after translation from coords to xy
        elif np.sum(single_mask) == 0:
            #print(f'Glacier {id} empty!!')
            if not create_blank:
                append = False

        if append and patch.size == size:
            patches.append(patch)
            masks.append(single_mask)
            RGI.append(id)

    return np.array(patches), np.array(masks), np.array(RGI)


def create_dataset_train(image, mask, patch_size, max_iter, seen, max_height, threshold=None, mode=None, random_state=None, invalid_value=-32767.):

    """
    This function only creates the train/val images (not the masks).
    """

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size


    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    seen = seen #useful?
    patches = []

    for _ in range(max_iter):
        append = True
        rng = random.seed(random_state)
        # create random indexes
        i_s = random.randint(0, i_h - p_h + 1)
        j_s = random.randint(0, i_w - p_w + 1)
        # calculate the indexes of the center
        center_i = int(i_s + (p_h/2))
        center_j = int(j_s + (p_w/2))
        # calculate the patch and the mask patch
        patch = image[i_s:i_s+p_h, j_s:j_s+p_w] #xarray
        mask_patch = mask[i_s:i_s+p_h, j_s:j_s+p_w] #xarray

        if invalid_value in patch:
            append = False

        # CONDITION 1: presence of glaciers inside the patch - we want to sample only glacier-free regions.
        #if np.sum(mask_patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48]) != 0: # discard if the 96x96 center box is not glacier-free.
        if float(mask_patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48].sum()) != 0:
        #if float(mask_patch.sum()) != 0:  # discard if the patch is not entirely glacier-free.
            append = False

        # CONDITION 2: if this region has already been (partially) sampled
        #if float(np.sum(seen[center_i-15:center_i+16, center_j-15:center_j+16])) > 200: # discard if the 32x32 region around the center has been previously sampled to some extent.
        if float(seen[center_i-15:center_i+16, center_j-15:center_j+16].sum()) > 200: # discard if the 32x32 region around the center has been previously sampled to some extent.
            append = False

        # CONDITION 3: do not exceed a maximum height value
        # modified for other mountain ranges
        if float(patch.max()) > max_height:
            append = False

        # CONDITION 4: the patch should be above a certain threshold (mean or max)
        if mode == 'max':
            #max = np.max(patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48])
            max = float(patch.max())
            if max < threshold:
                append = False
        if mode == 'average':
            #avg = np.mean(patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48])
            avg = float(patch.mean())
            if avg < threshold:
                append = False

        if append and patch.size == size:
            seen[center_i-15:center_i+16, center_j-15:center_j+16] += 1 # To keep track of created patch regions, fill a 32x32 box centered on the patch region with 1.
            patches.append(patch)
            #fig, axs = plt.subplots(2,1)
            #im0 = axs[0].imshow(patch, cmap='terrain')
            #im1 = axs[1].imshow(mask_patch)
            #plt.show()

        if len(patches) == 10:
            break

    return patches, seen


def flow_train_dataset(image, mask, region, patch_size, train_path, val_path, max_height, threshold=None, mode=None, total=15000, postfix='', random_state=None, invalid_value=-32767.):

    seen = xr.zeros_like(image)

    cumulative_patch = np.zeros((1, patch_size[0], patch_size[1])) # ndarray used only for plotting

    num, limit = 0, total
    pbar = tqdm(total = limit)
    while True:
        flow, seen = create_dataset_train(image, mask, patch_size, 50000, seen, max_height, threshold, mode, random_state, invalid_value)

        cumulative_patch = np.append(cumulative_patch, np.array(flow), axis=0)

        if len(flow) < 10:
            print("Unable to find new patches")
            break

        # SAVE THE IMAGES
        # We have a batch of 10 images. We keep all but the last one as train and the last one as val.
        for img in flow[:-1]:
            img.rio.to_raster(train_path + f'RGI_{region}_'+ str(num) + '.tif', dtype=np.uint16)
            num += 1
        flow[-1].rio.to_raster(val_path + f'RGI_{region}_'+ str(num) + '.tif', dtype=np.uint16)
        num += 1

        pbar.update(10)
        if num >= limit:
            break

    pbar.close()

    """ inspect the mean training patch to see if there is any bias """
    average_patch = np.average(cumulative_patch, axis=0)
    print('Shapes: ', cumulative_patch.shape, average_patch.shape)

    fig, axs = plt.subplots()
    im0 = axs.imshow(average_patch, cmap='terrain')
    cax = plt.axes([0.8, 0.12, 0.03, 0.75])
    cbar = fig.colorbar(im0, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('Height (m)', rotation=270, fontsize=13, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    plt.show()

    show = True
    if show:
        """ Show the final maps. Note that the created 256x256 patches are only shown for the inner 32x32 center boxes. """
        fig, (ax0, ax1) = plt.subplots(2,1, figsize=(6, 8))
        im0 = image.rio.clip_box(minx=7.2,miny=45.5,maxx=8.0,maxy=46.3).plot.imshow(ax=ax0, cmap='terrain', cbar_kwargs={'label':'Height (m)'})
        im1 = mask.rio.clip_box(minx=7.2,miny=45.5,maxx=8.0,maxy=46.3).plot.imshow(ax=ax1, cmap='Blues', add_colorbar=False)
        im2 = seen.rio.clip_box(minx=7.2,miny=45.5,maxx=8.0,maxy=46.3).where(seen!=0.0).plot.imshow(ax=ax1, cmap='spring', cbar_kwargs={'label':'Training region'})
        im3 = image.rio.clip_box(minx=7.2,miny=45.5,maxx=8.0,maxy=46.3).plot.imshow(ax=ax1, cmap='terrain', alpha=0.4, add_colorbar=False)
        ax0.set_xlabel('')
        ax0.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax0.set_title('')
        ax1.set_title('')
        ax0.set_xlim(7.2, 8.0)
        ax0.set_ylim(45.5, 46.3)
        ax1.set_xlim(7.2, 8.0)
        ax1.set_ylim(45.5, 46.3)
        plt.show()

    return seen


def create_test_images_full_noedge(image, mask, patch_size, coords_frame, create_blank=False, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    RGI = []
    masks = []

    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()

    for r, c, id in tqdm(zip(rows, cols, RGIId)):
        r = r - int(p_h/2) if r >= int(p_h/2) else r
        c = c - int(p_w/2) if c >= int(p_w/2) else c

        append = True
        patch = image[r:r+p_h, c:c+p_w]
        mask_patch = mask[r:r+p_h, c:c+p_w]

        # extract mask_patch edge
        #edge = [mask_patch[0,:-1], mask_patch[:-1,-1], mask_patch[-1,::-1], mask_patch[-2:0:-1,0]]
        #edge = np.concatenate(edge)

        # remove glaciers bordering the edge of the mask
        #for i in np.unique(edge):
        #    if i != 0:
        #        mask_coords = np.nonzero(mask_patch == i)
        #        mask_patch[mask_coords] = 0

        # convert all mask values from labeled to 1
        mask_patch[mask_patch > 0] = 1

        if invalid_value in patch:
            append = False

        # at low resolution, some glaciers are not visible
        # after translation from coords to xy
        elif np.sum(mask_patch) == 0:
            if not create_blank:
                append = False

        if append and patch.size == size:
            patches.append(patch)
            masks.append(mask_patch)
            RGI.append(id)

    return np.array(patches), np.array(masks), np.array(RGI)

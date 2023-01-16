import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from tqdm import tqdm
import glob

input('WAIT')

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
                    default="examples/inpaint/case1.png", help="path to the image file")
parser.add_argument("--mask", type=str,
                    default="examples/inpaint/case1_mask.png", help="path to the mask file")
parser.add_argument("--out", type=str,
                    default="examples/inpaint/case1_out_test.png", help="path for the output file")
parser.add_argument("--checkpoint", type=str,
                    default="pretrained/states_tf_places2.pth", help="path to the checkpoint file")
parser.add_argument("--tfmodel", action='store_true',
                    default=False, help="use model from models_tf.py?")
parser.add_argument('--burned', default=False, type=str2bool, const=True, 
                    nargs='?', help='Run all burned glaciers')
parser.add_argument('--all',  default=False, type=str2bool, const=True, 
                    nargs='?', help='Run all glaciers')



def pt_to_rgb(pt): return pt[0].cpu().permute(1, 2, 0)*0.5 + 0.5

def main():

    args = parser.parse_args()

    IMG             = args.image
    MASK            = args.mask
    if args.all:
        if IMG[:-1] != '/':
            IMG = ''.join((IMG, '/'))
        if MASK[:-1] != '/':
            MASK = ''.join((MASK, '/'))
        
        OUTDIR          = args.out
        if OUTDIR[:-1] != '/':
            OUTDIR = ''.join((OUTDIR, '/'))

    if args.tfmodel:
        from Deepfillv2.libs.networks_tf import Generator
    else:
        from Deepfillv2.libs.networks import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(args.checkpoint)['G']
    generator.load_state_dict(generator_state_dict)

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
        for i in range(len(RGI_burned)):

            # load image and mask
            img = cv2.imread(args.image + RGI_burned[i] + '.tif', -1)
            img_max = np.amax(img)
            img_min = np.amin(img)
            img = (img - img_min) / (img_max - img_min)
            img = (np.stack((img,)*3, axis=-1) * 255).astype(np.uint8)
            #print(np.mean(img), img.shape)

            image = Image.fromarray((img))#.astype(np.uint8))

            mask = cv2.imread(args.mask + RGI_burned[i] + '_mask.tif', -1)
            #mask_save = mask
            mask = np.stack((mask,)*3, axis=-1).astype(np.uint8)
            mask = Image.fromarray((mask))

            # prepare input
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)

            _, h, w = image.shape
            grid = 8

            image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
            mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

            image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
            mask = (mask > 0.).to(dtype=torch.float32,
                                device=device)  # 1.: masked 0.: unmasked

            image_masked = image * (1.-mask)  # mask image

            ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
            x = torch.cat([image_masked, ones_x, ones_x*mask],
                        dim=1)  # concatenate channels

            with torch.no_grad():
                _, x_stage2 = generator(x, mask)

            # complete image
            image_inpainted = image * (1.-mask) + x_stage2 * mask

            # save inpainted image
            output_denorm = (pt_to_rgb(image_inpainted)[:,:,0].numpy() * (img_max - img_min) + img_min)
            cv2.imwrite(args.out + RGI_burned[i] + '.tif', output_denorm.astype(np.uint16))

            print(f"Saved output file at: {args.out + RGI_burned[i] + '.tif'}")#, with denormalised max: {np.max(output_denorm)}")
    
    elif args.all:

        image_paths, mask_paths = [], []
        for filename in glob.iglob(IMG + '*'):
            image_paths.append(filename)
        for filename in glob.iglob(MASK + '*'):
            mask_paths.append(filename)
        print("# images: {} and # masks: {}".format(len(image_paths), len(mask_paths)))

         # load image and mask
        for i in tqdm(range(len(image_paths))):
            img = cv2.imread(image_paths[i], -1)
            img_max = np.amax(img)
            img_min = np.amin(img)
            img = (img - img_min) / (img_max - img_min)
            img = (np.stack((img,)*3, axis=-1) * 255).astype(np.uint8)

            image = Image.fromarray((img))

            mask = cv2.imread(mask_paths[i], -1)
            mask = np.stack((mask,)*3, axis=-1).astype(np.uint8)
            mask = Image.fromarray((mask))

            # prepare input
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)

            _, h, w = image.shape
            grid = 8

            image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
            mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)


            image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
            mask = (mask > 0.).to(dtype=torch.float32,
                                device=device)  # 1.: masked 0.: unmasked

            image_masked = image * (1.-mask)  # mask image

            ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
            x = torch.cat([image_masked, ones_x, ones_x*mask],
                        dim=1)  # concatenate channels

            with torch.no_grad():
                _, x_stage2 = generator(x, mask)

            # complete image
            image_inpainted = image * (1.-mask) + x_stage2 * mask
            output_denorm = (pt_to_rgb(image_inpainted)[:,:,0].numpy() * (img_max - img_min) + img_min)
            cv2.imwrite(OUTDIR + image_paths[i][-18:], output_denorm.squeeze().astype(np.uint16))
    
    else:
        print("Invalid option, choices: --burned true and/or --all true")




if __name__ == '__main__':
    main()

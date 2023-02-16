# Estimate the ice volume of Earth's glaciers with deep learning

<!--
<p align="center">
  <img
  src="images/logo.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px"
  width="300"
  >
</p>
-->

## Download dem tiles ️🛰 ⛰️
This model uses ASTER GDEM v3. 
Download it from https://search.earthdata.nasa.gov/search. 
 To select the region of interest you may specify the rectancle SW and NE coordinates:

| RGI | SW (lat, lon) | NE (lat, lon)|
| :---:  | :---: | :---: |
| 08 (Scandinavia)  | 58, 4 | 72, 35 |
| 11 (Central Europe)  | 41, -2 | 49, 21 |
| 13, 15, 15 (Asia) | 26, 66 | 47, 105 |
| 18 (New Zealand) | -47, 166 | -38, 177 |

## Create mosaic 🗺️
To create the DEM mosaic and the mosaic_mask where of all glaciers contained inside the region, run:
```
python create_mosaic.py --input /PATH_TO_DEM_TILES/ --output /PATH_TO_OUTPUT_MOSAIC/ --create_mask True --version '62' --epsg "EPSG:4326" --region None

--create_mask (default=True): if you want to produce the mosaic_mask of all glaciers
--epsg (default="EPSG:4326" the DEM projection).
--version (default='62'): oggm version to extract glaciers.
--region (default=None): RGI region as XX.
```
This code creates two files: ```mosaic_RGI_xx.tif``` and ```mosaic_RGI_xx_mask.tif``` in the specified output path.
The mask file contains 1 if the pixel belongs to a glacier (segmented using Bresenham’s line algorithm), and 0 otherwise.
The glacier shapefiles are extracted from ```oggm``` library.

## Create test dataset 🕵🏿
This code creates the test dataset of all glaciers contained in the mosaic. It consists of three folders of .tif files: 
- images/: glacier DEM patch
- masks/: glacier mask 
- masks_full/: mask of ALL glaciers inside the patch

```
python create_test.py --input PATH --outdir PATH --region None --shape 256 --version '62' --epsg "EPSG:4326"
 
--input PATH: path of mosaic and mosaic_mask files
--outdir PATH: path for the generated test dataset
--region (default=None): RGI region as XX
--version (default='62'): oggm version to extract glaciers.
--shape (default=256): size of test patches
--epsg (default="EPSG:4326" the DEM projection)
```
The masks are always centered in the image, therefore the inpainting will be always done in the center.

⚠️ Warning1: if we want to inpaint the glacier to get its bedrock, we should apply the model to the full mask, since the
presence of all the surrounding glaciers should be accounted for.  

⚠️ Warning2: The test images are currently built as fixed 256x256 images. If a glacier is bigger in any of the two 
dimensions, it is discarded. A solution could be to keep them as their original shape and scaling as data transformation
to the same size before the forward pass. Or to increase to 512x512. Currently, roughly up to 3% of glaciers are discarded.

## Create train dataset 🏋️
This code creates the training dataset. 
```
python create_train.py --input PATH --outdir PATH --region None --threshold 1500

--input PATH: input dem mosaic .tif file
--outdir PATH (default='dataset/'): path for the output files
--region (default=None): RGI region as XX
--shape (default=256): size of train patches
--version (default='62'): oggm version
--epsg (default="EPSG:4326" the DEM projection)
--max_height (default=9999): max desired height of training samples
--threshold (default=2000): Threshold value to sample high elevation regions
--mode (default='average'): Threshold mode: average or max
--samples (default=4000): Number of samples to attempt to create
--postfix (default='a'): postfix added behind sample files
```

Only the dem patches are created; the masks are created on-the-fly (see Deepfillv2 code).
We randomly create patches from inside the mosaic. However, the created patch is kept if all the following conditions are met: 
- If there is an ```invalid_value``` within the patch
- If the central 96x96 patch contains no glacier pixels. Note that, however, glaciers may be inside the patch but outside
the inner 96x96 box.
- If the central 32x32 box has not been previously sampled (however some degree of overlap is allowed).
- If the maximum patch value does not exceed ```max_height```
- If the ```threshold```/```mode``` condition is met.

Of all the created images, 90%-10% are saved as train-validation. The algorithm stop if no new patches can be created or
the total amount of ```samples``` is reached.

## Train
```
python deepfillv2_train.py --config CONFIGFILE.yaml --mask MASKTYPE

--config (default="Deepfillv2/configs/train.yaml"): config .yaml file
--mask (default="box"): box or segmented mask type
```
The .yaml file contains all the configuration parameters. In particular, set the "dataset_path" variable to 
indicate the training dataset you want and "checkpoint_dir" to indicate where the model will be saved.

## Test
```
 python deepfillv2_test.py --image PATH --mask PATH --fullmask PATH --out PATH --checkpoint PTHFILE --tfmodel False --all False --burned False

--image: input folder with image files
--mask: input folder with mask files
--fullmask: input folder with full mask files
--out: path to saved results 
--checkpoint: path to the checkpoint file
--tfmodel (default=False): use model from models_tf.py?
--all (default=False): run all glaciers in input folder
--burned (default=False): run all burned glaciers in input folder 
 ```
If ```--all``` is True, the code inpaints all images contained in ```--image``` using the masks contained in ```--mask```.
The results are saved as ```.tif``` files in the ```--out``` folder. # TODO: give option to use either mask or full masks.

If ```--burned``` is True, the code only inpaints 68 glaciers of RGI11. In this case the relevant input paths
(```image/mask/fullmask```) should necessarily be those that contain the relevant RGI11 files. 

## Acknowledgments

[<img align="left" alt="CCAI" src="img/logo_CCAI.png" height="70" />](https://www.climatechange.ai/)
[<img aligh="right" alt="EU" src="img/logo_MSCA.png" height="70" />](https://marie-sklodowska-curie-actions.ec.europa.eu/)

This project was funded by the Climate Change AI Innovation Grants program,
hosted by Climate Change AI with the additional support of Canada Hub of Future
Earth.

The project has also received funding from the European Union’s Horizon Europe research 
and innovation programme under the Marie Skłodowska-Curie program.
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

## Download dem tiles
This model uses ASTER GDEM v3. 
Download it from https://search.earthdata.nasa.gov/search. 
 To select the region of interest you may specify the rectancle SW and NE coordinates:

| RGI | SW (lat, lon) | NE (lat, lon)|
| :---:  | :---: | :---: |
| 08   | 58, 4 | 72, 35 |
| 11   | 41, -2 | 49, 21 |
| 13, 15, 15 | 26, 66 | 47, 105 |
| 18   | -47, 166 | -38, 177 |

## Create mosaic
To create the DEM mosaic and the mosaic_mask where of all glaciers contained inside the region, run:
```
python create_mosaic.py --input '/PATH_TO_DEM_TILES/' --output /PATH_TO_OUTPUT_MOSAIC/ --create_mask True --version '62' --epsg "EPSG:4326" --region None

--create_mask (default=True): if you want to produce the mosaic_mask of all glaciers
--epsg (default="EPSG:4326" the DEM projection).
--version (default='62'): oggm version to extract glaciers.
--region (default=None): specify Randolph Glacier Inventory region as XX.
```
This code creates two files: ```mosaic_RGI_xx.tif``` and ```mosaic_RGI_xx_mask.tif``` in the specified output path.
The mask file contains 1 if the pixel belongs to a glacier (segmented using Bresenham’s line algorithm), and 0 otherwise.
The glacier shapefiles are extracted from ```oggm``` library.




### Funding
<!--
[<img aligh="right" alt="CCAI" src="images/logoCCAI.png" height="70">](https://www.climatechange.ai/)
[<img aligh="right" alt="EU" src="images/logoEU.png" height="70">](https://marie-sklodowska-curie-actions.ec.europa.eu/)
-->

This project was funded by the Climate Change AI Innovation Grants program,
hosted by Climate Change AI with the additional support of Canada Hub of Future
Earth.

The project has also received funding from the European Union’s Horizon Europe research 
and innovation programme under the Marie Skłodowska-Curie program.
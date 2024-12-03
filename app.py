from arkitekt_next import progress, register
import time
from mikro_next.api.schema import Image, from_array_like, PartialPixelViewInput, RangePixelLabel
from cellpose import models
from typing import Annotated
import xarray as xr
model_type = "cyto3"


class ModelType:
    CYTO3 = "cyto3"
    NUCLEI3 = "nuclei3"




@register(name="Segment with Cyto Cellpose", description="Segment cytoplasm using Cellpose")
def segment_cyto(image: Image, 
                 segmentation_channel: int = 0,
                 nuclei_channel: int =  0,
                 
                model_type: ModelType = ModelType.CYTO3) -> Image:

    
    t_size = image.data.sizes["t"]
    z_size = image.data.sizes["z"]


    model = models.Cellpose(gpu=True, model_type=model_type)
    t_stacks = []

    

    for i in range(image.data.sizes["t"]):
        progress(int((i+1)/t_size * 100), f"Processing frame {i+1}/{t_size}")
        if z_size > 1:

            array = image.data.sel(t=i).transpose("z", "c", "y", "x").compute().data
            print(array.shape)


            masks, flows, styles, diams = model.eval(array, channels=[segmentation_channel, nuclei_channel], do_3D=True)
            
            masks = xr.DataArray(masks, dims=("z", "y", "x"))
            masks.expand_dims("c")
            t_stacks.append(masks)

        else:
            array = image.data.sel(t=i, z=0).transpose("x", "y", "c").compute().data

            masks, flows, styles, diams = model.eval(array, channels=[segmentation_channel, nuclei_channel])

            masks = xr.DataArray(masks, dims=("y", "x"))
            expanded_masks = masks.expand_dims("c")
            expanded_masks = expanded_masks.expand_dims("z")
            t_stacks.append(expanded_masks)


    masks = xr.concat(t_stacks, dim="t")

    return from_array_like(masks, name="Segmented Image", pixel_views=[PartialPixelViewInput()])
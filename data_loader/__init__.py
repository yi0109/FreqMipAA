from .blender import *
from .llff import *


dataset_dict = {
    "blender": SinglescaleBlenderDataset,
    "multiscale_blender": MultiscaleBlenderDataset,
    "multiscale_blender_sr": MultiscaleSRBlenderDataset,
    "video_blender": VideoBlenderDataset,
    "llff": SinglescaleLLFFDataset,
    "multiscale_llff": MultiscaleLLFFDataset
}

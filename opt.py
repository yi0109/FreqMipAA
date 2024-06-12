import yaml
import argparse
from typing import Any

from debug_utils import debug_print
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def flatten_dict(dictionary, parent_key='', sep='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened_dict[new_key] = value
    return flattened_dict


def unflatten_dict(flattened_dict, sep='.'):
    unflattened_dict = {}
    for key, value in flattened_dict.items():
        parts = key.split(sep)
        current_dict = unflattened_dict
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return unflattened_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # base config
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="./log")
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_arbitrary_resolution", action="store_true")
    parser.add_argument("--seed", type=int, default=20211202)
    parser.add_argument("--n_threads", type=int, default=-1)

    args, unknown = parser.parse_known_args()

    # load yaml file
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # override default arguments
    parser2 = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser2.set_defaults(**flatten_dict(cfg))


    # ---------- model arguments ---------- #
    parser2.add_argument("--model.name", type=str)

    
    parser2.add_argument("--model.n_masks", type=int)
    
    parser2.add_argument("--model.scale_types", type=str, action="append")
    parser2.add_argument("--model.mask_init", type=str)
    
    parser2.add_argument("--model.learnable_mask", action="store_true")
    parser2.add_argument("--model.train_freq_domain", action="store_true")
    

    # grid
    parser2.add_argument("--model.den_channels", type=int, action="append")
    parser2.add_argument("--model.app_channels", type=int, action="append")

    # density related
    parser2.add_argument("--model.density_shift", type=int)
    parser2.add_argument("--model.density_activation", type=str)

    # frequency related
    parser2.add_argument("--model.inverse_method", type=str, default="dct")
    parser2.add_argument("--model.force_mask", action="store_true")
    parser2.add_argument("--model.force_seperate", action="store_true")
    parser2.add_argument("--model.line_freq", action="store_true")
    

    # to_rgb module
    parser2.add_argument("--model.feat_n_freqs", type=int)
    parser2.add_argument("--model.view_n_freqs", type=int)
    parser2.add_argument("--model.to_rgb_in_features", type=int)
    parser2.add_argument("--model.to_rgb_hidden_features", type=int)

    # misc
    parser2.add_argument("--model.distance_scale", type=int)
    parser2.add_argument("--model.alpha_mask_threshold", type=float)
    parser2.add_argument("--model.raymarch_weight_threshold", type=float)


    # ---------- loss arguments ---------- #
    # regularization
    parser2.add_argument("--loss.l1_weight_init", type=float)
    parser2.add_argument("--loss.l1_weight_rest", type=float)
    parser2.add_argument("--loss.ortho_weight", type=float)
    parser2.add_argument("--loss.tv_weight_den", type=float)
    parser2.add_argument("--loss.tv_weight_app", type=float)

    # only for mipTensoRF
    parser2.add_argument("--loss.end_lossmult", type=int)


    # ---------- dataset arguments ---------- #
    parser2.add_argument("--dataset.name", type=str)
    parser2.add_argument("--dataset.data_dir", type=str)
    parser2.add_argument("--dataset.sr_data_dir", default=None, type=str)
    parser2.add_argument("--dataset.n_downsamples", type=int)
    parser2.add_argument("--dataset.downsample", type=float)
    parser2.add_argument("--dataset.return_radii", action="store_true")
    parser2.add_argument("--dataset.patch_size", type=int)


    
    parser2.add_argument("--training.begin_mask", type=int)

    #freqency
    parser2.add_argument("--training.adding_mask", nargs='+', type=int, help='adding mask steps')
    parser2.add_argument("--training.adjust_mask_lr", nargs='+', type=int, help='adding mask steps')
    
    # rendering
    parser2.add_argument("--training.n_samples", type=int)
    parser2.add_argument("--training.step_ratio", type=float)
    parser2.add_argument("--training.ndc_ray", action="store_true")

    # alpha mask
    parser2.add_argument("--training.alpha_mask_threshold", type=float)

    # learning rate
    parser2.add_argument("--training.lr_grid", type=float)
    parser2.add_argument("--training.lr_network", type=float)
    parser2.add_argument("--training.lr_mask", type=float)
    parser2.add_argument("--training.lr_decay_iters", type=int)
    parser2.add_argument("--training.lr_decay_target_ratio", type=float)
    parser2.add_argument("--training.lr_upsample_reset", action="store_true")
    

    # loader
    parser2.add_argument("--training.batch_size", type=int)
    parser2.add_argument("--training.n_iters", type=int)

    # grid size
    parser2.add_argument("--training.n_voxel_init", type=int)
    parser2.add_argument("--training.n_voxel_final", type=int)
    parser2.add_argument("--training.upsample_list", type=int, action="append")
    parser2.add_argument("--training.update_alpha_mask_list", type=int, action="append")

    # misc
    parser2.add_argument("--training.vis_every", type=int)
    parser2.add_argument("--training.progress_refresh_rate", type=int)
    parser2.add_argument("--training.render_train", action="store_true")
    parser2.add_argument("--training.render_test", action="store_true")
    parser2.add_argument("--training.render_path", action="store_true")
    parser2.add_argument("--training.ckpt", type=str)


    # ---------- testing arguments ---------- #
    parser2.add_argument("--testing.ndc_ray", action="store_true")
    parser2.add_argument("--testing.render_train", action="store_true")
    parser2.add_argument("--testing.render_test", action="store_true")
    parser2.add_argument("--testing.render_path", action="store_true")
    parser2.add_argument("--testing.ckpt", type=str)

    # override arguments if available
    args2 = vars(parser2.parse_args(unknown))

    # convert to original dict
    cfg = EasyDict(unflatten_dict(args2))
    debug_print(f'model name : {cfg.model["name"]}')
    
    if cfg.model["name"] == "MipFreqTensorVM":
        debug_print("additional kwargs for mipfreq")

        cfg.model["apply_mask"] = False


        if cfg.model["scale_types"] is not None:
            # arg & template
            scale_types = cfg.model["scale_types"]
            _scale_types = ["distance", "cylinder_radius", "cone_radius"]
            
            # the length must be 1 or 2
            assert (0 < len(scale_types)) and (len(scale_types) <= 2)
            
            # check if each element is one of the three types
            assert all([s in _scale_types for s in cfg.model["scale_types"]])

            # check if all elements are unique
            if len(scale_types) == 2:
                assert len(scale_types) == len(set(scale_types))

    return args, cfg

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tensor_base import TensorBase, MipTensorBase
from debug_utils import debug_print
from cosine_transform import idctn, dctn
from dct2 import idct_2d, dct_2d

class MipFreqTensorVM(MipTensorBase):
    def __init__(self, **kwargs):
        debug_print('init MipFreqTensorVM')
        super(MipFreqTensorVM, self).__init__(**kwargs)

        self.den_mats, self.den_vecs = self._init_VM(self.den_channels, 0.1)
        self.app_mats, self.app_vecs = self._init_VM(self.app_channels, 0.1)
        self.filter_freq_mats = False
        self.save_debug_mats = False
        self.apply_clamp = False
        self.add_mask_idx = -1
        self.basis_mat = nn.Linear(sum(self.app_channels), self.to_rgb_in_features, bias=False, device=self.device)
        if self.apply_mask:
            self.init_masks()
        debug_print(f'train_freq_domain : {self.train_freq_domain}')

    def adding_mask(self):
        self.add_mask_idx = self.add_mask_idx + 1
        print(f'add_mask_idx = {self.add_mask_idx}')

    def _init_VM(self, channels, scale):
        assert len(channels) == 3

        mats, vecs = [], []
        for i, ch in enumerate(channels):
            v_size = self.grid_size[self.vec_mode[i]]
            m_size0, m_size1 = self.grid_size[self.mat_mode[i]]
            mats.append(nn.Parameter(scale * torch.randn((1, ch, m_size1, m_size0))))
            vecs.append(nn.Parameter(scale * torch.randn((1, ch, v_size, 1))))
        
        return nn.ParameterList(mats).to(self.device), nn.ParameterList(vecs).to(self.device)
    
    def _create_identity_mask(self, kernel_sizes):
        kernel = torch.ones(*kernel_sizes, device=self.device)
 
        return kernel
    
    def _create_gaussian_mask_top_left(self, size, std_dev):
        t_x = torch.linspace(0, size[0] - 1, steps=size[0], device=self.device)

        if size[1] == 1:
            gaussian = torch.exp(-(t_x**2 / (2 * std_dev**2))).unsqueeze(1)
        else:
            t_y = torch.linspace(0, size[1] - 1, steps=size[1], device=self.device)
            x, y = torch.meshgrid(t_x, t_y)
            gaussian = torch.exp(-((x**2 + y**2) / (2 * std_dev**2)))
        

        return gaussian / gaussian.max()
     
    
    def _create_gaussian_masks_top_left(self, size, std_devs):
        masks = []
        for std_dev in std_devs:
            masks.append(self._create_gaussian_mask_top_left(size, std_dev))
        return torch.stack(masks)
      
    def _init_mask(self, channels, scale=None):
        debug_print(f'### init mask')

        def random_normal(out_channels, kernel_sizes):
            assert scale is not None
            return scale * torch.randn((out_channels, 1, *kernel_sizes))

        def dct_mask(out_channels, kernel_sizes):
            kernel = self._create_gaussian_mask_top_left(kernel_sizes, self.den_mats[i].shape[-2]/8)
            kernel = kernel.unsqueeze(0).repeat(out_channels, 1, 1, 1)
            return kernel

        def identity(out_channels, kernel_sizes):
            height, width = kernel_sizes

            kernel = torch.ones(out_channels, 1, *kernel_sizes)
     
            return kernel


        
        print(f'### mask init : {self.mask_init}')
        if self.mask_init == "identity":
            f = identity
        elif self.mask_init == "random_normal":
            f = random_normal
        elif self.mask_init == "dct":
            f = dct_mask
        else:
            raise ValueError("Invalid init_type")
        
        mat_masks, vec_masks, thres_masks, dct_std_devs = [], [], [], []
        self.mask_mat_size = self.den_mats[0].shape[2:]
        self.mask_vec_size = self.den_vecs[0].shape[2]
     
        for i, ch in enumerate(channels):
            self.mask_mat_size = self.den_mats[i].shape[2:]
            self.mask_vec_size = self.den_vecs[i].shape[2]
                      
            mat_mask = f(ch*self.n_masks, self.mask_mat_size).to(self.device)
            vec_mask = f(ch*self.n_masks, (self.mask_vec_size, 1)).to(self.device)
  
            thres_mask = torch.full((ch*self.n_masks,), 0.5).to(self.device)
            dct_std_dev = torch.full((ch*self.n_masks,), self.den_mats[i].shape[-2]/10, dtype=torch.float32).to(self.device)
            
            if self.learnable_mask:
                mat_mask = nn.Parameter(mat_mask)
                vec_mask = nn.Parameter(vec_mask)
                thres_mask = nn.Parameter(thres_mask)
                dct_std_dev = nn.Parameter(dct_std_dev) 
            
            mat_masks.append(mat_mask)
            vec_masks.append(vec_mask)
            thres_masks.append(thres_mask)
            dct_std_devs.append(dct_std_dev)
        
        if self.learnable_mask:
            mat_masks = nn.ParameterList(mat_masks)
            vec_masks = nn.ParameterList(vec_masks)
            thres_masks = nn.ParameterList(thres_masks)
            dct_std_devs = nn.ParameterList(dct_std_devs)

        return mat_masks, vec_masks, thres_masks, dct_std_devs

    def init_clamp(self):
        self.apply_clamp = False

    def init_masks(self):
        assert self.scale_types is not None, f"scale types : {self.scale_types}"        
        self.den_mats = self._forward_param_list(self.den_mats)
        self.app_mats = self._forward_param_list(self.app_mats)
        if self.line_freq:
            self.den_vecs = self._forward_param_list(self.den_vecs)
            self.app_vecs = self._forward_param_list(self.app_vecs)

        self.apply_mask = True
        self.den_mat_masks, self.den_vec_masks, self.den_thres_masks, self.den_dct_std_devs = self._init_mask(self.den_channels, scale=0.1)
        self.app_mat_masks, self.app_vec_masks, self.app_thres_masks, self.app_dct_std_devs = self._init_mask(self.app_channels, scale=0.1)        
        self.mat_force_masks, self.vec_force_masks = self._init_force_mask_list()
        self.max_add_mask_idx = len(self.mat_force_masks)
        
        debug_print(f'### init masks end : mask : {self.apply_mask}')

    def _init_force_mask_list(self):
        debug_print(f'create force mask')

        if self.force_seperate is False:
            dct_f_mask_idx_list = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
        else:
            dct_f_mask_idx_list = [[5],[4],[2],[1]]
        dct_std_list = [16, 10, 8, 6, 4, 2, 1, 0]
        std_list = dct_std_list
        self.f_mask_idx_list = dct_f_mask_idx_list
        mat_force_masks = [] 
        vec_force_masks = [] 
       
        for i, ch in enumerate(self.den_channels):
            mat_list = []
            vec_list = []
            for std in std_list:
                if std == 0:
                    mat_list.append(self._create_identity_mask(self.den_mat_masks[i].shape[-2:]).detach())
                    vec_list.append(self._create_identity_mask(self.den_vec_masks[i].shape[-2:]).detach())
                else:
                    mat_list.append(self._create_gaussian_mask_top_left(self.den_mat_masks[i].shape[-2:],self.den_mat_masks[i].shape[-2]/std).detach())
                    vec_list.append(self._create_gaussian_mask_top_left(self.den_vec_masks[i].shape[-2:],self.den_vec_masks[i].shape[-2]/std).detach())
            mat_force_masks.append(mat_list)
            vec_force_masks.append(vec_list)
        return mat_force_masks, vec_force_masks
    
    

    def get_params(self, lr_init_grid=0.02, lr_init_network=0.001, lr_init_mask=0.001):
        print(f'get params : mask : {self.apply_mask}')
        params = [
            {"params": self.den_mats, "lr": lr_init_grid},
            {"params": self.den_vecs, "lr": lr_init_grid},
            {"params": self.app_mats, "lr": lr_init_grid},
            {"params": self.app_vecs, "lr": lr_init_grid},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
            {"params": self.to_rgb.parameters(), "lr": lr_init_network},
        ]
        
        if self.apply_mask:
            debug_print('add mask params')
            params += [
                {"params": self.den_mat_masks, "lr": lr_init_mask},
                {"params": self.den_vec_masks, "lr": lr_init_mask},
                {"params": self.app_mat_masks, "lr": lr_init_mask},
                {"params": self.app_vec_masks, "lr": lr_init_mask},
                {"params": self.den_thres_masks, "lr": lr_init_mask},
                {"params": self.app_thres_masks, "lr": lr_init_mask},
                {"params": self.den_dct_std_devs, "lr": lr_init_mask},
                {"params": self.app_dct_std_devs, "lr": lr_init_mask},
            ]
        return params
    
    def _setup_coordinates(self, n_samples, pts, scales=None):
        if scales is None:
            # xy, xz, yz coordinates: (3, n_samples, 1, 2)
            coord_mat = pts[:, None, self.mat_mode].permute(2, 0, 1, 3)

            # 0z, 0y, 0x: (3, n_samples, 1, 2)
            zeros = torch.zeros((n_samples, 3, 1), device=self.device)
            coord_vec = torch.stack([zeros, pts[:, self.vec_mode, None]], dim=-1).permute(1, 0, 2, 3)
        else:
            # xys, xzs, yzs: (n_samples, 3, 3)
            coord_mat = torch.cat([pts[:, self.mat_mode], scales[..., None].expand(-1, 3, -1)], dim=-1)

            # (n_samples, 3, 3) --> (3, n_samples, 1, 1, 3)
            coord_mat = coord_mat.permute(1, 0, 2).view(3, -1, 1, 1, 3)

            # 0zs, 0ys, 0xs: (n_samples, 3, 3)
            zeros = torch.zeros((n_samples, 3), device=self.device)
            coord_vec = torch.stack([zeros, pts[:, self.vec_mode], scales.expand(-1, 3)], dim=-1)

            # (n_samples, 3, 3) --> (3, n_samples, 1, 1, 3)
            coord_vec = coord_vec.permute(1, 0, 2).view(3, -1, 1, 1, 3)
        return coord_mat, coord_vec

    def _inverse_features(self, mats, vecs):
        out_mats, out_vecs = [], []

        for m, v in zip(mats, vecs):     

            _m = self._inverse(m)
            if self.line_freq:
                _v = self._inverse(v)
            else:
                _v = v
            
            out_mats.append(_m)
            out_vecs.append(_v)
        
        return out_mats, out_vecs
    
    def _inverse_feature(self, features):
        out_features = []

        for m in features:    
            _m = self._inverse(m)
            out_features.append(_m)
            
        return out_features
    
    def _filter_with_freq_mask(self, mats, mat_masks, vecs, vec_masks, thres_masks, channels, n_scales):
        assert self.n_masks % n_scales == 0, f'n_masks : {self.n_masks}, n_scales : {self.n_scales}'
        out_mats, out_vecs = [], []

        for m, mm, v, vm, thres, fm_list,fv_list, ch in zip(mats, mat_masks, vecs, vec_masks, thres_masks, self.mat_force_masks, self.vec_force_masks, channels):
            # (1, ch, h, w) --> (1, ch*n_kernels, h, w)
            m = m.repeat(1, self.n_masks, 1, 1)
            v = v.repeat(1, self.n_masks, 1, 1)
            
            mm = mm.permute(1,0,2,3)
            vm = vm.permute(1,0,2,3)

            assert mm.shape==m.shape, f'mm : {mm.shape}, m : {m.shape} || vs : {vm.shape}, v : {v.shape}'
            assert vm.shape==v.shape, f'mm : {mm.shape}, m : {m.shape} || vs : {vm.shape}, v : {v.shape}'
 
            if self.force_mask:
                if all(self.add_mask_idx < len(lst) for lst in self.f_mask_idx_list) and self.add_mask_idx >= 0:

                    fm = torch.cat([fm_list[self.f_mask_idx_list[i][self.add_mask_idx]].repeat(1, ch, 1, 1).detach() for i in range(4)], dim=1)
                    fv = torch.cat([fv_list[self.f_mask_idx_list[i][self.add_mask_idx]].repeat(1, ch, 1, 1).detach() for i in range(4)], dim=1)

                    assert mm.shape == fm.shape, f'mm:{mm.shape}, fm:{fm.shape}, vm: {vm.shape}, fv: {fv.shape}'
                    assert vm.shape == fv.shape, f'mm:{mm.shape}, fm:{fm.shape}, vm: {vm.shape}, fv: {fv.shape}'
                    mm = mm * fm 
                    if self.line_freq:
                        vm = vm * fv

 

            clamp_mm = torch.clamp(torch.sigmoid(mm) - 0.5, min=0.0001, max=0.9999)
            clamp_vm = torch.clamp(torch.sigmoid(vm) - 0.5, min=0.0001, max=0.9999)
            _m = (m * clamp_mm)
            _v = (v * clamp_vm)
       
            _m = self._inverse(_m)

            if self.line_freq:
                _v = self._inverse(_v)
            
            _m = _m.reshape(1, n_scales, -1, ch, *m.shape[-2:]).transpose(2, 3)
            _v = _v.reshape(1, n_scales, -1, ch, *v.shape[-2:]).transpose(2, 3)
            
            out_mats.append(_m)
            out_vecs.append(_v)
        
        return out_mats, out_vecs
    
    
    def _inverse(self, data):
        return idctn(data, axes=(-2,-1))
        
    def _forward(self, data):
        return dctn(data, axes=(-2,-1))



    def _inverse_dct_param_list(self, param_list):
        new_param_list = nn.ParameterList()  
        for i, param in enumerate(param_list):
            inversed_param = idctn(param.data, axes = (-2,-1))
            new_param_list.append(nn.Parameter(inversed_param))  
        return new_param_list
    

    def _forward_param_list(self, param_list):
        new_param_list = nn.ParameterList()  
        for i, param in enumerate(param_list):
            forwarded_param = self._forward(param.data)
            new_param_list.append(nn.Parameter(forwarded_param))  
        return new_param_list

    def _inverse_param_list(self, param_list):
        new_param_list = nn.ParameterList()  
        for i, param in enumerate(param_list):
            inverse_data = self._inverse(param.data)
            new_param_list.append(nn.Parameter(inverse_data)) 
        return new_param_list
    
 
    def sample_den_features(self, pts, scales=None):
        # number of points
        n_samples = len(pts)

        if self.apply_mask is True and scales is not None:
            # number of scales
            n_scales = len(scales)

            coord_mats = []
            coord_vecs = []
            for scale in scales:
                # sampling positions with sales: (3, n_samples, 1, 2)
                coord_mat, coord_vec = self._setup_coordinates(n_samples, pts, scale)
                coord_mats.append(coord_mat)
                coord_vecs.append(coord_vec)

           
            mats, vecs = self._filter_with_freq_mask(
                self.den_mats, self.den_mat_masks,
                self.den_vecs, self.den_vec_masks, self.den_thres_masks,
                self.den_channels, n_scales
            )
            
                
        else:
            # sampling positions without scales: (3, n_samples, 1, 2)
            coord_mats, coord_vecs = self._setup_coordinates(n_samples, pts)
            mats, vecs = [], []
            for m, v in zip(self.den_mats, self.den_vecs):
                mats.append(m)
                vecs.append(v)

    

        # output tensor
        den_features = torch.zeros((n_samples,), device=self.device)

  
        for i in range(3):
            if self.apply_mask is True and scales is not None:
                for j in range(n_scales):
                    mat = F.grid_sample(mats[i][:, j], coord_mats[j][[i]], align_corners=True).view(-1, n_samples)
                    vec = F.grid_sample(vecs[i][:, j], coord_vecs[j][[i]], align_corners=True).view(-1, n_samples)
                    den_features = den_features + torch.sum(mat * vec, dim=0)
                den_features = den_features / n_scales
            
            else:
                mat = F.grid_sample(mats[i], coord_mats[[i]], align_corners=True).view(-1, n_samples)
                vec = F.grid_sample(vecs[i], coord_vecs[[i]], align_corners=True).view(-1, n_samples)
                den_features = den_features + torch.sum(mat * vec, dim=0)
        
        return den_features
    
    def sample_app_features(self, pts, scales=None):
        # number of points
        n_samples = len(pts)

        if self.apply_mask is True and scales is not None:
            # number of scales
            n_scales = len(scales)

            # sampling positions
            coord_mats = []
            coord_vecs = []
            for scale in scales:
                # sampling positions with sales: (3, n_samples, 1, 2)
                coord_mat, coord_vec = self._setup_coordinates(n_samples, pts, scale)
                coord_mats.append(coord_mat)
                coord_vecs.append(coord_vec)

            
            mats, vecs = self._filter_with_freq_mask(
                self.app_mats, self.app_mat_masks,
                self.app_vecs, self.app_vec_masks,self.app_thres_masks,
                self.app_channels, n_scales
            )
            self.filter_freq_mats = True
        
                
        
        else:
            # sampling positions without scales: (3, n_samples, 1, 2)
            coord_mats, coord_vecs = self._setup_coordinates(n_samples, pts)

            # 3 x (1, ch, h, w)
            # mats, vecs = self.app_mats, self.app_vecs
            # mats, vecs = self._inverse_features(self.app_mats, self.app_vecs)

            mats, vecs = [], []
            for m, v in zip(self.app_mats, self.app_vecs):
                mats.append(m)
                vecs.append(v)


            
            
        self.debug_mats = [mat.detach() for mat in mats]
        self.debug_vecs = [vec.detach() for vec in vecs]
        self.debug_channel = self.app_channels[0]
        self.save_debug_mats = True
        app_features = []
        for i in range(3):
            if self.apply_mask is True and scales is not None:
                for j in range(n_scales):
                    mat = F.grid_sample(mats[i][:, j], coord_mats[j][[i]], align_corners=True).view(-1, n_samples)
                    vec = F.grid_sample(vecs[i][:, j], coord_vecs[j][[i]], align_corners=True).view(-1, n_samples)
                    if j == 0:
                        app_features.append(mat * vec)
                    else:
                        app_features[-1] = app_features[-1] + (mat * vec)
                app_features[-1] = app_features[-1] / n_scales
            else:
                mat = F.grid_sample(mats[i], coord_mats[[i]], align_corners=True).view(-1, n_samples)
                vec = F.grid_sample(vecs[i], coord_vecs[[i]], align_corners=True).view(-1, n_samples)
                app_features.append(mat * vec)
        
        return self.basis_mat(torch.cat(app_features).T)
        
    @torch.no_grad()
    def upsample_volume_grid(self, grid_size):
        debug_print("=====> UPSAMPLING FEATURE GRID ...")
        def upsample_VM(mats, vecs):
            for i in range(3):
                v0 = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(F.interpolate(
                    mats[i].data, size=(grid_size[m1], grid_size[m0]),
                    mode="bilinear", align_corners=True
                ))
                vecs[i] = nn.Parameter(F.interpolate(
                    vecs[i].data, size=(grid_size[v0], 1),
                    mode="bilinear", align_corners=True
                ))
            return mats, vecs
        


        self.den_mats, self.den_vecs = upsample_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = upsample_VM(self.app_mats, self.app_vecs)
        self.update_step_size(grid_size)

    
    @torch.no_grad()
    def shrink_volume_grid(self, new_aabb):
        debug_print("=====> SHRINKING FEATURE GRID ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
           

        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        def shrink_VM(mats, vecs):
            for i in range(3):
                v = self.vec_mode[i]
                m0, m1 = self.mat_mode[i]
                mats[i] = nn.Parameter(mats[i].data[..., t_l[m1]:b_r[m1], t_l[m0]:b_r[m0]])
                vecs[i] = nn.Parameter(vecs[i].data[..., t_l[v]:b_r[v], :])
            return mats, vecs
        

        self.den_mats, self.den_vecs = shrink_VM(self.den_mats, self.den_vecs)
        self.app_mats, self.app_vecs = shrink_VM(self.app_mats, self.app_vecs)
        
        new_size = b_r - t_l
        self.aabb = new_aabb
        self.update_step_size(new_size)

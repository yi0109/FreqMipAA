import torch

def dct(coefs, coords=None):
    '''
    coefs: [..., C] # C: n_coefs
    coords: [..., S] # S: n_samples
    '''
    if coords is None:
        coords = torch.ones_like(coefs) \
               * torch.arange(coefs.size(-1)).to(coefs.device) # \
               # / coefs.size(-1)
    # cos = torch.cos(torch.pi * coords.unsqueeze(-1)
    cos = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5) / coefs.size(-1)
                    * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.5))
    # cos = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.) / coefs.size(-1)
    #                 * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.))
    return torch.einsum('...C,...SC->...S', coefs*(2/coefs.size(-1))**0.5, cos)


def dctn(coefs, axes=None):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))
    out = coefs
    for ax in axes:
        out = out.transpose(-1, ax)
        out = dct(out)
        out = out.transpose(-1, ax)
    return out


def idctn(coefs, axes=None, n_out=None, **kwargs):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))

    if n_out is None or isinstance(n_out, int):
        n_out = [n_out] * len(axes)

    out = coefs
    for ax, n_o in zip(axes, n_out):
        out = out.transpose(-1, ax)
        out = idct(out, n_o, **kwargs)
        out = out.transpose(-1, ax)
    return out


def idct(coefs, n_out=None):
    N = coefs.size(-1)
    if n_out is None:
        n_out = N
    '''
    # TYPE II
    out = torch.cos(torch.pi * (torch.arange(N).unsqueeze(-1) + 0.5)
                    * torch.arange(1, N) / N)
    out = 2 * torch.einsum('...C,...SC->...S', coefs[..., 1:], out)
    return out + coefs[..., :1]
    '''
    # TYPE IV
    out = torch.cos(torch.pi * (torch.arange(N).to(coefs.device) + 0.5) / N
                    * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device) + 0.5))
    # CCT version
    # out = torch.cos(torch.pi / N * (torch.arange(N).to(coefs.device))
    #                 * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device)))
    # return 2 * torch.einsum('...C,...SC->...S', coefs, out)
    return torch.einsum('...C,...SC->...S', coefs*(2/N)**0.5, out)



def apply_low_pass_filter(dct_coefficients, filter_size, return_mask=False):
    """
    Apply a low pass filter to the DCT coefficients.
    :param dct_coefficients: 3D tensor of DCT coefficients (1 x height x width).
    :param filter_size: Fraction of the coefficients to keep (0 to 1).
    :param return_mask: If True, also returns the filter mask.
    :return: Filtered DCT coefficients, (optional) filter mask.
    """
    _, height, width = dct_coefficients.shape
    filter_height = int(height * filter_size)
    filter_width = int(width * filter_size)

    # Create a mask that keeps the low frequency components and zeros out the high frequency ones
    mask = torch.zeros_like(dct_coefficients)
    mask[:, :filter_height, :filter_width] = 1

    if return_mask:
        return dct_coefficients * mask, mask
    else:
        return dct_coefficients * mask

def process_and_save_images(image_path, filter_size, output_image_path, output_filter_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_data = torch.tensor(np.array(image)).float().unsqueeze(0)  # Add batch dimension

    # Apply DCT
    dct_image = dctn(image_data)

    # Apply low pass filter and get the filter mask
    filtered_dct, filter_mask = apply_low_pass_filter(dct_image, filter_size, return_mask=True)

    # Apply IDCT
    reconstructed_image = idctn(filtered_dct)

    # Save the reconstructed image
    Image.fromarray(reconstructed_image.squeeze(0).numpy().astype('uint8')).save(f'{filter_size}_{output_image_path}')

    # Save the filter mask as an image
    filter_mask_image = filter_mask.squeeze(0).numpy() * 255  # Convert to 8-bit image
    Image.fromarray(filter_mask_image.astype('uint8')).save(f'{filter_size}_{output_filter_path}')


if __name__ == '__main__':
    
    import numpy as np
    from PIL import Image
    
    from scipy.fftpack import dct as org_dct
    from scipy.fftpack import dctn as org_dctn
    from scipy.fftpack import idct as org_idct
    from scipy.fftpack import idctn as org_idctn

    arr = torch.randn((1, 8, 240, 250)) * 10
    print((arr - dctn(idctn(arr, (-2, -1)), (-2, -1))).abs().max())
    print((arr - idctn(dctn(arr, (-2, -1)), (-2, -1))).abs().max())
    print((arr - dctn(idctn(arr, (-2,)), (-2,))).abs().max())
    print((arr - idctn(dctn(arr, (-2,)), (-2,))).abs().max())
    print((arr - idctn(dctn(arr, (-2,)), (-2,))).abs().max())
    print((org_idctn(arr.numpy(), 4, axes=(-2, -1), norm='ortho')
          - idctn(arr, (-2, -1)).numpy()).max())
    
    image_path = '/home/zeroin7/hdd_workspace/workspace/neuralfields/MipFreqGrid/mipTensoRF/log/lego_2/blender_mip_tensor_freq_vm_discrete_60000_dct_thres_sigmoid/imgs_vis/050000_002_d0.png'  # Replace with your image path
    output_path = 'filtered_image.jpg'  # Output image path
    output_filter_path = 'filter_mask.jpg'
    filter_size = 0.2  # Fraction of the DCT coefficients to keep
    
    process_and_save_images(image_path, filter_size, output_path, output_filter_path)
    '''
    arr = torch.randn((3, 8))

    print(arr) # org_idct(arr.numpy(), 4))
    print(dct(idct(arr)))
    print(idct(dct(arr)))
    print(idct(arr).numpy())
    print(org_idctn(arr.numpy(), 4, axes=(-2, -1), norm='ortho') - idctn(arr).numpy())

    print(arr)
    print(org_dct(arr.numpy()))
    print(org_dct(arr.numpy()) - dct(arr, torch.arange(8) / 8).numpy())
    print()
    print(org_dct(org_dct(arr.numpy()), type=3))
    print(org_dct(org_dct(arr.numpy()), type=3)
          - idct(dct(arr, torch.arange(8) / 8)).numpy())
    '''

    # print(idct(dct(arr, torch.arange(16) / 16)) / torch.sqrt(torch.tensor(16)))
    # print(idct(dct(arr)))
    # print(org_dct(arr.numpy()) - dct(arr).numpy())

    # ndarr = torch.randn((3, 2, 4, 5))
    # axes = (3, ) # (1, 2, 3)
    # print(org_dctn(ndarr.numpy(), axes=axes) - dctn(ndarr, axes=axes).numpy())


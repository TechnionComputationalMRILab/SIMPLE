import SimpleITK as sitk
import matplotlib.pyplot as plt
from util.util import mkdir, mkdirs
import pandas as pd
import numpy as np
import torch
import os


def save_nifti(volume, path):
    volume_nda = volume.numpy()
    volume_sitk = sitk.GetImageFromArray(volume_nda)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(volume_sitk)

    return

def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin

def convert_image_range(itk_image, min_clamp_val, max_clamp_val, clamp_en=False): #check pixel ID (GetPixelID())
    if clamp_en:
        clamper = sitk.ClampImageFilter()
        clamper.SetLowerBound(float(min_clamp_val))
        clamper.SetUpperBound(float(max_clamp_val))
        itk_image = clamper.Execute(itk_image)

    converter = sitk.RescaleIntensityImageFilter()
    converter.SetOutputMinimum(0)
    converter.SetOutputMaximum(255)
    converted_image = converter.Execute(itk_image)#(clamped_image)
    image = sitk.Cast(converted_image, sitk.sitkUInt8)

    return image

def resample_image(itk_image, plane=None, eval_plane=None, img=None):
    original_size = itk_image.GetSize()
    original_spacing = list(itk_image.GetSpacing())
    assert original_spacing[0] == original_spacing[1]
    out_spacing = [original_spacing[0], original_spacing[1], original_spacing[0]]
    out_spacing = tuple(out_spacing)
    original_spacing = tuple(original_spacing)

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(itk_image)

def resample_org_image(itk_image, img=None):
    original_ax_spacing = list(img.GetSpacing())
    original_spacing = list(itk_image.GetSpacing())
    original_size = itk_image.GetSize()
    assert original_spacing[0] == original_spacing[1]
    out_spacing = [original_ax_spacing[0], original_ax_spacing[1], original_spacing[2]]
    out_spacing = tuple(out_spacing)
    original_spacing = tuple(original_spacing)

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(itk_image)

def extract_volume_from_dicom(case_path, format, _min=0, _max=2048, clamp_en=True, stride=None, plane=None, eval_plane=None, data_dir=None, case_idx=None, prependicular_case=None):
    img = read_MRI_case(case_path, format)
    interp_img = resample_image(img)

    # mkdir(os.path.join(data_dir, 'historgrams'))
    # create_histogram_(sitk.GetArrayFromImage(img), f'hist_oasis_1_case_{case_idx}', os.path.join(data_dir, 'historgrams'))

    img = convert_image_range(img, _min, _max, clamp_en=clamp_en)
    interp_img = convert_image_range(interp_img, _min, _max, clamp_en=clamp_en)

    # if plane != eval_plane and stride is not None:
    #     interp_img = smooth_image(img, plane, eval_plane, stride)

    img_nda = sitk.GetArrayFromImage(img)
    interp_img_nda = sitk.GetArrayFromImage(interp_img)

    img_nda = minmax_scaler(img_nda, vmin=-1, vmax=1)
    interp_img_nda = minmax_scaler(interp_img_nda, vmin=-1, vmax=1)

    return img, interp_img, img_nda, interp_img_nda

def extract_patches_with_overlap(volume, patch_size=64, overlap_ratio=0.125):
    patches = []
    depth, height, width = volume.shape
    patch_depth = patch_height = patch_width = patch_size
    overlap_depth = int(patch_depth * overlap_ratio)
    overlap_height = int(patch_height * overlap_ratio)
    overlap_width = int(patch_width * overlap_ratio)

    d_step = patch_depth - overlap_depth
    h_step = patch_height - overlap_height
    w_step = patch_width - overlap_width

    for d in range(0, depth - patch_depth + 1, d_step):
        for h in range(0, height - patch_height + 1, h_step):
            for w in range(0, width - patch_width + 1, w_step):
                patch_id = dict()
                patch = volume[d:d + patch_depth, h:h + patch_height, w:w + patch_width]
                patch_id['patch'] = patch
                patches.append(patch_id)

    return patches

def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1, round_down=True):
    if round_down:
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
    else:
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) / dim_stride + 1
    return dim_out

def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])

    # (B, C, D, H, W)
    x = x.reshape(-1, channels, d_dim_in, h_dim_in * w_dim_in)
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1),
                                   dilation=(dilation[0], 1))
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]),
                                   stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x

def combine_patches_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])

    x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2],
                            h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]),
                                 padding=(padding[1], padding[2]), stride=(stride[1], stride[2]),
                                 dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1),
                                 padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x

def reconstruct_volume(opt, patches_3d_list, output_shape):
    overlap = int(opt.patch_size * opt.overlap_ratio)

    patches_3d = torch.cat(patches_3d_list)
    for i in range(overlap):
        patches_3d[:, :, i, :, :] *= (i + 1) / (overlap + 1)
        patches_3d[:, :, :, i, :] *= (i + 1) / (overlap + 1)
        patches_3d[:, :, :, :, i] *= (i + 1) / (overlap + 1)
        patches_3d[:, :, -(i + 1), :, :] *= (i + 1) / (overlap + 1)
        patches_3d[:, :, :, -(i + 1), :] *= (i + 1) / (overlap + 1)
        patches_3d[:, :, :, :, -(i + 1)] *= (i + 1) / (overlap + 1)

    output_vol = combine_patches_3d(patches_3d, opt.patch_size, output_shape,
                                    stride=int(opt.patch_size * (1 - opt.overlap_ratio)))

    ones = torch.ones_like(patches_3d).cpu()
    for i in range(overlap):
        ones[:, :, i, :, :] *= (i + 1) / (overlap + 1)
        ones[:, :, :, i, :] *= (i + 1) / (overlap + 1)
        ones[:, :, :, :, i] *= (i + 1) / (overlap + 1)
        ones[:, :, -(i + 1), :, :] *= (i + 1) / (overlap + 1)
        ones[:, :, :, -(i + 1), :] *= (i + 1) / (overlap + 1)
        ones[:, :, :, :, -(i + 1)] *= (i + 1) / (overlap + 1)
    ones_vol = combine_patches_3d(ones, opt.patch_size, output_shape, stride=int(opt.patch_size * (1 - opt.overlap_ratio)))
    recon_vol = output_vol.cpu() / ones_vol

    return recon_vol

def pad_volume(vol, dim):
    vol = change_dim(vol, dim)

    slices_num = vol.shape[0]
    if slices_num > dim:
        start = int((slices_num - dim)/2)
        end = start + dim
        res_vol = vol[start:end, :, :]
    else:
        pad_vol = torch.zeros(dim, dim, dim) - 1
        start = int((dim - slices_num)/2)
        end = start + slices_num
        pad_vol[start:end, :, :] =  vol
        res_vol = pad_vol

    return res_vol

def change_dim(image, target_dim):
    if len(image.shape) == 2:
        target_size = (target_dim, target_dim)
    elif len(image.shape) == 3:
        target_size = (image.shape[0], target_dim, target_dim)
    else:
        target_size = None
        print(f'number of image dimensions is {len(image.shape)} != 2 or 3')

    current_size = image.shape

    # Initialize the output array with zeros
    if 'torch' in str(image.dtype):
        output_image = torch.zeros(target_size, dtype=image.dtype) - 1
    else:
        output_image = np.zeros(target_size, dtype=image.dtype) - 1

    # Calculate the amount to pad or crop for each dimension
    pad_crop = [((t - c) // 2, (t - c + 1) // 2) for t, c in zip(target_size, current_size)]

    input_slices = tuple(slice(max(0, -p[0]), c - max(0, -p[1])) for p, c in zip(pad_crop, current_size))
    output_slices = tuple(slice(max(0, p[0]), t - max(0, p[1])) for p, t in zip(pad_crop, target_size))

    # Copy the data from the input to the output
    output_image[output_slices] = image[input_slices]

    return output_image

def calc_dims(case, opt):
    new_dim = np.zeros(3, dtype=int)
    old_dim = np.zeros(3, dtype=int)
    start_index = np.zeros(3, dtype=int)
    for i in range(3):
        d_dim_no_round = get_dim_blocks(case.shape[i], dim_kernel_size=opt.patch_size,
                                        dim_stride=int(opt.patch_size * (1 - opt.overlap_ratio)), round_down=False)
        d_dim_upper = int(np.ceil(d_dim_no_round))
        new_dim[i] = opt.patch_size * d_dim_upper - int(opt.patch_size * opt.overlap_ratio) * (d_dim_upper - 1)
        old_dim[i] = case.shape[i]
        start_index[i] = int((new_dim[i] - old_dim[i]) / 2)
    return new_dim, old_dim, start_index

def simple_test_preprocess(case, opt):
    _, _, _, interp_vol_nda = extract_volume_from_dicom(case, opt.data_format, _min=opt.global_min, _max=opt.global_max, clamp_en=opt.clamp_en)

    interp_vol_nda = change_dim(interp_vol_nda, target_dim=opt.vol_cube_dim)
    interp_vol = torch.from_numpy(interp_vol_nda).to(torch.float32)
    new_dim, old_dim, s = calc_dims(interp_vol, opt)

    padded_case = torch.zeros((1, 1, new_dim[0], new_dim[1], new_dim[2])) - 1
    padded_case[:, :, s[0]:(s[0] + old_dim[0]), s[1]:(s[1] + old_dim[1]), s[2]:(s[2] + old_dim[2])] = interp_vol
    patches_3d = extract_patches_3d(padded_case, kernel_size=opt.patch_size, stride=int(opt.patch_size * (1 - opt.overlap_ratio)))

    return patches_3d, padded_case


def simple_train_preprocess(opt):
    df = pd.read_csv(os.path.join(opt.csv_name), low_memory=False)
    cases_paths = df.loc[:, opt.eval_plane]

    cases_num = len(cases_paths)

    opt.data_dir = os.path.join(opt.main_root, opt.model_root, 'data')
    save_train_dir = os.path.join(opt.data_dir, 'train')

    mkdirs([opt.data_dir, save_train_dir])

    if opt.global_min == 0 and opt.global_max == 0:
        opt.global_min, opt.global_max = find_grayscale_limits(cases_paths, opt.data_format)

    save_idx = 0

    for case_idx in range(cases_num):
        case = cases_paths[case_idx]
        print(f'case no: {case_idx} / {cases_num}, {case=}')
        half_d = int(opt.patch_size / 2)

        _, _, _, interp_vol_nda = extract_volume_from_dicom(case, opt.data_format, _min=opt.global_min, _max=opt.global_max, clamp_en=opt.clamp_en)
        interp_vol_nda = change_dim(interp_vol_nda, target_dim=opt.vol_cube_dim)
        interp_vol = torch.from_numpy(interp_vol_nda).to(torch.float32).cpu().detach()
        interp_vol = interp_vol[half_d : -half_d, half_d : -half_d, half_d : -half_d]

        interp_patches = extract_patches_with_overlap(interp_vol, opt.patch_size, opt.overlap_ratio)
        if 'coronal' in opt.planes:
            cor_atme_vol = torch.load(os.path.join(opt.main_root, opt.atme_cor_root, 'data', 'generation', f'case_{case_idx}', 'atme_vol.pt')).cpu().detach()
            cor_atme_vol = cor_atme_vol[half_d : -half_d, half_d : -half_d, half_d : -half_d]
            cor_atme_patches = extract_patches_with_overlap(cor_atme_vol, opt.patch_size, opt.overlap_ratio)
            assert (len(interp_patches) == len(cor_atme_patches))
        if 'axial' in opt.planes:
            ax_atme_vol = torch.load(os.path.join(opt.main_root, opt.atme_ax_root, 'data', 'generation', f'case_{case_idx}', 'atme_vol.pt')).cpu().detach()
            ax_atme_vol = ax_atme_vol[half_d : -half_d, half_d : -half_d, half_d : -half_d]
            ax_atme_patches = extract_patches_with_overlap(ax_atme_vol, opt.patch_size, opt.overlap_ratio)
            assert (len(interp_patches) == len(ax_atme_patches))
        if 'sagittal' in opt.planes:
            sag_atme_vol = torch.load(os.path.join(opt.main_root, opt.atme_sag_root, 'data', 'generation', f'case_{case_idx}', 'atme_vol.pt')).cpu().detach()
            sag_atme_vol = sag_atme_vol[half_d : -half_d, half_d : -half_d, half_d : -half_d]
            sag_atme_patches = extract_patches_with_overlap(sag_atme_vol, opt.patch_size, opt.overlap_ratio)
            assert (len(interp_patches) == len(sag_atme_patches))


        for i in range(len(interp_patches)):
            interp_patch = interp_patches[i]['patch'].unsqueeze(0).clone()
            data = {'interp_patch': interp_patch}
            if 'coronal' in opt.planes:
                cor_atme_patch = cor_atme_patches[i]['patch'].unsqueeze(0).clone()
                data['cor_atme_patch'] = cor_atme_patch
            if 'axial' in opt.planes:
                ax_atme_patch = ax_atme_patches[i]['patch'].unsqueeze(0).clone()
                data['ax_atme_patch'] = ax_atme_patch
            if 'sagittal' in opt.planes:
                sag_atme_patch = sag_atme_patches[i]['patch'].unsqueeze(0).clone()
                data['sag_atme_patch'] = sag_atme_patch
            torch.save(data, os.path.join(save_train_dir, f'data_{save_idx}.pt'))
            save_idx += 1

def read_MRI_case(case_path, format):
    if format == 'dicom':
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(case_path)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    elif format == 'nifti':
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(case_path)
        img = reader.Execute()
    else:
        img = None
        print(f'format {format} is not exist!')

    img = permute_img_dims_order(img)

    return img

def permute_img_dims_order(img):
    permute_filter = sitk.PermuteAxesImageFilter()
    spacing = img.GetSpacing()
    assert spacing[0] == spacing[1] or spacing[0] == spacing[2] or spacing[1] == spacing[2]
    if spacing[0] == spacing[1]:
        return img
    elif spacing[0] == spacing[2]:
        new_order = [0, 2, 1]
        permute_filter.SetOrder(new_order)
        return permute_filter.Execute(img)
    else: # spacing[1] == spacing[2]
        new_order = [1, 2, 0]
        permute_filter.SetOrder(new_order)
        return permute_filter.Execute(img)

def find_grayscale_limits(cases, data_format='dicom'):
    global_min = np.inf
    global_max = 0

    for i, case in enumerate(cases):
        img = read_MRI_case(case, data_format)
        img_nda = sitk.GetArrayFromImage(img)
        _min = np.min(img_nda)
        _max = np.max(img_nda)
        if _min < global_min: global_min = _min
        if _max > global_max: global_max = _max

    return global_min, global_max

def smooth_image(img, plane, eval_plane, stride):
    dim = None
    if eval_plane == 'coronal':
        if plane == 'axial': dim = 1
        if plane == 'sagittal': dim = 0
    elif eval_plane == 'axial':
        if plane == 'sagittal': dim = 1
        if plane == 'coronal': dim = 1
    elif eval_plane == 'sagittal':
        if plane == 'axial': dim = 0
        if plane == 'coronal': dim = 0

    sigma = max(stride / 2.355, 0.00001)

    image_dimension = img.GetDimension()
    sigma_array = [0.00001] * image_dimension
    sigma_array[dim] = sigma

    gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian_filter.SetSigma(sigma_array)
    smoothed_img = gaussian_filter.Execute(img)

    return smoothed_img

def stride_image(opt, interp_img):
    stride = None
    if opt.eval_plane == 'coronal':
        if opt.plane == 'axial': stride = 'y'
        if opt.plane == 'sagittal': stride = 'x'
    elif opt.eval_plane == 'axial':
        if opt.plane == 'sagittal': stride = 'y'
        if opt.plane == 'coronal': stride = 'y'
    elif opt.eval_plane == 'sagittal':
        if opt.plane == 'axial': stride = 'x'
        if opt.plane == 'coronal': stride = 'x'

    if stride == None:
        return interp_img
    else:
        strided_interp_img = np.zeros_like(interp_img)
        if stride == 'x':
            interp_img = np.transpose(interp_img)
            strided_interp_img = np.transpose(strided_interp_img)

        for k in range(0, opt.vol_cube_dim, opt.stride):
            strided_interp_img[k:k + opt.stride, :] = interp_img[k, :]

        if stride == 'x':
            strided_interp_img = np.transpose(strided_interp_img)

        return strided_interp_img

def create_histogram_(image, title, save_path):
    total_hist=np.zeros((256,))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for i in range(image.shape[0]):
        histogram,bin_edges=np.histogram(image[i,:,:],bins=256,range=(np.min(image),np.max(image)))
        total_hist+=histogram
    plt.figure()
    plt.title(f'GrayscaleHistogram {title}')
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([np.min(image),np.max(image)])
    plt.ylim([0.0,np.max(total_hist)])
    plt.plot(bin_edges[0:-1],total_hist)
    plt.savefig(os.path.join(save_path,f'{title}.png'))
    plt.close()

def atme_train_preprocess(opt):
    df = pd.read_csv(os.path.join(opt.csv_name), low_memory=False)
    cases_paths = df.loc[:, opt.plane]
    ax_cases_path = df.loc[:, 'axial']

    save_idx = 0

    if opt.global_min == 0 and opt.global_max == 0:
        opt.global_min, opt.global_max = find_grayscale_limits(cases_paths, opt.data_format)

    for i, case in enumerate(cases_paths):
        print(f'case no: {i} / {len(cases_paths)}, {case=}')
        ax_case = ax_cases_path[i]
        org_vol, interp_vol, org_vol_nda, interp_vol_nda = extract_volume_from_dicom(case, opt.data_format, _min=opt.global_min, _max=opt.global_max, clamp_en=opt.clamp_en, stride=opt.stride, plane=opt.plane, eval_plane=opt.eval_plane, data_dir=opt.data_dir, case_idx=i, prependicular_case=ax_case) #eval_case=eval_case) #prependicular_case=ax_case)

        org_slices_num = org_vol.GetSize()[2]
        interp_slices_num = interp_vol.GetSize()[2]

        slice_spacing = list(org_vol.GetSpacing())[2]

        for s in range(opt.crop_val, (org_slices_num - opt.crop_val)):
            org_img = org_vol_nda[s, :, :]
            org_coord = org_vol.TransformIndexToPhysicalPoint((0, 0, s))
            dist = []

            for j in range(interp_slices_num):
                interp_coord = interp_vol.TransformIndexToPhysicalPoint((0, 0, j))
                d = np.linalg.norm(np.asarray(org_coord) - np.asarray(interp_coord))
                abs_d = np.abs(d-(slice_spacing/2))
                dist.append(abs_d)
            dist_np = np.array(dist)
            indices = dist_np.argsort()[:2]

            interp_coord0 = interp_vol.TransformIndexToPhysicalPoint((0, 0, int(indices[0])))
            interp_coord1 = interp_vol.TransformIndexToPhysicalPoint((0, 0, int(indices[1])))

            d1 = np.linalg.norm(np.asarray(org_coord) - np.asarray(interp_coord0))
            d2 = np.linalg.norm(np.asarray(org_coord) - np.asarray(interp_coord1))

            a = d1 / (d1 + d2)
            b = 1 - a

            interp_img = b * interp_vol_nda[int(indices[0]), :, :] + a * interp_vol_nda[int(indices[1]), :, :]

            org_img = change_dim(org_img, target_dim=opt.vol_cube_dim)
            interp_img = change_dim(interp_img, target_dim=opt.vol_cube_dim)

            interp_img = stride_image(opt, interp_img)

            org_img = np.expand_dims(org_img, axis=0)
            interp_img = np.expand_dims(interp_img, axis=0)

            torch.save(torch.from_numpy(org_img).to(torch.float32), os.path.join(opt.data_dir, 'original', f'img_{save_idx}.pt'))
            torch.save(torch.from_numpy(interp_img).to(torch.float32), os.path.join(opt.data_dir, 'interpolation', f'img_{save_idx}.pt'))

            save_idx += 1

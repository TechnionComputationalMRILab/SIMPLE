import SimpleITK as sitk
from util.util import mkdir, mkdirs
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import re


EXCLUDE_CASES = ['1809432969308/20190902/cor 2D FIESTA (7)', '3413257422381/20181012/cor 2D FIESTA (5)',
                 '3842830969373/20180601/cor 2D FIESTA (4)', '3554556187316/20200224/cor 2D FIESTA (4)',
                 '3196140605214/20201223/cor 2D FIESTA (4)', '3104779470389/20190508/Ax 2D FIESTA (5)',
                 '3239335163204/20201021/Ax 2D FIESTA (3)', '3401307597322/20200103/Ax 2D FIESTA (3)',
                 '3472726507223/20190222/Ax 2D FIESTA (4)']


def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin

def convert_image_range(itk_image, min_clamp_val, max_clamp_val): #check pixel ID (GetPixelID())
    clamper = sitk.ClampImageFilter()
    clamper.SetLowerBound(float(min_clamp_val))
    clamper.SetUpperBound(float(max_clamp_val))
    clamped_image = clamper.Execute(itk_image)

    converter = sitk.RescaleIntensityImageFilter()
    converter.SetOutputMinimum(0)
    converter.SetOutputMaximum(255)
    converted_image = converter.Execute(clamped_image)
    image = sitk.Cast(converted_image, sitk.sitkUInt8)

    return image

def resample_image(itk_image):
    original_spacing = list(itk_image.GetSpacing())
    original_size = itk_image.GetSize()
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

def organize_data(opt): #should be different according to the dataset
    if os.path.exists(os.path.join(opt.main_root, 'axial_cases_paths.pt')) and \
            os.path.exists(os.path.join(opt.main_root, 'coronal_cases_paths.pt')): return

    ax_cases_paths = []
    cor_cases_paths = []
    for d1 in os.listdir(opt.dataroot):
        if str(d1) == 'metadata.db': continue
        for d2 in os.listdir(os.path.join(opt.dataroot, str(d1))):
            max_ax_serial_num = 0
            max_cor_serial_num = 0
            ax_case = ''
            cor_case = ''
            for d3 in os.listdir(os.path.join(opt.dataroot, str(d1), str(d2))):
                if os.path.join(str(d1), str(d2), str(d3)) in EXCLUDE_CASES: continue
                d3_arr = d3.split()
                if len(d3_arr) < 3 or len(d3_arr) > 4: continue
                if 'fiesta' not in d3_arr[2].lower() and 'fiesta' not in d3_arr[1].lower(): continue
                if 'ax' in d3_arr[0].lower():
                    pattern = r'[()]'
                    result = re.split(pattern, d3_arr[-1])
                    ax_serial_num = int(result[1])
                    if ax_serial_num >= max_ax_serial_num:
                        max_ax_serial_num = ax_serial_num
                        ax_case = os.path.join(opt.dataroot, str(d1), str(d2), str(d3))
                if 'cor' in d3_arr[0].lower():
                    pattern = r'[()]'
                    result = re.split(pattern, d3_arr[-1])
                    cor_serial_num = int(result[1])
                    if cor_serial_num >= max_cor_serial_num:
                        max_cor_serial_num = cor_serial_num
                        cor_case = os.path.join(opt.dataroot, str(d1), str(d2), str(d3))

            if ax_case != '' and cor_case != '':
                ax_cases_paths.append(ax_case)
                cor_cases_paths.append(cor_case)

    torch.save(ax_cases_paths, os.path.join(opt.main_root, 'axial_cases_paths.pt'))
    torch.save(cor_cases_paths, os.path.join(opt.main_root, 'coronal_cases_paths.pt'))

def extract_volume_from_dicom(case_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    interp_img = resample_image(img)

    img = convert_image_range(img, 0, 2048)
    interp_img = convert_image_range(interp_img, 0, 2048)

    img_nda = sitk.GetArrayFromImage(img)
    interp_img_nda = sitk.GetArrayFromImage(interp_img)

    img_nda = minmax_scaler(img_nda, vmin=-1, vmax=1)
    interp_img_nda = minmax_scaler(interp_img_nda, vmin=-1, vmax=1)

    return img, interp_img, torch.from_numpy(img_nda), torch.from_numpy(interp_img_nda)

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
    patches_3d = torch.cat(patches_3d_list)
    output_vol = combine_patches_3d(patches_3d, opt.patch_size, output_shape,
                                    stride=int(opt.patch_size * (1 - opt.overlap_ratio)))

    ones = torch.ones_like(patches_3d).cpu()
    ones_vol = combine_patches_3d(ones, opt.patch_size, output_shape, stride=int(opt.patch_size * (1 - opt.overlap_ratio)))
    recon_vol = output_vol.cpu() / ones_vol

    return recon_vol

def pad_volume(vol):
    slices_num = vol.shape[0]
    if slices_num > 512:
        start = int((slices_num - 512)/2)
        end = start + 512
        return vol[start:end, :, :]
    else:
        pad_vol = torch.zeros(512, 512, 512) - 1
        start = int((512 - slices_num)/2)
        end = start + slices_num
        pad_vol[start:end, :, :] =  vol
        return pad_vol

def calc_dims(case, opt):
    d_dim_no_round = get_dim_blocks(case.shape[0], dim_kernel_size=opt.patch_size,
                                    dim_stride=int(opt.patch_size * (1 - opt.overlap_ratio)), round_down=False)
    d_dim_upper = int(np.ceil(d_dim_no_round))
    new_dim = opt.patch_size * d_dim_upper - int(opt.patch_size * opt.overlap_ratio) * (d_dim_upper - 1)
    old_dim = case.shape[0]
    return new_dim, old_dim

def simple_test_preprocess(case, opt):
    _, _, _, interp_case = extract_volume_from_dicom(case)

    new_dim, old_dim = calc_dims(interp_case, opt)

    start_idx = int((new_dim - old_dim) / 2)
    padded_case = torch.zeros((1, 1, new_dim, 512, 512)) - 1
    padded_case[:, :, start_idx:(start_idx + old_dim), :, :] = interp_case

    patches_3d = extract_patches_3d(padded_case, kernel_size=opt.patch_size, stride=int(opt.patch_size * (1 - opt.overlap_ratio)))

    return patches_3d, padded_case


def simple_train_preprocess(opt):
    cor_cases_paths = torch.load(os.path.join(opt.main_root, 'coronal_cases_paths.pt'))

    cases_num = len(cor_cases_paths)

    opt.save_dir = os.path.join(opt.main_root, opt.simple_root)
    opt.data_dir = os.path.join(opt.main_root, opt.simple_root, 'data')
    save_train_dir = os.path.join(opt.data_dir, 'train')

    mkdirs([opt.save_dir, opt.data_dir, save_train_dir])

    save_idx = 0

    for case_idx in range(cases_num):
        cor_case = cor_cases_paths[case_idx]

        _, _, _, interp_vol_nda = extract_volume_from_dicom(cor_case)
        cor_atme_vol = torch.load(os.path.join(opt.main_root, opt.atme_cor_root, 'data', 'generation', f'case_{case_idx}', 'atme_vol.pt')).cpu().detach()
        ax_atme_vol = torch.load(os.path.join(opt.main_root, opt.atme_ax_root, 'data', 'generation', f'case_{case_idx}', 'atme_vol.pt')).cpu().detach()

        if case_idx == 7 or case_idx == 70:
            cor_atme_vol = torch.flip(cor_atme_vol, [0])
            ax_atme_vol = torch.flip(ax_atme_vol, [0])

        interp_patches = extract_patches_with_overlap(interp_vol_nda, opt.patch_size, opt.overlap_ratio)
        cor_atme_patches = extract_patches_with_overlap(cor_atme_vol, opt.patch_size, opt.overlap_ratio)
        ax_atme_patches = extract_patches_with_overlap(ax_atme_vol, opt.patch_size, opt.overlap_ratio)

        assert(len(interp_patches)==len(cor_atme_patches))
        assert(len(cor_atme_patches)==len(ax_atme_patches))

        for i in range(len(interp_patches)):
            interp_patch = interp_patches[i]['patch'].unsqueeze(0).clone()
            cor_atme_patch = cor_atme_patches[i]['patch'].unsqueeze(0).clone()
            ax_atme_patch = ax_atme_patches[i]['patch'].unsqueeze(0).clone()

            data = {'interp_patch': interp_patch, 'cor_atme_patch': cor_atme_patch, 'ax_atme_patch': ax_atme_patch}

            torch.save(data, os.path.join(save_train_dir, f'data_{save_idx}.pt'))
            save_idx += 1

def atme_train_preprocess(opt):
    cases_paths = torch.load(os.path.join(opt.main_root, f'{opt.plane}_cases_paths.pt'))

    save_idx = 0

    for i, case in enumerate(cases_paths):
        print(f'case no: {i} in {len(cases_paths)}, {case=}')
        org_vol, interp_vol, org_vol_nda, interp_vol_nda = extract_volume_from_dicom(case)

        if org_vol.GetSize()[1] != 512: continue

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

            if opt.plane == 'axial':
                strided_interp_img = np.zeros_like(interp_img)
                for i in range(0, 512, opt.stride):
                    strided_interp_img[i:i + opt.stride, :] = interp_img[i, :]
                interp_img = strided_interp_img

            org_img = np.expand_dims(org_img, axis=0)
            interp_img = np.expand_dims(interp_img, axis=0)

            torch.save(torch.from_numpy(org_img), os.path.join(opt.data_dir, 'original', f'img_{save_idx}.pt'))
            torch.save(torch.from_numpy(interp_img), os.path.join(opt.data_dir, 'interpolation', f'img_{save_idx}.pt'))

            save_idx += 1

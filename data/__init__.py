import numpy as np
from torch.utils.data import Dataset, DataLoader
from .preprocess import simple_train_preprocess, simple_test_preprocess, atme_train_preprocess, extract_volume_from_dicom
import torch
import os

def create_simple_train_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'simple.py'/'test.py'

    """
    opt.data_dir = os.path.join(opt.main_root, opt.simple_root, opt.data_name)

    if not os.path.exists(os.path.join(opt.data_dir, 'train')):
        os.makedirs(os.path.join(opt.data_dir, 'train'))

    empty_data_dir = True if len(os.listdir(os.path.join(opt.data_dir, 'train'))) == 0 else False

    if opt.calculate_dataset or empty_data_dir:
        simple_train_preprocess(opt)

    dataset = SimpleTrainDataset(os.path.join(opt.data_dir, 'train'))
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    return data_loader

def create_simple_test_dataset(case, opt):
    dataset = SimpleTestDataset(case, opt)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    return data_loader

def create_atme_train_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'simple.py'/'test.py'

    """
    if not os.path.exists(os.path.join(opt.data_dir, 'original')):
        os.makedirs(os.path.join(opt.data_dir, 'original'))
    if not os.path.exists(os.path.join(opt.data_dir, 'interpolation')):
        os.makedirs(os.path.join(opt.data_dir, 'interpolation'))

    empty_data_dir = True if len(os.listdir(os.path.join(opt.data_dir, 'original'))) == 0 else False
    if opt.calculate_dataset or empty_data_dir:
        atme_train_preprocess(opt)

    dataset = AtmeTrainDataset(opt.data_dir)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    return data_loader

def create_atme_test_dataset(opt, case, case_index):
    dataset = AtmeTestDataset(opt.plane, case, case_index)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)
    return data_loader

class SimpleTrainDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.path, f'data_{idx}.pt'))
        cor_interp_3d = data["interp_patch"]
        cor_gen_3d = data["cor_atme_patch"]
        cor_samp_3d = data["ax_atme_patch"]

        if self.transform is not None:
            cor_interp_3d = self.transform(cor_interp_3d)
            cor_samp_3d = self.transform(cor_samp_3d)
            cor_gen_3d = self.transform(cor_gen_3d)

        ret_dict = {'A': cor_interp_3d, 'B': cor_gen_3d, 'C': cor_samp_3d}

        return ret_dict

class SimpleTestDataset(Dataset):
    def __init__(self, case, opt, transform=None, target_transform=None):
        self.patches_3d, self.padded_case = simple_test_preprocess(case, opt)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.patches_3d.shape[0]

    def __getitem__(self, idx):
        interp_patch_3d = self.patches_3d[idx, 0, :, :, :].unsqueeze(0)

        if self.transform is not None:
            interp_patch_3d = self.transform(interp_patch_3d)

        ret_dict = {'A': interp_patch_3d}

        return ret_dict

class AtmeTrainDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, 'original')))

    def __getitem__(self, idx):
        org_img = torch.load(os.path.join(self.path, 'original', f'img_{idx}.pt'))
        interp_img = torch.load(os.path.join(self.path, 'interpolation', f'img_{idx}.pt'))

        if self.transform is not None:
            org_img = self.transform(org_img)
            interp_img = self.transform(interp_img)

        ret_dict = {'A': interp_img,
                    'B': org_img,
                    'A_paths': os.path.join(self.path, 'interpolation', f'img_{idx}.pt'),
                    'B_paths': os.path.join(self.path, 'original', f'img_{idx}.pt'),
                    'batch_indices': idx}

        return ret_dict

class AtmeTestDataset(Dataset):
    def __init__(self, plane, case, case_index, transform=None, target_transform=None):
        _, _, _, case_interp_vol = extract_volume_from_dicom(case)
        self.case_interp_vol = case_interp_vol
        self.plane = plane
        self.start_idx = None
        self.end_idx = None
        if self.plane == 'axial': self.case_interp_vol = self.pad_vol()
        self.case_index = case_index
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.case_interp_vol.shape[0]

    def __getitem__(self, idx):
        interp_img = self.case_interp_vol[idx, :, :]

        if self.transform is not None:
            interp_img = self.transform(interp_img)

        ret_dict = {'A': interp_img,
                    'batch_indices': idx}

        return ret_dict

    def pad_vol(self):
        padding_case = torch.zeros((512, 512, 512)) - 1
        if self.case_interp_vol.shape[0] <= 512:
            self.start_idx = int(np.ceil((512 - self.case_interp_vol.shape[0]) / 2))
            self.end_idx = self.start_idx + self.case_interp_vol.shape[0]
            padding_case[self.start_idx:self.end_idx, :, :] = self.case_interp_vol
        else:
            self.start_idx = int(np.ceil((self.case_interp_vol.shape[0] - 512) / 2))
            self.end_idx = self.start_idx + 512
            padding_case = self.case_interp_vol[self.start_idx:self.end_idx, :, :]

        return torch.movedim(padding_case, (0, 1, 2), (1, 0, 2))

    def crop_volume(self, vol):
        vol = torch.movedim(vol, (0, 1, 2), (1, 0, 2))
        if self.end_idx > 512:
            return vol
        else:
            return vol[self.start_idx:self.end_idx, :, :]
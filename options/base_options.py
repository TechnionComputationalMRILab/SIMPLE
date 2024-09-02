import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--main_root', type=str, default='outputs', help='path to save script outputs')
        parser.add_argument('--dataroot', required=True, default='/tcmldrive/databases/Private/RambamMRE082022Sorter/', help='path to images')
        parser.add_argument('--save_dir', default='outputs', help='path to save script outputs (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--calculate_dataset', default=False, action=argparse.BooleanOptionalAction, help='if specified, calculate and pre-preprocess dataset')
        parser.add_argument('--exp_name', type=str, default='trial', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--data_name', type=str, default='data', help='name of the data directory. It decides where to store samples and models')
        parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self, parser=None):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        print('here-0')
        if not self.initialized:  # check if it has been initialized
            if parser is None:
                parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        print('here-1')
        # get the basic options
        opt, _ = parser.parse_known_args()
        print(f'{opt=}')
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            # default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.main_root, opt.atme_root, opt.exp_name, opt.checkpoints_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, parser=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(parser)

        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix


        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
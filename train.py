from data.preprocess import organize_data
from options.train_atme_options import TrainAtmeOptions
from options.train_simple_options import TrainSimpleOptions
import atme
import simple

if __name__ == '__main__':
    simple_opt = TrainSimpleOptions().parse()

    simple.setup(simple_opt)

    organize_data(simple_opt)

    # Coronal ATME
    # atme_opt = TrainAtmeOptions().parse()
    # atme_opt.plane = 'coronal'
    # atme_opt.atme_root = atme_opt.atme_cor_root
    # atme.train(atme_opt)
    # atme.test(atme_opt)

    # Axial ATME
    # atme_opt.plane = 'axial'
    # atme_opt.atme_root = atme_opt.atme_ax_root
    # atme.train(atme_opt)
    # atme.test(atme_opt)

    # SIMPLE
    simple.train(simple_opt)

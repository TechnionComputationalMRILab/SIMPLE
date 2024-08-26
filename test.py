from options.train_simple_options import TrainSimpleOptions
import simple


if __name__ == '__main__':
    simple_opt = TrainSimpleOptions().parse()
    simple.test(simple_opt)
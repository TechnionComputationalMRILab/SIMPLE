from options.atme_options import AtmeOptions
from options.simple_options import SimpleOptions
import atme
import simple
import argparse


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description="Main parser with subparsers")
    subparsers = main_parser.add_subparsers(dest="model")

    parser1 = subparsers.add_parser(name="atme")
    parser2 = subparsers.add_parser(name="simple")

    atme_opt = AtmeOptions().parse(parser1)
    simple_opt = SimpleOptions().parse(parser2)

    opt = main_parser.parse_args()

    if opt.model == "atme":
        AtmeOptions().print_options(atme_opt)
        atme.test(atme_opt)
    elif opt.model == "simple":
        SimpleOptions().print_options(simple_opt)
        simple.test(simple_opt)
    else:
        print(f'model {opt.model} is not exist!')
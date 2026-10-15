import argparse

import cusvna

parser = argparse.ArgumentParser(prog="python -m cusvna", description=cusvna.__doc__.strip().split("\n")[0])
parser.add_argument("--include", action="store_true", help="print the include directory (for -I)")
parser.add_argument("--version", action="version", version=cusvna.__version__)
args = parser.parse_args()
if args.include:
    print(cusvna.get_include())
else:
    parser.print_help()

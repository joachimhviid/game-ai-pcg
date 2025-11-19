import sys
from black import main


def run():
    sys.argv = ["black", "."]
    raise SystemExit(main())

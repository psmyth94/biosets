import argparse

from .__version__ import __version__


def main():
    print("use biosets instead. available commands: --version")
    # print(f"{__file__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"{__version__}")
    parser.parse_args()


if __name__ == "__main__":
    main()

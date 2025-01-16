# cli.py

import argparse

from flask_ml.flask_ml_cli import MLCli
from .server import server


def main():
    parser = argparse.ArgumentParser(description="<Enter a description of your tool here>")
    cli = MLCli(server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()

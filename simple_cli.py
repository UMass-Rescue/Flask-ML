# cli.py

import argparse

from flask_ml.flask_ml_cli import MLCli
from simple_server import server


def main():
    parser = argparse.ArgumentParser(description="Transform Case of multiple text inputs")
    cli = MLCli(server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()

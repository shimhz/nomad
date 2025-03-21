import click
from nomad import Nomad


@click.command()
@click.option("--nmr", type=str, help="Path to non-matching reference files")
@click.option("--deg", type=str, help="Path to test files")
@click.option(
    "--device",
    type=str,
    default=None,
    help="Specify device, cuda or cpu. Automatically set cuda if None and GPU is detected",
)
def main(nmr, deg, device):

    nomad_model = Nomad(device)
    nomad_avg = nomad_model.predict(nmr, deg)


if __name__ == "__main__":
    main()

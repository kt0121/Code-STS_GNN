"""SimGNN runner."""

from args import argument_parser
from simgnn import SimGNNTrainer
from utils import tab_printer


def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = argument_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
    trainer.score()
    if args.save_path:
        trainer.save()


if __name__ == "__main__":
    main()

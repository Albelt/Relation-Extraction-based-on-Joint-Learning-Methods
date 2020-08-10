import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from args import train_argparser
from spert import input_reader
from spert.spert_trainer import SpERTTrainer


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


if __name__ == '__main__':
    arg_parser = train_argparser()
    args, _ = arg_parser.parse_known_args()
    __train(args)

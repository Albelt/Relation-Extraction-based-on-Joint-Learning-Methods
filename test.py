from args import eval_argparser
from spert import input_reader
from spert.spert_trainer import SpERTTrainer

def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


if __name__ == '__main__':
    arg_parser = eval_argparser()
    args, _ = arg_parser.parse_known_args()
    __eval(args)
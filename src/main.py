import os
import argparse

from transformers import AutoTokenizer

from utils.io_utils import *
from data_processing import get_dataset_readers
from prompt import get_prompts
from llm import get_generations
from evaluate import get_evaluations


def main(args):

    # make experiment directory
    os.makedirs(args.exp_dir, exist_ok=True)

    # read in data
    dataset_reader = get_dataset_readers(args)
    data = dataset_reader.split()
    print(f"Dataset has {len(data)} examples")

    # prompt generation
    prompts = get_prompts(data, args.exp_dir, args.prompt_template)

    # response generation
    responses = get_generations(prompts, args.exp_dir, args.model_id)

    # extract answers from response
    doc_predictions = dataset_reader.aggregate(responses)
    dp_filepath = os.path.join(args.exp_dir, f"doc_predictions.jsonl")
    write_jsonl(doc_predictions, dp_filepath)

    # evaluate results
    get_evaluations(doc_predictions, args.exp_dir)


if __name__ == "__main__":
    # take args
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, help="Filepath to experiment directory")
    parser.add_argument("--eval_data", type=str, help="Filepath to evaluation data")
    parser.add_argument("--model_id", type=str, help="ID of LLMs, e.g. gpt-4")
    parser.add_argument(
        "--prompt_template",
        type=str,
        help="`doc_template` or `qa_template`)",
    )

    args = parser.parse_args()
    main(args)

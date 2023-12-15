import os

from utils.io_utils import *


def get_prompts(data: list, exp_dir: str, prompt_template: str):

    # get all the arguments
    TEMPLATES = {"doc_template": doc_template, "qa_template": qa_template}
    template_fn = TEMPLATES[prompt_template]
    prompts_filepath = os.path.join(exp_dir, "prompts.json")

    # read in if exists
    if os.path.exists(prompts_filepath):
        prompts = read_json(prompts_filepath)

    else:
        prompts = dict()
        for example in data:

            e_key = example["example_key"]
            if e_key not in prompts:

                # generate prompt
                prompt = template_fn(example)
                prompts[e_key] = prompt
        write_json(prompts, prompts_filepath)
    return prompts


def doc_template(example: dict) -> str:
    """Example prompt instantiated from this template:
    ```
    Annotate all entity mentions in the following text with coreference clusters.
    Use Markdown tags to indicate clusters in the output, with the following format
    [mention](#cluster_name)

    Input: [Tom](#) and [Mary](#) go to [the park](#). [It](#) was full of trees.
    Output: [Tom](#cluster_0) and [Mary](#cluster_1) go to [the park](#cluster_3). [It](#cluster_3) was full of trees.
    ```
    """
    # instructions
    prompt = "Annotate all entity mentions in the following text with coreference clusters. Use Markdown tags to indicate clusters in the output, with the following format [mention](#cluster_name)\n\n"

    # add example itself
    prompt += "Input: {0}\nOutput:".format(example["input_context_str"])
    prompt += " " + example["output_priming"]

    return prompt


def qa_template(example: dict) -> str:
    """Example prompt instantiated from this template:
    ```
    Please carefully read the following passages. For each passage, you must identify
    which noun the mention marked in *bold* refers to.

    Passage: [Tom] and [Mary] go to [the park]. *It* was full of trees.
    Question: In the above passage, what does *It* refer to?
    Answer: *It* refers to [the park]
    ```
    """
    # instructions
    prompt = "Please carefully read the following passages. For each passage, you must identify which noun the mention marked in *bold* refers to.\n\n"

    # add example itself
    prompt += "Passage: {0}\nQuestion: In the above passage, what does {1} refer to?\nAnswer: {1} refers to ".format(
        example["context_str"],
        example["anaphor_str"],
    )

    return prompt

import os
import torch
import time

import openai
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# from peft import PeftModel, PeftConfig

from utils.io_utils import *


def get_generations(prompts: dict, exp_dir: str, model_id: str) -> dict:

    # run generation
    generation_filepath = os.path.join(exp_dir, "generations.json")
    model = MODEL_TYPE[model_id](model_id)
    if isinstance(model, OpenAIModel):
        model.compute_cost(prompts)
    generations = model.inference(prompts, generation_filepath)

    return generations


class HFModels:
    """Wrapper for HuggingFace Models (e.g. Llama)"""

    def __init__(
        self, model_name: str, max_generated_len: int, max_context_len: int = 2048
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            # load_in_8bit=True,
            # bnb_8bit_quant_type="nf8",
            # bnb_8bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map=device,
            device_map="auto",
            # torch_dtype=torch.float16,
            load_in_8bit=True,
            rope_scaling={"type": "dynamic", "factor": 2},
            quantization_config=bnb_config,
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.max_generated_len = max_generated_len  # TODO: this is obsolete
        self.max_context_len = max_context_len

    def inference(self, prompts: dict, generation_filepath: str) -> dict:

        # resume generation if exist
        generations = dict()
        if os.path.exists(generation_filepath):
            generations = read_json(generation_filepath)

        for e_key, prompt in prompts.items():

            if e_key not in generations:

                # try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_context_len - input_ids.shape[1],
                    # temperature=0.9,
                    do_sample=False,  # greedy decoding
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=13,
                )
                input_len = input_ids.shape[1]
                # generated tokens
                generated_tokens = outputs.sequences[0, input_len:-1].tolist()
                # generated_tokens = outputs.sequences[0].tolist()
                generated_text = self.tokenizer.decode(generated_tokens)
                try:  # because of some annoying encoding bug
                    print("Finished Prompt: {0}{1}".format(prompt, generated_text))
                except:
                    continue
                generations[e_key] = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                }
                write_json(generations, generation_filepath)
                # except:
                # print("Cannot generate text for example={0}".format(example_key))

        write_json(generations, generation_filepath)
        return generations


class OpenAIModel:
    """Wrapper for OpenAI Models (eg gpt-35, gpt-4)"""

    # per-token pricing https://openai.com/api/pricing/ snapshot on 11/09/2023
    MODEL_INFO = {
        "gpt-4": {
            "max_context_len": 8000,
            "input_cost": 0.00003,  # $0.03 per 1K tokens
            "output_cost": 0.00006,  # $0.06 per 1K tokens
        },
        "gpt-4-32k": {
            "max_context_len": 32000,
            "input_cost": 0.00006,  # $0.06 per 1K tokens
            "output_cost": 0.00012,  # $0.12 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-16k": {
            "max_context_len": 16000,
            "input_cost": 0.000003,  # $0.003 per 1K tokens
            "output_cost": 0.000004,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-instruct": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
    }

    def __init__(self, model_name: str) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_info = OpenAIModel.MODEL_INFO[model_name]
        self.max_context_len = self.model_info["max_context_len"]

    def inference(self, prompts: dict, generation_filepath: str) -> dict:

        openai.api_key = os.environ["OPENAI_API_KEY"]

        # resume generation if exist
        generations = dict()
        if os.path.exists(generation_filepath):
            generations = read_json(generation_filepath)

        for e_key, prompt in prompts.items():

            if e_key not in generations:

                try:
                    max_generated_len = self.max_context_len - len(
                        self.tokenizer.encode(prompt)
                    )

                    if self.model_name in ["gpt-3.5-turbo-instruct"]:
                        completion = openai.Completion.create(
                            engine=self.model_name,
                            prompt=prompt,
                            max_tokens=max_generated_len,
                            temperature=0,
                        )
                        output_text = completion.choices[0].text
                    else:
                        completion = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[
                                {
                                    "role": "assistant",
                                    "content": prompt,
                                },
                            ],
                            max_tokens=max_generated_len,
                            temperature=0,
                        )
                        output_text = completion.choices[0].message["content"]
                    print("Finished Prompt: {0}{1}".format(prompt, output_text))
                    generations[e_key] = {
                        "prompt": prompt,
                        "generated_text": output_text,
                    }
                    write_json(generations, generation_filepath)

                    if "gpt-4" in self.model_name:
                        time.sleep(60)

                except:
                    print("Cannot generate text for example={0}".format(e_key))

        write_json(generations, generation_filepath)
        return generations

    def compute_cost(self, prompts: dict):
        """Estimate the amount of $ it takes to run this experiment =
        $/token x sum over all prompts of (# of tokens/prompt)
        """
        estimated_cost = 0
        for key, prompt in prompts.items():

            # tokenize and get tokens for this input
            num_input_tokens = len(self.tokenizer.encode(prompt))
            max_output_tokens = self.max_context_len - num_input_tokens
            estimated_cost += (
                self.model_info["input_cost"] * num_input_tokens
                + self.model_info["output_cost"] * max_output_tokens
            )

        # output the cost
        print(f"Estimated cost for {self.model_name}: ${estimated_cost:0.2f}")


class AzureModel:
    """Wrapper for Azure Models (eg gpt-35, gpt-4)
    TODO: refactor into OpenAI when released
    """

    # per-token pricing https://openai.com/api/pricing/ snapshot on 11/09/2023
    MODEL_INFO = {
        "gpt-4": {
            "max_context_len": 8000,
            "input_cost": 0.00003,  # $0.03 per 1K tokens
            "output_cost": 0.00006,  # $0.06 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "max_context_len": 4000,
            "input_cost": 0.0000015,  # $0.003 per 1K tokens
            "output_cost": 0.000002,  # $0.004 per 1K tokens
        },
        "gpt-3.5-turbo-16k": {
            "max_context_len": 16000,
            "input_cost": 0.000003,  # $0.003 per 1K tokens
            "output_cost": 0.000004,  # $0.004 per 1K tokens
        },
    }

    def __init__(self, model_name: str, max_generated_len: int) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_info = OpenAIModel.MODEL_INFO[model_name]
        self.max_context_len = self.model_info["max_context_len"]
        self.max_generated_len = max_generated_len  # TODO I think this is deprecated

    def inference(self, prompts: dict, generation_filepath: str) -> dict:

        DEPLOYMENT_NAME = "coref"

        # resume generation if exist
        generations = dict()
        if os.path.exists(generation_filepath):
            generations = read_json(generation_filepath)

        for e_key, prompt in prompts.items():

            if e_key not in generations:

                try:
                    max_generated_len = self.max_context_len - len(
                        self.tokenizer.encode(prompt)
                    )

                    completion = openai.ChatCompletion.create(
                        engine=DEPLOYMENT_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        max_tokens=max_generated_len,
                        temperature=0,
                    )
                    output_text = completion.choices[0].message["content"]
                    print("Finished Prompt: {0}{1}".format(prompt, output_text))
                    generations[e_key] = {
                        "prompt": prompt,
                        "generated_text": output_text,
                    }
                    write_json(generations, generation_filepath)

                except:
                    print("Cannot generate text for example={0}".format(e_key))

        write_json(generations, generation_filepath)
        return generations

    def compute_cost(
        self,
        prompts: dict,
    ):
        """Estimate the amount of $ it takes to run this experiment =
        $/token x sum over all prompts of (# of tokens/prompt)
        """
        estimated_cost = 0
        for key, prompt in prompts.items():

            # tokenize and get tokens for this input
            num_input_tokens = len(self.tokenizer.encode(prompt))
            max_output_tokens = self.max_context_len - num_input_tokens
            estimated_cost += (
                self.model_info["input_cost"] * num_input_tokens
                + self.model_info["output_cost"] * max_output_tokens
            )

        # output the cost
        print(f"Estimated cost for {self.model_name}: ${estimated_cost:0.2f}")


MODEL_TYPE = {
    "llama": HFModels,
    "llama-2": HFModels,
    "codellama": HFModels,
    "gpt-3.5-turbo": OpenAIModel,
    "gpt-3.5-turbo-16k": OpenAIModel,
    "gpt-3.5-turbo-instruct": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-4-32k": OpenAIModel,
}

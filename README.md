# Are Large Language Models Robust Coreference Resolvers? 

![image](./approach.pdf)

## Setup 
1. Create conda environment
```
conda create -n coref-llms python==3.8
conda activate coref-llms
pip install -r requirements.txt
```

2. Set up PATH and OpenAI API key (with the environment variable OPENAI_API_KEY)
```
export PYTHONPATH=.
export OPENAI_API_KEY=/your/openai/api_key
```

## Download Data 
We follow this repo to pre-process the raw coref data into jsonlines files: https://github.com/shtoshni/fast-coref

## Run Code 
```
python src/main.py \
	--exp_dir [experiment directory] \
	--eval_data ./data/example.jsonl \
	--model_id [id of models to be evaluated, e.g. `gpt-4`] \
	--prompt_template [either `doc_template` or `qa_template`] \
```

As an example (taken from WikiCoref dataset), you can generate the coreference annotation as follows 
```
python src/main.py \
	--exp_dir ./test \
	--eval_data ./data/example.jsonl \
	--model_id gpt-4 \
	--prompt_template doc_template
```

## Citations
```
@misc{le2023large,
      title={Are Large Language Models Robust Coreference Resolvers?}, 
      author={Nghia T. Le and Alan Ritter},
      year={2023},
      eprint={2305.14489},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import time
import concurrent.futures

import tiktoken
import shortuuid
import tqdm

from utils import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_openai_azure,
    chat_completion_mistral,
    http_completion_gemini,
    chat_completion_cohere,
    chat_completion_together_ai,
    chat_completion_huggingface,
    generate_MoA_response,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)

import transformers
from transformers import AutoTokenizer, GenerationConfig
import torch

##################################################

def load_HF_pipeline(model_path: str, max_new_tokens: int):

        model_id = model_path
        model = model_path
    
        if model == "microsoft/Phi-3-small-8k-instruct":
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                #model_kwargs={"torch_dtype": torch.bfloat16} if model == "meta-llama/Meta-Llama-3-8B-Instruct" else {"torch_dtype": "auto"},
                #model_kwargs={"torch_dtype": "auto"},
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )

        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                     "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            pipeline.tokenizer.padding_side = 'left'

        pipeline.model.config.is_encoder_decoder = False

        ########################################

        terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        ########################################

        generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            model_id, 
            return_unused_kwargs=True
        )

        generation_config.batch_size = 1
        
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = True
        #generation_config.temperature = temperature
        generation_config.top_p = 0.9
        generation_config.num_return_sequences = 1
        generation_config.is_encoder_decoder = False
        generation_config.eos_token_id = terminators if model in ["meta-llama/Meta-Llama-3-8B-Instruct"] else pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                     "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            generation_config.pretraining_tp = 1
        
        pipeline.model.config = generation_config

        return pipeline, generation_config

##################################################

def search_string_in_jsonl(file_path, search_string):
    if not os.path.exists(file_path):
        return False
    
    found = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if search_string in line:
                found = True
                break
                #print(f"Found the string in line: {line.strip()}")
    #if not found:
     #   print(f"The string '{search_string}' was not found in the file.")
    return found

def get_answer(
    question: dict, model: str, endpoint_info: dict, num_choices: int, max_tokens: int, temperature: float, answer_file: str, api_dict: dict, 
    pipeline=None, generation_config=None
):
    
    if search_string_in_jsonl(answer_file, question["question_id"]):
        return
    
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    api_type = endpoint_info["api_type"]

    conv = []

    if "system_prompt" in endpoint_info.keys():
        conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
    elif model in OPENAI_MODEL_LIST:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    ################################################

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    #total_model_to_outputs_dict = []
    for i in range(num_choices):
        turns = []
        #current_model_to_outputs_dicts = []
        for j in range(len(question["turns"])):
            conv.append({"role": "user", "content": question["turns"][j]["content"]})
            if api_type == "anthropic":
                output = chat_completion_anthropic(model=endpoint_info["model_name"],
                                                   messages=conv,
                                                   temperature=temperature,
                                                   max_tokens=max_tokens)
            elif api_type == "mistral":
                output = chat_completion_mistral(model=endpoint_info["model_name"],
                                                 messages=conv,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)
            elif api_type == "gemini":
                output = http_completion_gemini(model=endpoint_info["model_name"],
                                                message=question["turns"][j]["content"],
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            elif api_type == "azure":
                output = chat_completion_openai_azure(model=endpoint_info["model_name"],
                                                      messages=conv,
                                                      temperature=temperature,
                                                      max_tokens=max_tokens,
                                                      api_dict=api_dict)
            elif api_type == "cohere":
                output = chat_completion_cohere(model=endpoint_info["model_name"],
                                                messages=conv,
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            elif api_type == "together_ai":
                output = chat_completion_together_ai(model=endpoint_info["model_name"],
                                                     #models=endpoint_info["models"],
                                                     candidate_count=endpoint_info["candidate_count"],
                                                     messages=conv,
                                                     temperature=endpoint_info["temperature"],
                                                     max_tokens=max_tokens)
            elif api_type == "huggingface":

                generation_config.temperature = temperature if temperature != 0.0 else 0.7
                output = chat_completion_huggingface(messages=conv,
                                                     pipeline=pipeline, 
                                                     generation_config=generation_config)
                

            elif api_type == "MoA":

                output = generate_MoA_response(aggregator_model=endpoint_info["aggregator_model"],
                                               reference_models=endpoint_info["reference_models"],
                                               rounds=endpoint_info["rounds"],
                                               messages=conv,
                                               temperature=temperature,
                                               max_tokens=max_tokens)
                                               #generate_fn=generate_together)

                #print(f"Instruction: {question['turns'][j]['content']}")
                #print(f"Output: {output}")
                
            else:
                output = chat_completion_openai(model=endpoint_info["model_name"], 
                                                messages=conv, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                api_dict=api_dict)
            conv.append({"role": "assistant", "content": output})
            turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})
            #current_model_to_outputs_dicts.append(model_to_outputs_dict)

        choices.append({"index": i, "turns": turns})
        #total_model_to_outputs_dict.append(current_model_to_outputs_dicts)

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
        "question": [question["turns"][idx]["content"] for idx in range(len(question["turns"]))],
        #"model_to_outputs_dict": total_model_to_outputs_dict
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    args = parser.parse_args()

    settings = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    existing_answer = load_model_answers(os.path.join("data", settings["bench_name"], "model_answer"))
    
    print(settings)

    for model in settings["model_list"]:
        assert model in endpoint_list
        endpoint_info = endpoint_list[model]

        question_file = os.path.join("data", settings["bench_name"], "question.jsonl")
        questions = load_questions(question_file)

        if len(model.split("/")) >= 2:
            model_name = model.split("/")[1]
            answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model_name}.jsonl")
        else:
            answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model}.jsonl")
        print(f"Output to {answer_file}")

        if "parallel" in endpoint_info:
            parallel = endpoint_info["parallel"]
        else:
            parallel = 1

        # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
        if "tokenizer" in endpoint_info:
            question_list = [question["turns"][0]["content"] for question in questions]
            if model in OPENAI_MODEL_LIST:
                tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
                tokens = [tokenizer.encode(prompt) for prompt in question_list]
                max_tokens = [(settings["max_tokens"] - len(token) - 100) for token in tokens]
            else:
                from transformers import AutoTokenizer
                
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])

                tokens = tokenizer(question_list)
                max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
        else:
            max_tokens = [settings["max_tokens"]] * len(questions)

        ################################################

        if endpoint_info["api_type"] == "huggingface": 
            print(f"Max Tokens: {max_tokens[0]}")
            pipeline, generation_config = load_HF_pipeline(endpoint_info["model_name"], max_tokens[0]) 
        else:
            pipeline, generation_config = None, None

        ################################################

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            count = 0
            for index, question in enumerate(questions):
                if model in existing_answer and question["question_id"] in existing_answer[model]:
                    count += 1
                    continue
                future = executor.submit(
                    get_answer,
                    question,
                    model,
                    endpoint_info,
                    settings["num_choices"],
                    max_tokens[index],
                    settings["temperature"],
                    answer_file,
                    get_endpoint(endpoint_info["endpoints"]),
                    pipeline=pipeline,
                    generation_config=generation_config
                )
                futures.append(future)
            if count > 0:
                print(f"{count} number of existing answers")
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

        reorg_answer_file(answer_file)

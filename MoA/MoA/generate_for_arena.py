# """Generate answers using MoA for Arena-Hard-Auto

# Usage:
# python gen_moa_answer.py --parallel 32
# """
# import argparse
# import json
# import os
# import time
# import concurrent.futures

# import tiktoken
# import shortuuid
# import tqdm
# from loguru import logger
# from typing import List

# from arena_hard_auto.utils import (
#     load_questions,
#     load_model_answers,
#     make_config,
#     get_endpoint,
#     chat_completion_openai,
#     chat_completion_anthropic,
#     chat_completion_openai_azure,
#     chat_completion_mistral,
#     http_completion_gemini,
#     chat_completion_cohere,
#     reorg_answer_file,
#     OPENAI_MODEL_LIST,
#     temperature_config,
# )

# from utils import (
#     generate_together,
#     generate_together_stream,
#     generate_openai,
#     inject_references_to_messages,
#     generate_with_references,
#     DEBUG,
# )


# def get_answer(
#     question: dict, model: str, endpoint_info: dict, reference_models: List[str], num_choices: int, max_tokens: int, temperature: float, answer_file: str, api_dict: dict, rounds: int, provider: str
# ):
#     if question["category"] in temperature_config:
#         temperature = temperature_config[question["category"]]

#     api_type = endpoint_info["api_type"]
#     conv = []

#     if "system_prompt" in endpoint_info.keys():
#         conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
#     elif model in OPENAI_MODEL_LIST:
#         conv.append({"role": "system", "content": "You are a helpful assistant."})

#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     choices = []

#     for i in range(num_choices):
#         turns = []
#         messages = []

#         for j in range(len(question["turns"])):
#             qs = question["turns"][j]["content"]
#             messages.append({"role": "user", "content": qs})

#             references = []

#             if len(reference_models) > 0:
#                 prev_references = []

#                 for i_round in range(rounds):
#                     if DEBUG:
#                         logger.info(
#                             f"Round {i_round+1}/{rounds} to collecting reference responses."
#                         )

#                     references = []

#                     for reference_model in reference_models:
#                         reference = generate_with_references(
#                             model=reference_model,
#                             messages=messages,
#                             references=prev_references,
#                             temperature=temperature,
#                             max_tokens=max_tokens,
#                             generate_fn=generate_together,
#                         )
#                         if reference:
#                             references.append(reference)

#                     if i_round < rounds - 1:
#                         prev_references = references

#             if api_type == "anthropic":
#                 output = chat_completion_anthropic(
#                     model=endpoint_info["model_name"],
#                     messages=conv,
#                     temperature=temperature,
#                     max_tokens=max_tokens
#                 )
#             elif api_type == "mistral":
#                 output = chat_completion_mistral(
#                     model=endpoint_info["model_name"],
#                     messages=conv,
#                     temperature=temperature,
#                     max_tokens=max_tokens
#                 )
#             elif api_type == "gemini":
#                 output = http_completion_gemini(
#                     model=endpoint_info["model_name"],
#                     message=qs,
#                     temperature=temperature,
#                     max_tokens=max_tokens
#                 )
#             elif api_type == "azure":
#                 output = chat_completion_openai_azure(
#                     model=endpoint_info["model_name"],
#                     messages=conv,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     api_dict=api_dict
#                 )
#             elif api_type == "cohere":
#                 output = chat_completion_cohere(
#                     model=endpoint_info["model_name"],
#                     messages=conv,
#                     temperature=temperature,
#                     max_tokens=max_tokens
#                 )
#             else:
#                 output = chat_completion_openai(
#                     model=endpoint_info["model_name"],
#                     messages=conv,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     api_dict=api_dict
#                 )

#             if output is None:
#                 print(f"Warning: No output for model {model} on question {question['question_id']}")
#                 continue

#             output = output.strip()

#             conv.append({"role": "assistant", "content": output})
#             turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})

#         choices.append({"index": i, "turns": turns})

#     # Dump answers
#     ans = {
#         "question_id": question["question_id"],
#         "answer_id": shortuuid.uuid(),
#         "model_id": model,
#         "choices": choices,
#         "tstamp": time.time(),
#     }

#     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#     with open(answer_file, "a") as fout:
#         fout.write(json.dumps(ans) + "\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True, help="The model to use for generating answers.")
#     parser.add_argument("--answer-file", type=str, required=True, help="The output answer file.")
#     parser.add_argument("--parallel", type=int, default=1, help="The number of concurrent API calls.")
#     parser.add_argument("--setting-file", type=str, default="arena_hard_auto/config/gen_answer_config.yaml")
#     parser.add_argument("--endpoint-file", type=str, default="arena_hard_auto/config/api_config.yaml")
#     parser.add_argument("--reference-models", type=str, default=None, help="Comma-separated list of reference models.")
#     parser.add_argument("--rounds", type=int, default=1, help="Number of rounds for reference model generation.")
#     parser.add_argument("--provider", type=str, default="together", help="API provider: 'together' or 'openai'.")
#     args = parser.parse_args()

#     settings = make_config(args.setting_file)
#     endpoint_list = make_config(args.endpoint_file)
#     existing_answer = load_model_answers(os.path.join("data", settings["bench_name"], "model_answer"))

#     print(settings)

#     for model in settings["model_list"]:
#         assert model in endpoint_list
#         endpoint_info = endpoint_list[model]

#         question_file = os.path.join("arena_hard_auto/data", settings["bench_name"], "question.jsonl")
#         questions = load_questions(question_file)

#         answer_file = os.path.join("outputs", settings["bench_name"], "model_answer", f"{model}.jsonl")

#         print(f"Output to {answer_file}")

#         if "parallel" in endpoint_info:
#             parallel = endpoint_info["parallel"]
#         else:
#             parallel = 1

#         # We want to maximize the number of tokens generated per answer: max_tokens = specified token # - input tokens #
#         if "tokenizer" in endpoint_info:
#             question_list = [question["turns"][0]["content"] for question in questions]
#             if model in OPENAI_MODEL_LIST:
#                 tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
#                 tokens = [tokenizer.encode(prompt) for prompt in question_list]
#                 max_tokens = [(settings["max_tokens"] - len(token) - 100) for token in tokens]
#             else:
#                 from transformers import AutoTokenizer
#                 os.environ["TOKENIZERS_PARALLELISM"] = "false"
#                 tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])
#                 tokens = tokenizer(question_list)
#                 max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
#         else:
#             max_tokens = [settings["max_tokens"]] * len(questions)

#         reference_models = args.reference_models.split(",") if args.reference_models else []

#         with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
#             futures = []
#             count = 0
#             for index, question in enumerate(questions):
#                 if model in existing_answer and question["question_id"] in existing_answer[model]:
#                     count += 1
#                     continue
#                 future = executor.submit(
#                     get_answer,
#                     question,
#                     model,
#                     endpoint_info,
#                     reference_models,
#                     settings["num_choices"],
#                     max_tokens[index],
#                     settings["temperature"],
#                     answer_file,
#                     get_endpoint(endpoint_info["endpoints"]),
#                     args.rounds,
#                     args.provider,
#                 )
#                 futures.append(future)
#             if count > 0:
#                 print(f"{count} number of existing answers")
#             for future in tqdm.tqdm(
#                 concurrent.futures.as_completed(futures), total=len(futures)
#             ):
#                 future.result()

#         reorg_answer_file(answer_file)











# FROM GITHUB: WORKS 
# """Generate answers using api endpoints.

# Usage:
# python gen_api_answer --parallel 32
# """
# import argparse
# import json
# import os
# import time
# import concurrent.futures

# import tiktoken
# import shortuuid
# import tqdm

# from arena_hard_auto.utils import (
#     load_questions,
#     load_model_answers,
#     make_config,
#     get_endpoint,
#     chat_completion_openai,
#     chat_completion_anthropic,
#     chat_completion_openai_azure,
#     chat_completion_mistral,
#     http_completion_gemini,
#     chat_completion_cohere,
#     reorg_answer_file,
#     OPENAI_MODEL_LIST,
#     temperature_config,
# )

# from utils import (
#     generate_together,
#     generate_together_stream,
#     generate_openai,
#     inject_references_to_messages,
#     generate_with_references,
#     DEBUG,
# )


# def get_answer(
#     question: dict, model: str, endpoint_info: dict, num_choices: int, max_tokens: int, temperature: float, answer_file: str, api_dict: dict
# ):
#     if question["category"] in temperature_config:
#         temperature = temperature_config[question["category"]]

#     api_type = endpoint_info["api_type"]

#     conv = []

#     if "system_prompt" in endpoint_info.keys():
#         conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
#     elif model in OPENAI_MODEL_LIST:
#         conv.append({"role": "system", "content": "You are a helpful assistant."})

#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     choices = []
#     for i in range(num_choices):
#         turns = []
#         for j in range(len(question["turns"])):
#             conv.append({"role": "user", "content": question["turns"][j]["content"]})
#             if api_type == "anthropic":
#                 output = chat_completion_anthropic(model=endpoint_info["model_name"],
#                                                    messages=conv,
#                                                    temperature=temperature,
#                                                    max_tokens=max_tokens)
#             elif api_type == "mistral":
#                 output = chat_completion_mistral(model=endpoint_info["model_name"],
#                                                  messages=conv,
#                                                  temperature=temperature,
#                                                  max_tokens=max_tokens)
#             elif api_type == "gemini":
#                 output = http_completion_gemini(model=endpoint_info["model_name"],
#                                                 message=question["turns"][j]["content"],
#                                                 temperature=temperature,
#                                                 max_tokens=max_tokens)
#             elif api_type == "azure":
#                 output = chat_completion_openai_azure(model=endpoint_info["model_name"],
#                                                       messages=conv,
#                                                       temperature=temperature,
#                                                       max_tokens=max_tokens,
#                                                       api_dict=api_dict)
#             elif api_type == "cohere":
#                 output = chat_completion_cohere(model=endpoint_info["model_name"],
#                                                 messages=conv,
#                                                 temperature=temperature,
#                                                 max_tokens=max_tokens)
#             else:
#                 output = chat_completion_openai(model=endpoint_info["model_name"], 
#                                                 messages=conv, 
#                                                 temperature=temperature, 
#                                                 max_tokens=max_tokens, 
#                                                 api_dict=api_dict)
#             conv.append({"role": "assistant", "content": output})

#             turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})
#         choices.append({"index": i, "turns": turns})

#     # Dump answers
#     ans = {
#         "question_id": question["question_id"],
#         "answer_id": shortuuid.uuid(),
#         "model_id": model,
#         "choices": choices,
#         "tstamp": time.time(),
#     }

#     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#     with open(answer_file, "a") as fout:
#         fout.write(json.dumps(ans) + "\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--setting-file", type=str, default="arena_hard_auto/config/gen_answer_config.yaml"
#     )
#     parser.add_argument(
#         "--endpoint-file", type=str, default="arena_hard_auto/config/api_config.yaml"
#     )
#     args = parser.parse_args()

#     settings = make_config(args.setting_file)
#     endpoint_list = make_config(args.endpoint_file)

#     existing_answer = load_model_answers(os.path.join("data", settings["bench_name"], "model_answer"))
    
#     print(settings)

#     for model in settings["model_list"]:
#         assert model in endpoint_list
#         endpoint_info = endpoint_list[model]

#         question_file = os.path.join("arena_hard_auto/data", settings["bench_name"], "question.jsonl")
#         questions = load_questions(question_file)

#         answer_file = os.path.join("arena_hard_auto/data", settings["bench_name"], "model_answer", f"{model}.jsonl")
#         print(f"Output to {answer_file}")

#         if "parallel" in endpoint_info:
#             parallel = endpoint_info["parallel"]
#         else:
#             parallel = 1

#         # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
#         if "tokenizer" in endpoint_info:
#             question_list = [question["turns"][0]["content"] for question in questions]
#             if model in OPENAI_MODEL_LIST:
#                 tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
#                 tokens = [tokenizer.encode(prompt) for prompt in question_list]
#                 max_tokens = [(settings["max_tokens"] - len(token) - 100) for token in tokens]
#             else:
#                 from transformers import AutoTokenizer
                
#                 os.environ["TOKENIZERS_PARALLELISM"] = "false"
#                 tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])

#                 tokens = tokenizer(question_list)
#                 max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
#         else:
#             max_tokens = [settings["max_tokens"]] * len(questions)

#         with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
#             futures = []
#             count = 0
#             for index, question in enumerate(questions):
#                 if model in existing_answer and question["question_id"] in existing_answer[model]:
#                     count += 1
#                     continue
#                 future = executor.submit(
#                     get_answer,
#                     question,
#                     model,
#                     endpoint_info,
#                     settings["num_choices"],
#                     max_tokens[index],
#                     settings["temperature"],
#                     answer_file,
#                     get_endpoint(endpoint_info["endpoints"]),
#                 )
#                 futures.append(future)
#             if count > 0:
#                 print(f"{count} number of existing answers")
#             for future in tqdm.tqdm(
#                 concurrent.futures.as_completed(futures), total=len(futures)
#             ):
#                 future.result()

#         reorg_answer_file(answer_file)





"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

# from fastchat.llm_judge.common import load_questions, temperature_config
# from fastchat.model import load_model, get_conversation_template
# from fastchat.utils import str_to_torch_dtype

# import openai
import requests
from loguru import logger

import argparse
import json
import os
import time
import concurrent.futures

import tiktoken
import shortuuid
import tqdm

from arena_hard_auto.utils import (
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
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)

from utils import (
    generate_together,
    generate_together_stream,
    generate_openai,
    inject_references_to_messages,
    generate_with_references,
    DEBUG,
)


"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

# from fastchat.llm_judge.common import (
#     load_questions,
#     temperature_config,
#     chat_completion_openai,
#     chat_completion_anthropic,
#     chat_completion_palm,
# )
# from fastchat.llm_judge.gen_model_answer import reorg_answer_file
# from fastchat.model.model_adapter import get_conversation_template

from typing import List
from loguru import logger
import openai

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    DEBUG,
)


def get_answer(
    question: dict,
    model: str,
    reference_models: List[str],
    num_choices: int,
    max_tokens: int,
    answer_file: str,
    rounds: int,
    provider: str,
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []

    if provider == "together":
        generate_fn = generate_together
    elif provider == "openai":
        generate_fn = generate_openai
    else:
        assert False

    for i in range(num_choices):
        turns = []
        messages = []

        for j in range(len(question["turns"])):
            # Check if the item in 'turns' is a string or a dictionary with 'content'
            if isinstance(question["turns"][j], dict):
                qs = question["turns"][j]['content']  # Accessing content if it's a dictionary
            else:
                qs = question["turns"][j]  # Directly using the string

            messages.append({"role": "user", "content": qs})

            references = []

            if len(reference_models) > 0:
                prev_references = []

                for i_round in range(rounds):
                    if DEBUG:
                        logger.info(f"Round {i_round+1}/{rounds} to collecting reference responses.")

                    references = []

                    for reference_model in reference_models:
                        reference = generate_with_references(
                            model=reference_model,
                            messages=messages,
                            references=prev_references,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            generate_fn=generate_fn,
                        )

                        if reference is not None:
                            references.append(reference)

                    if i_round < rounds - 1:
                        prev_references = references

            output = generate_with_references(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                generate_fn=generate_fn,
                references=references,
            ).strip()

            messages.append({"role": "assistant", "content": output})
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="arena-hard-v0.1",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-72B-Instruct")
    parser.add_argument("--reference-models", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    question_file = f"arena_hard_auto/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"arena_hard_auto/data/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    if args.reference_models is None:
        reference_models = []
    else:
        reference_models = args.reference_models.split(",")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                reference_models,
                args.num_choices,
                args.max_tokens,
                answer_file,
                args.rounds,
                args.provider,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)

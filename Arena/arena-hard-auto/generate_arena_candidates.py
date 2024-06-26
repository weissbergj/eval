

import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification 
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, GenerationConfig
from datasets import load_dataset, load_from_disk
import transformers
from transformers.pipelines.pt_utils import KeyDataset

import pandas as pd
from itertools import combinations, permutations
from typing import List

import random
from datasets import Dataset, DatasetDict
from datetime import datetime
import sys
import os
import logging
import random
random.seed(43) #43
from tabulate import tabulate
import openai
import time
from tqdm import tqdm
import anthropic
from safetensors.torch import load_model, save_model 
import shutil
import json
import subprocess

from datasets import Dataset, concatenate_datasets

#################################################

# Parameters

# Set the models and their configs in:
# - api_config.yaml
# - gen_answer_config.yaml
# - judge_config.yaml 

#################################################

# Ensembling Model Selection

# Mixture of Agents Models
#models = ["Qwen/Qwen1.5-72B-Chat", "Qwen/Qwen1.5-110B-Chat", "microsoft/WizardLM-2-8x22B",
#          "mistralai/Mixtral-8x22B-Instruct-v0.1", "meta-llama/Llama-3-70b-chat-hf", "databricks/dbrx-instruct"]

models = ["Qwen/Qwen1.5-7B-Chat","meta-llama/Meta-Llama-3-8B-Instruct", "Nexusflow/Starling-LM-7B-beta", 
          "berkeley-nest/Starling-LM-7B-alpha", "teknium/OpenHermes-2.5-Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.2",
          "cognitivecomputations/dolphin-2.2.1-mistral-7b", "microsoft/Phi-3-mini-4k-instruct", #"upstage/SOLAR-10.7B-Instruct-v1.0",
          "HuggingFaceH4/zephyr-7b-beta", "microsoft/Phi-3-small-8k-instruct"]

# Ensembling Parameters
perform_ensembling = False
ranker_config = {
    "ranker_checkpoint": "llm-blender/PairRM",

    "ranker_model": "microsoft/deberta-v3-large",
    "ranker_max_length": 1024, #512, 1024
    "ranker_batch_size": 16, #32
    "source_max_length": 256, # 128, 256
    "candidate_max_length": 256, # 128, 256
    "device": "cuda:0"
}

# Check if dataset already exists
ensemble_model_id = "ensemble_v2"
final_dataset_path = f"data/arena-hard-v0.1/model_answer/{ensemble_model_id}.jsonl"
if os.path.exists(final_dataset_path):
    print("Ensemble Dataset Already Exists: ", final_dataset_path)
    sys.exit()

#################################################

if not perform_ensembling:
    
    candidate_generation_command = "python gen_answer.py"

    print("Generation Command: ", candidate_generation_command)
    print("Generating candidates...")
    with subprocess.Popen(candidate_generation_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end='')  # Print the output in real-time

    ##########################################

    judgement_command = "python gen_judgment.py"
    print("Generating judgements...")
    with subprocess.Popen(judgement_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end='')  # Print the output in real-time

    ##########################################

    show_results_command = "python show_result.py"
    print("Showing results...")
    show_results_result = subprocess.run(show_results_command, shell=True, capture_output=True, text=True)

    print("------------------------------------------------")
    print("Arena-Hard-Auto Results:")
    for line in show_results_result.stdout.split("\n"):
        print(line)
    print("------------------------------------------------")

else:

    print("Performing ensembling for Arena-Hard-Auto...")

    #################################################

    # Load datasets
    total_datasets = []
    print("Loading Models...")
    for model_name in models:
        model_id = model_name.split("/")[1]
        print(f"Loading model: {model_id}")
        saved_jsonl_path = f"data/arena-hard-v0.1/model_answer/{model_id}.jsonl"
        dataset = pd.read_json(saved_jsonl_path, lines=True)
        total_datasets.append(dataset)

    #################################################

    # Gather candidates

    instructions = [instruction[0] for instruction in total_datasets[0]["question"].tolist()]
    ensemble_candidates = []

    for row_idx in range(len(total_datasets[0])):
        candidates = []
        question_id = total_datasets[0].iloc[row_idx]["question_id"]
        for dataset in total_datasets:
            assert dataset.iloc[row_idx]["question_id"] == question_id
            current_choices = dataset["choices"].iloc[row_idx]
            assert len(current_choices["turns"]) == 1
            current_candidates = [choice["turns"][0]["content"] for choice in current_choices]
            candidates.append(current_candidates)
        ensemble_candidates.append(candidates)

    #################################################

    # Perform ranking over candidates
    import llm_blender
    blender = llm_blender.Blender()
    blender.loadranker(ranker_config['ranker_checkpoint'])

    #################################################

    breakpoint()

    # Score first turn candidates
    assert len(instructions) == len(ensemble_candidates)
    print("Performing Ensemble Candidate Ranking for First Turn Candidates with PairRM Ranker...")
    scores = blender.rank(instructions, ensemble_candidates, return_scores=True, batch_size=ranker_config['ranker_batch_size'])
    ranks = [sorted(range(len(score)), key=lambda i: score[i], reverse=True) for score in scores]
        
    assert len(ranks) == len(ensemble_candidates)
    top_candidate_texts_from_ranker = [ensemble_candidates[i][ranks[i][0]] for i in range(len(ranks))]

    #################################################

    # Create new ensemble JSONL

    ensemble_choices = []
    for idx in range(len(top_candidate_texts_from_ranker)):
        ensemble_choices.append([{
            "turns": {"idx": 0,
                      "content": top_candidate_texts_from_ranker[idx]}
        }])

    question_ids = dataset["question_id"].tolist()
    category_ids = dataset["category"].tolist()
    cluster_ids = dataset["cluster"].tolist()

    os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)
    with open(os.path.expanduser(final_dataset_path), "a") as fout:
        for question_id, category_id, model_id, cluster_id, choices in zip(question_ids, category_ids, cluster_ids, ensemble_choices):
            ans_json = {
                "question_id": question_id,
                "category": category_id,
                "model_id": model_id,
                "cluster_id": cluster_id,
                "choices": choices,
            }
            fout.write(json.dumps(ans_json) + "\n")

    #################################################

    print("Ensemble Results Saved: ", final_dataset_path)
    print(" - Add the model to the config files: api_config.yaml, gen_answer_config.yaml, judge_config.yaml")
    print(" - Run the following commands to generate judgements and show results:")
    print("python gen_judgment.py")
    print("python show_result.py")
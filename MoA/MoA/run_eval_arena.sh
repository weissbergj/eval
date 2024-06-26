export DEBUG=1

# Generate answers for Arena using the MoA method
python generate_for_arena.py --model "Qwen/Qwen1.5-72B-Chat" \
    --reference-models "microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct" \
    --answer-file outputs/arena/arena-together-MoA-round1.jsonl \
    --parallel 1 --rounds 1

# Evaluate the generated answers for Arena
python eval_arena.py --model-list arena-together-MoA-round1 --parallel 32

# Show results for Arena
python show_arena_result.py
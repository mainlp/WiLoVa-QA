# import pandas as pd
from tqdm import tqdm
import re
import pickle
# from deep_translator import GoogleTranslator
import os
import torch
import sys
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Conversation, Message, Role, SystemContent, ReasoningEffort

# =========================
# model-name mapping 
# Paths to local tokenizer and model directories
MODEL_PATHS = {
    "llama3_8b": {
        "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/meta-llama/Llama-3.1-8B-Instruct/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "model": "/nfs/gdata/llms/hf-models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "llama3_70b": {
        "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/meta-llama/Meta-Llama-3.1-70B-Instruct/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b",
        "model": "/nfs/gdata/llms/hf-models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b",
    },
    "qwen2.5_7b": {
        "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/Qwen/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "model": "/nfs/gdata/llms/hf-models/model/Qwen/Qwen2.5-7B-Instruct/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    },
    # "qwen2.5_14b": {
    #     "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/Qwen/Qwen2.5-14B-Instruct/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    #     "model": "/nfs/gdata/llms/hf-models/model/Qwen/Qwen2.5-14B-Instruct/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    # },
    "qwen2.5_72b": {
        "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/Qwen/Qwen2.5-72B-Instruct/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
        "model": "/nfs/gdata/llms/hf-models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
    },
    "gpt_oss_20b": {
        "tokenizer": "/nfs/gdata/llms/hf-models/tokenizer/openai--gpt-oss-20b/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb",
        "model": "/nfs/gdata/llms/hf-models/models--openai--gpt-oss-20b/models--openai--gpt-oss-20b/snapshots/d666cf3b67006cf8227666739edf25164aaffdeb",
    },
}

# parse CLI args to configure everything
# Usage example: python3 -u dialectqa_llm_judge.py 0 all_setting_prompts_zho.pkl gpt_oss_20b
# Usage example(distributed across multiple GPUs): python3 -u dialectqa_llm_judge.py 0,1,2,3 all_setting_prompts_zho.pkl gpt_oss_20b
gpu_id = sys.argv[1]
prompts_pkl = sys.argv[2]
model_name = sys.argv[3]

# Resolve paths using the selected model
tokenizer_path = MODEL_PATHS[model_name]["tokenizer"]  # NEW
model_path = MODEL_PATHS[model_name]["model"]          # NEW

# Tell vLLM to sleep when idle
os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
# set GPU via env from CLI arg
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id #select the GPU number n #masks all GPUs except GPU n.

if model_name in ['llama3_70b','qwen2.5_72b']:
    parallel_size = 4
else:
    # use only one GPU
    parallel_size = 1

print(f"Using {parallel_size} GPU(s), selected GPU number: {gpu_id}")

# import vLLM after setting CUDA_VISIBLE_DEVICES
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

#------
def build_llm_judge_prompt(question: str, ground_truth: str, generated_answer: str) -> str:
    return f"""
Your task is to evaluate whether a generated answer correctly answers the question, using the provided ground truth answer as reference.
Question:
{question}
Ground Truth Answer:
{ground_truth}
Generated Answer:
{generated_answer}
Please output a single capitalzied word (YES or NO) as evaluation result, without any additional explanation:
    - YES: the generated answer is correct according to the ground truth answer.
    - NO: the generated answer is incorrect according to the ground truth answer.
"""

def render_prompt_in_harmony_style(prompt):
    # initialize Harmony encoder once
    harmony_encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # system message with reasoning effort
    system = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.LOW)   # LOW / MEDIUM / HIGH
    )
    # build Harmony conversation
    harmony_conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.USER, prompt),
    ])
    # >render assistant prefill tokens
    prefill_ids = harmony_encoder.render_conversation_for_completion(harmony_conversation, Role.ASSISTANT)
    return TokensPrompt(prompt_token_ids=prefill_ids), harmony_encoder

if __name__ == '__main__':
    # load the prompt_messages
    with open(prompts_pkl, "rb") as f:
        all_setting_results_all_lang = pickle.load(f)
    print('Prompts loaded')

    # # Cap per-GPU memory so auto-placement doesn’t overpack GPU 0
    # max_memory = {
    #     0: "75GiB",   # a bit below physical VRAM
    #     1: "75GiB"
    # }

    # Initialize vLLM on your local snapshot
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,   # omit if tokenizer is in model dir
        dtype="auto", #automatically choose the compute precision (data type) based on your GPU and model
        tensor_parallel_size=parallel_size,
        max_model_len=5000,       # cap total prompt+output to ?k tokens
        #max_num_seqs=16, # max concurrent requests to handle in parallel
        #gpu_memory_utilization=0.9, # cap GPU memory usage to leave some room for other processes
    )
    print("vLLM model initialized.")

    # generate the results by looping through the prompt_messages
    all_setting_evals = {}
    for lang_key,all_setting_prompt_one_lang in all_setting_results_all_lang.items():
        print(lang_key)
        all_setting_evals_one_lang = {}
        for setting_key,one_setting_prompts in all_setting_prompt_one_lang.items():
            print(setting_key)
            one_setting_evals = []

            for question_dict in tqdm(one_setting_prompts):
                question = question_dict['question']
                ground_truth_answer = question_dict['ground_truth_answer']
                generated_answer = question_dict['answer']
                prompt = build_llm_judge_prompt(question,ground_truth_answer,generated_answer)
                eval = None  # 🆕 Initialize
                for attempt in range(10):  # 🆕 Try up to n times
                    # Generate output with sampling
                    sampling_params = SamplingParams(
                        max_tokens= 500,  
                        temperature= 0.1, # very low temperature for simple Y/N judgment
                        #top_k=50,
                        top_p=0.95,
                        stop=["<|return|>", "<|call|>","<|assistant|>"], # Add Harmony stop strings so the reply ends cleanly at a boundary
                        # no seed => natural variation across calls
                    )

                    # use Harmony style for gpt_oss
                    if model_name in ['gpt_oss_20b']:
                        harmony_prompt,harmony_encoder = render_prompt_in_harmony_style(prompt)
                        out = llm.generate([harmony_prompt], sampling_params)[0]
                        # vLLM gives you both text and token IDs
                        #text = out.outputs[0].text
                        output_tokens = out.outputs[0].token_ids  # <-- these are the completion token IDs
                        # --- 3) Parse the completion token IDs back into structured Harmony messages ---
                        messages = harmony_encoder.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
                        ## among Harmony channels, keep only final
                        generated_text = "".join(c.text for m in messages if m.channel == "final" for c in m.content).strip()
                    else:
                        out = llm.generate([prompt], sampling_params)[0]
                        generated_text = out.outputs[0].text

                    # Extract answer
                    eval = generated_text
                    # 🆕 If answer is YES or NO ..., break early
                    if eval in ("YES", "NO"):
                        break

                #print(setting_key,'-generated_text:',generated_text)
                print(setting_key,'-eval:',eval)
                question_dict['eval'] = eval
                one_setting_evals.append(question_dict)
            all_setting_evals_one_lang[setting_key] = one_setting_evals
        all_setting_evals[lang_key] = all_setting_evals_one_lang
        
    with open(f"{prompts_pkl.replace('.pkl','')}_{model_name}_LLMjudge.pkl", "wb") as f:
        pickle.dump(all_setting_evals, f)
        print("results saved")
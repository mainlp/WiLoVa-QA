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
# parse CLI args to configure everything
# Usage: python3 dialectqa.py <gpu_id> <prompts_pkl> <model_name> <tokenizer_path> <model_path>
gpu_id = sys.argv[1]
prompts_pkl = sys.argv[2]
model_name = sys.argv[3]
tokenizer_path = sys.argv[4]
model_path = sys.argv[5]

# Tell vLLM to sleep when idle
os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
# set GPU via env from CLI arg
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id #select the GPU number n #masks all GPUs except GPU n.

if model_name in ['llama3_70b','qwen2.5_72b','gpt_oss_120b']:
    parallel_size = 4
else:
    # use only one GPU
    parallel_size = 1

print(f"Using {parallel_size} GPU(s), selected GPU number: {gpu_id}")

# import vLLM after setting CUDA_VISIBLE_DEVICES
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

#------

def extract_answer_from_text(text):
    """
    Extracts the first <Answer>...</Answer> content from the text.
    Returns the answer string if found, otherwise None.
    """
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

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
        all_prompts = pickle.load(f)
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
        max_model_len=10000,       # cap total prompt+output to ?k tokens
        #max_num_seqs=16, # max concurrent requests to handle in parallel
        #gpu_memory_utilization=0.9, # cap GPU memory usage to leave some room for other processes
    )
    print("vLLM model initialized.")

    # generate the results by looping through the prompt_messages
    all_results = {}
    for lang_key,all_setting_prompt_one_lang in all_prompts.items():
        print(lang_key)
        all_setting_results_one_lang = {}
        for setting_key,one_setting_prompts in all_setting_prompt_one_lang.items():
            print(setting_key)
            one_setting_results = []

            for question_dict in tqdm(one_setting_prompts):
                prompt = question_dict['prompt']
                answer = None  # 🆕 Initialize
                for attempt in range(5):  # 🆕 Try up to n times
                    # Generate output with sampling
                    sampling_params = SamplingParams(
                        max_tokens= 5000,  
                        temperature= 0.3, # low temperature to stay focused for factual QA
                        #top_k=50,
                        top_p=0.95,
                        stop=["<|return|>", "<|call|>","<|assistant|>"], # Add Harmony stop strings so the reply ends cleanly at a boundary
                        # no seed => natural variation across calls
                    )

                    # use Harmony style for gpt_oss
                    if model_name in ['gpt_oss_20b','gpt_oss_120b']:
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
                    answer = extract_answer_from_text(generated_text)
                    # 🆕 If answer is not None and not ..., break early
                    if answer and answer != '...':
                        break

                #print(setting_key,'-generated_text:',generated_text)
                print(setting_key,'-answer:',answer)
                question_dict['generated_text'] = generated_text
                question_dict['answer'] = answer
                one_setting_results.append(question_dict)
            all_setting_results_one_lang[setting_key] = one_setting_results
        all_results[lang_key] = all_setting_results_one_lang
        
    with open(f"{prompts_pkl.replace('.pkl','')}_{model_name}_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
        print("results saved")
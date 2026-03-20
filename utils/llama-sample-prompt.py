from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import io
import os
import re
import json

# model_name = "meta-llama/Llama-2-7b-chat-hf"  # or your local model path
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

token = None

# load model
GPU_DEVICES = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES


tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


data_file = "../data/zh.zh-yue-YM.json"
with io.open(data_file, "r", encoding="utf8") as f:
    data_dict = json.load(f)


def message_generate(batch_instance):
    batch_message = []
    for one_instance in batch_instance:
        promise = one_instance[0]
        hypothesis = one_instance[1]

        comments = ""
        for i, comments_id in enumerate(ordered_comments_id):
            the_comments = one_instance[-1][comments_id][0]
            the_comments_label = one_instance[-1][comments_id][1]
            comments += f"\nComment {i + 1}: {the_comments} So I choose {option_dict[the_comments_label]}. "
        messages = [
            {"role": "user",
             "content": f"Please carefully and fairly base your selection on the comments below to determine whether the following statement is true (entailment), "
                        f"undetermined (neutral), or false (contradiction) given the context below and select ONE of the listed options and start your answer with a single letter. "
                        f"\nContext: {promise} \nStatement: {hypothesis} {comments}\nA. {option_order_word[0]} \nB. {option_order_word[1]} \nC. {option_order_word[2]}. \nAnswer:"}
        ]
        batch_message.append(messages)
    return batch_message


print(model.device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

all_outputs = {}
for query_k in data_dict.keys():
    all_outputs[query_k] = {}
    for question_k in range(1, 4):
        if f"Question{question_k}" in data_dict[query_k].keys() and data_dict[query_k][f"Question{question_k}"]:
            # prompt = "What is the capital of France?"
            prompt = data_dict[query_k][f"Question{question_k}"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs,
                                     max_new_tokens=256,
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     output_logits=True,
                                     eos_token_id=terminators,
                                     do_sample=False,
                                     )
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            all_outputs[query_k][f"Question{question_k}"] = (data_dict[query_k][f"Question{question_k}"], outputs)



savedir = "../model_outputs/%s/output.json" % (model_name.split("/")[1])
with open(savedir,"w") as f:
    json.dump(all_outputs,f,indent=4)
    f.write('\n')


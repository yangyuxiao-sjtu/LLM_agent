import os
import sys
import openai
import random
import json
import threading
import torch
from transformers import AutoTokenizer
import transformers

openai.api_key = "sk-proj-jxew31rgcBtYjchHn8ziT3BlbkFJf3H5tds737YtWMTz4RS3"
import queue

Llama_model = None
tokenizer = None


def his_to_str(history, metadata=None):
    prompt = ""
    for action in history:
        if isinstance(action["metadata"], str) and isinstance(action["action"], str):
            prompt += (
                "The objects you have seen are:"
                + action["metadata"]
                + "\n"
                + action["action"]
                + "\n"
            )

    if isinstance(metadata, str):
        prompt += "The objects you have seen are:" + metadata + "\n"
    return prompt


def choose_examples(prompt_path, example_num):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    short_idx = [
        random.randint(0, len(prompts["short"]) - 1) for _ in range(example_num)
    ]
    long_idx = [random.randint(0, len(prompts["long"]) - 1) for _ in range(example_num)]
    short_examples = [prompts["short"][i] for i in short_idx]
    long_examples = [prompts["long"][i] for i in long_idx]
    return short_examples, long_examples


def call_openai_api(model, max_token, top_p, stop, sys_prompt, user_prompt, n):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_token,
        top_p=top_p,
        n=n,
        logprobs=True,
        top_logprobs=4,
        stop=stop,
    )
    return response


def load_llama():
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipline = transformers.pipeline(
        "text-generation",
        model=model_name,
        device_map="balanced_low_0",
        model_kwargs={"load_in_8bit": True},
    )
    return pipline, tokenizer


# this func is try to use opensource llm(LLAMA3) instead of GPT-4
def call_llama(max_token, stop, sys_prompt, user_prompt, n):
    global Llama_model, tokenizer
    if Llama_model == None:
        Llama_model, tokenizer = load_llama()
    input_prompt = sys_prompt + user_prompt
    input_length = len(input_prompt.split())
    # different way in calculating max_token

    return_ls = []
    for _ in range(n):
        sequences = Llama_model(
            input_prompt,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.encode(stop),
            max_new_tokens=max_token,
            truncation=True,
        )
        output = sequences[0]["generated_text"].replace(input_prompt, "")
        output = output.replace("\n", "")
        return_ls.append(output)
    return return_ls


def call_llm(
    model,
    max_token=150,
    top_p=0.8,
    stop=["\n", "."],
    sys_prompt=None,
    user_prompt=None,
    n=1,
):
    if model == "llama":
        return call_llama(max_token, stop, sys_prompt, user_prompt, n)
    response = {}
    if sys_prompt == None:
        assert "sys_prompt is None!"
    if user_prompt == None:
        assert "user_prompt is None!"
    for i in range(1, 8):
        try:
            response = call_openai_api(
                model, max_token, top_p, stop, sys_prompt, user_prompt, n
            )
            break
        except Exception as e:
            response = {"error": e}

    return response


def openai_thread(
    model, max_token, top_p, stop, sys_prompt, user_prompt, n, result_queue, tag
):
    res = call_llm(
        model=model,
        max_token=max_token,
        top_p=top_p,
        stop=stop,
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        n=n,
    )
    result_queue.put((res, tag))


def call_llama_thread(max_token, stop, sys_prompts, user_prompts, tags, n):
    ret_list = []
    if tags == None:
        tags = [i for i in range(len(sys_prompts))]
    for i in range(len(sys_prompts)):
        res = call_llama(max_token, stop, sys_prompts[i], user_prompts[i], n)
        ret_list.append((res, tags[i]))
    return ret_list


# aiming to call multiple openai at same time
def call_llm_thread(
    model,
    max_token,
    top_p=0.8,
    stop=["\n", "."],
    sys_prompts=[],
    user_prompts=[],
    tags=None,
    n=1,
):
    if len(sys_prompts) != len(user_prompts):
        assert "sys_prompts should has same number with user_prompts!"
        return
    if model == "llama":
        return call_llama_thread(max_token, stop, sys_prompts, user_prompts, tags, n)
    result_queue = queue.Queue()
    res_list = []

    for i in range(len(sys_prompts)):

        sys_prompt = sys_prompts[i]
        user_prompt = user_prompts[i]

        if tags != None:
            thread = threading.Thread(
                target=openai_thread,
                args=(
                    model,
                    max_token,
                    top_p,
                    stop,
                    sys_prompt,
                    user_prompt,
                    n,
                    result_queue,
                    tags[i],
                ),
            )
        else:
            thread = threading.Thread(
                target=openai_thread,
                args=(
                    model,
                    max_token,
                    top_p,
                    stop,
                    sys_prompt,
                    user_prompt,
                    n,
                    result_queue,
                ),
            )
        thread.start()
    for _ in range(len(sys_prompts)):
        res = result_queue.get()
        res_list.append(res)
    while not result_queue.empty():
        res = result_queue.get()
        res_list.append(res)
    return res_list


# def call_llm(model, tokenizer, sys_prompt, user_prompt, max_token, top_n=1):
#     with torch.no_grad():
#         text = sys_prompt + "\n" + user_prompt
#         inputs = tokenizer(text, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(model.device)
#         logits = model.forward(input_ids)
#         log_prob = logits.logits.view(-1, logits.logits.size(-1))[-1]
#         values, indices = torch.topk(log_prob, k=top_n, dim=-1, largest=True)
#         res = []
#         for idx in indices:
#             res.append(tokenizer.decode(idx))
#         i = 1
#         del log_prob, logits
#         while i < max_token:
#             for j, new_text in enumerate(res):
#                 inputs = tokenizer(text + " " + new_text, return_tensors="pt")
#                 input_ids = inputs["input_ids"].to(model.device)
#                 logits = model.forward(input_ids)
#                 # only count the prob of first token as the probability of the new_text
#                 next_token = torch.argmax(logits.logits, dim=-1).reshape(-1)[-1]
#                 res[j] = res[j] + tokenizer.decode(next_token)
#                 del logits, input_ids, inputs
#             i += 1
#         return res

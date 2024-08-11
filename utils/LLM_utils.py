import os
import sys
import openai
from openai import OpenAI
import random
import json
import threading
import torch
from transformers import AutoTokenizer
import transformers
from vllm import LLM, SamplingParams
import yaml
from LLM_subgoal import sentence_embedder

from sentence_transformers.util import cos_sim

# openai.api_key = "sk-proj-jxew31rgcBtYjchHn8ziT3BlbkFJf3H5tds737YtWMTz4RS3"
import queue

Llama_model = None
tokenizer = None
token_used = 0

use_vllm = False
openai_client =None

def knn_retriver(data, key_func, get_prompt, input, n, same_ICL=True):
    encoded = sentence_embedder.encode(input)
    ls = []
    for item in data:
        cmp_list = key_func(item)
        if isinstance(cmp_list, str):
            cmp_list = [cmp_list]
        dist = 0
        for cmp in cmp_list:
            tmp = cos_sim(sentence_embedder.encode(cmp), encoded)
            if tmp > dist:
                dist = tmp
        if same_ICL == False and dist == 1.0:
            continue
        ls.append((item, dist))
    top_k = sorted(ls, key=lambda x: x[1], reverse=True)
    top_k = top_k[:n]
    ret = [item for (item, _) in top_k]
    knn_prompt = get_prompt(ret)
    return knn_prompt


def his_to_str(history, metadata=None, multi_obs=True, add_prefix=False):
    prompt = ""
    if multi_obs == True:
        num = 100
    else:
        num = 1
    l = len(history)
    if isinstance(metadata, str) and num > 0:
        prompt += "The objects you have seen are:" + metadata + "\n"
        num -= 1
    for i, action in enumerate(history):
        if isinstance(action["metadata"], str) and num > 0:
            prompt += "The objects you have seen are:" + action["metadata"] + "\n"
            num -= 1
        if isinstance(action["action"], str):
            if add_prefix:
                prompt += "Act: " + action["action"] + "\n" + ">OK\n"
            else:
                prompt += action["action"] + "\n" + ">OK\n"

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


def load_vllm():

    # Create an LLM.
    llm = LLM(
        model="/mnt/sda/yuxiao_code/meta-llama4b/Meta-Llama-3-8Bawq",
        quantization="AWQ",
        gpu_memory_utilization=0.9,
    )
    return llm


def call_vllm_llama(max_token, stop, sys_prompt, user_prompt, n):
    global Llama_model
    one_line = False
    if stop != None and "\n" in stop:
        one_line = True
    if Llama_model == None:
        Llama_model = load_vllm()
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_token, n=n
    )
    prompts = sys_prompt + user_prompt
    outputs = Llama_model.generate(prompts, sampling_params)
    res = []
    for output in outputs:
        tmp = output.outputs[0].text
        if one_line:
            tmp = tmp.split("\n")[0]
        res.append(tmp)
    return res


def increase_token(num):
    global token_used
    token_used += num


def get_used_token():
    global token_used
    return token_used


def call_deepseek(max_token, stop, sys_prompt, user_prompt, n, top_p=0.8):
    global token_used
    ## deepseek currently only support n=1
    #sk-bca6adb77186430fbcfceb92ebad4fb1
    res = []
    client = OpenAI(
        api_key="sk-bca6adb77186430fbcfceb92ebad4fb1",
        base_url="https://api.deepseek.com",
    )
    for tries in range(0, 8):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                logprobs=True,
                top_p=top_p,
                stop=stop,
                max_tokens=max_token,
            )
            token_used += response.usage.total_tokens
            if response.choices[0].message.content.strip() != "":
                res.append(response.choices[0].message.content)
            if len(res) == n:
                break
        except Exception as e:
            print('openai_error!',e)
    if len(res)<n:
        assert("call openai error!")
    return res
def load_client(key_path="/mnt/sda/yuxiao_code/chat_with_openai_api/openai_key.yaml"):
    openai._reset_client()
    key = yaml.safe_load(open(key_path))
    for k, v in key.items():
        setattr(openai, k, v)
    return openai._load_client()
def call_openai(max_token, stop, sys_prompt, user_prompt, n, top_p=0.8):
    global token_used
    global openai_client
    if openai_client ==None:
        openai_client=load_client()
    ## deepseek currently only support n=1
    #sk-bca6adb77186430fbcfceb92ebad4fb1
    res = []

    for tries in range(0, 8):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                top_p=top_p,
                stop=stop,
                max_tokens=max_token,
                n=n
            )
          
            token_used += response.usage.total_tokens
            for i in range(len(response.choices)):
                if response.choices[i].message.content.strip() != "":
                    res.append(response.choices[i].message.content)
            if len(res) == n:
                break
        except Exception as e:
            print('OPENAI_error!',e)
            print('response is:',  (response))
    if len(res)<n:
        assert("call openai error!")
    return res
# this func is try to use opensource llm(LLAMA3) instead of GPT-4
def call_llama(max_token, stop, sys_prompt, user_prompt, n):
    if use_vllm == True:
        return call_vllm_llama(max_token, stop, sys_prompt, user_prompt, n)
    global Llama_model, tokenizer, token_used

    if Llama_model == None:
        Llama_model, tokenizer = load_llama()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    input_prompt = sys_prompt + user_prompt
    input_length = len(input_prompt.split())
    increase_token(input_length * n)
    # different way in calculating max_token
    one_line = False
    if stop != None:
        if isinstance(stop, list) and len(stop) == 1:
            stop = stop[0]
            if stop == "\n":
                one_line = True
        if "\n" in stop:
            one_line = True
        stop = terminators + tokenizer.encode(stop)
    else:
        stop = terminators
    return_ls = []
    for _ in range(n):
        sequences = Llama_model(
            input_prompt,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=stop,
            max_new_tokens=max_token,
            truncation=True,
        )
        output = sequences[0]["generated_text"].replace(input_prompt, "")
        if one_line:
            output = output.split("\n")[0]
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
    elif model == "deepseek":
        return call_deepseek(
            max_token,
            stop,
            sys_prompt,
            user_prompt,
            n,
            top_p=top_p,
        )
    elif "gpt-4" in model:
        return call_openai(
            max_token,
            stop,
            sys_prompt,
            user_prompt,
            n,
            top_p=top_p,
        )
    elif "gpt" not in model:
        assert "error model!!"

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

def call_openai_thread(max_token,stop,sys_prompts,user_prompts,tags,n,top_p):
    ret_list = []
    print('openai_thread:',len(sys_prompts))
    if tags == None:
        tags = [i for i in range(len(sys_prompts))]
    for i in range(len(sys_prompts)):
        res = call_openai(max_token, stop, sys_prompts[i], user_prompts[i], n, top_p)
        ret_list.append((res, tags[i]))
    return ret_list

def call_deepseek_thread(max_token, stop, sys_prompts, user_prompts, tags, n, top_p):
    ret_list = []
    if tags == None:
        tags = [i for i in range(len(sys_prompts))]
    for i in range(len(sys_prompts)):
        res = call_deepseek(max_token, stop, sys_prompts[i], user_prompts[i], n, top_p)
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
    elif model == "deepseek":
        return call_deepseek_thread(
            max_token, stop, sys_prompts, user_prompts, tags, n, top_p=top_p
        )
    elif "gpt-4" in model:
           return call_openai_thread(
            max_token, stop, sys_prompts, user_prompts, tags, n, top_p=top_p
        )
    elif "gpt" not in model:
        assert "error model!!"
    return call_openai_thread(max_token,stop,sys_prompts,user_prompts,tags,n,top_p=p)
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

import os
import sys
import openai
import random
import json
openai.api_key = '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'
def his_to_str(history,metadata=None):
        prompt=None
        for action in history:
            if isinstance(action['metadata'],str )and isinstance(action['action'],str):
                if prompt== None:
                    prompt = 'The objects that might help you to solve the task are:'+action['metadata']+'\n'+action['action']+'\n'
                else:
                    prompt +='The objects that might help you to solve the task are:'+action['metadata']+'\n'+action['action']+'\n'
     
        if isinstance(metadata,str):
            if prompt == None :
                prompt = 'The objects that might help you to solve the task are:' +metadata +'\n'
            else:
                prompt+='The objects that might help you to solve the task are:' +metadata +'\n'
        return prompt
def choose_examples(prompt_path,example_num):
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompts =json.load(f)
    short_idx = [random.randint(0,len(prompts['short'])-1) for _ in range(example_num)]
    long_idx =[random.randint(0,len(prompts['long'])-1) for _ in range(example_num)]
    short_examples = [prompts['short'][i] for i in short_idx]
    long_examples = [prompts['long'][i] for i in long_idx] 
    return short_examples,long_examples
def call_openai_api(model,
                    max_token,
                    top_p,
                    stop,
                    sys_prompt,
                    user_prompt,
                    n):
    response = openai.ChatCompletion.create(
                    model=model,  
                    messages=[{"role": "system", "content": sys_prompt},{''"role": "user", "content":user_prompt}], 
                    max_tokens=max_token,
                    top_p=top_p,
                    n=n,
                    logprobs=True,
                    top_logprobs= 4,
                    stop=stop
                )
    return response
def call_openai(model,
                max_token,
                top_p,
                stop,
                sys_prompt,
                user_prompt,
                n):

    response= {}
    for i in range(1,8):
        try: 
            response = call_openai_api(model,max_token,top_p,stop,sys_prompt,user_prompt,n)
            break
        except Exception as e:
            response={}
    
 

    return response

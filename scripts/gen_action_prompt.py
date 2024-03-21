import os
import json
file_path = "../prompts/prompts.json"
save_path ='../prompts/action_prompts.json'
 
with open(file_path, 'r', encoding='utf-8') as f:
    data=json.load(f)
prompts ={'short':[],'long':[]}
for sample in data:
    prompt =None
    for sg in sample:
        if prompt == None:
            prompt ='Your task is: '+sg['task']+'\n'+ 'The objects that might help you to solve the task are:'+sg['predict']['output']+'\n'+sg['subgoal'].replace("HLA: ", "")+'\n'
        else:
            prompt+= 'The objects that might help you to solve the task are:'+sg['predict']['output']+'\n'+sg['subgoal'].replace("HLA: ", "")+'\n'
    if(len(sample)<=4):
         prompts['short'].append(prompt)
    else: 
         prompts['long'].append(prompt)

with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(prompts, json_file, indent=4)  




 
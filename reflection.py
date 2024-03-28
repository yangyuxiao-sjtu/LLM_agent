from utils.LLM_utils import his_to_str,call_openai_api
import os
import sys
class reflection():
    def __init__(
        self,
        model ="gpt-4",
        max_tokens=100,
        top_p=0.8,
        prompt_path='prompts/value_prompts.json',
        stop='\n'
    ):
        self.model=model
        self.max_tokens = max_tokens
        self.prompt_path = prompt_path
        self.top_p=top_p
        self.stop=stop
        self.base_prompt = """
You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. 
Do not summarize your environment, but rather think about what is wrong with the critic you generated. Devise a concise, new critic that accounts for your mistake with reference to specific  
critics that you should have generated. You will need this later when you are solving the same task. Give your plan after "Failure feedback:". Here are three example:
"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir,prompt_path),"r",encoding="utf-8") as f:
            data = f.read()
        self.sys_prompt= self.base_prompt+data
#we can use failure_info if needed
    def generate_reflection(self,task,history,failure_info =None):
        task_prompt = "Your task is: "+str(task)
        str_his = his_to_str(history)
        task_prompt+=str_his
        if failure_info!= None:
            task_prompt+=f"faliure reason is :{failure_info}"
        task_prompt += "Failure feedback:"
        response = call_openai_api(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            sys_prompt=self.sys_prompt,
            user_prompt=task_prompt
        )
        return response.choices[0].message['content']

             


from .utils.LLM_utils import his_to_str, call_openai_api, call_llm
import os
import sys


class reflection:
    def __init__(
        self,
        model="llama",
        max_tokens=100,
        top_p=0.8,
        prompt_path="prompts/reflection.txt",
        use_predict=True,
        stop="\n",
    ):
        self.model = model
        self.use_predict = use_predict
        self.max_tokens = max_tokens
        self.prompt_path = prompt_path
        self.top_p = top_p
        self.stop = stop
        self.base_prompt = """
You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. 
Do not summarize your environment, but rather think about what is wrong with the critic you generated. Devise a concise, new critic that accounts for your mistake with reference to specific  
critics that you should have generated. You will need this later when you are solving the same task. Give your plan after "Failure feedback:". Here are three example:
"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, prompt_path), "r", encoding="utf-8") as f:
            data = f.read()
        self.sys_prompt = self.base_prompt + data

    # we can use failure_info if needed
    def generate_reflection(
        self, task, history, critic=None, failure_info=None, predict=None
    ):

        task_prompt = "Your task is: " + str(task)
        if self.use_predict and predict != None:
            task_prompt += "The objects you have seen are:" + predict
        str_his = his_to_str(history)
        task_prompt += str_his

        if critic != None:
            task_prompt += "Critic:" + critic + "\n"
        task_prompt += "Status: Fail\n"
        if failure_info != None and "message" in failure_info:
            task_prompt += f" faliure reason is :{failure_info['message']}"
            task_prompt += """If, after your analysis, you determine that the errors were not a result of agent's actions, please output the following statement: "The errors were not caused by the agent, and it is advised to continue previous actions."
        """
        task_prompt += "Failure feedback:"
        response = call_llm(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            sys_prompt=self.sys_prompt,
            user_prompt=task_prompt,
            n=1,
        )
        res = ""
        if self.model == "GPT-4":
            res = response.choices[0].message["content"]
        elif self.model == "llama":
            res = response[0]
        else:
            assert "currently model should be GPT or llama!"
        return res

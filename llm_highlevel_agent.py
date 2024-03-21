from typing import Dict, List, Type, Union

from lgp.abcd.task import Task
from lgp.abcd.agent import Agent
from lgp.abcd.subgoal import Subgoal
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction

import copy
class LlmAgent(Agent):
    def __init__(self,
        adaptation_model,
        critic_model,
        value_model,
        action_propsal_model,
        predict_model,
        device :str
        ):
        super().__init__()
        #the trined LLaVa model
        self.adaptation_model=adaptation_model
        #the llm used in RAFA
        
        self.value= value_model
        self.action_proposal=action_propsal_model
        self.predict=predict_model

        self.device = device
        self.task=None
        # for debugging 
        self.trace = {}
        # handle failed action
        self.log_action_failed =False
        #maintain obs,action and critic
        self.action_history=[]
        self.failed_action=None
    def get_trace(self, device="cpu"):
        return {k: v.to(device) if v is not None else v for k, v in self.trace.items()}
    def clear_trace(self):
        self.trace={}
    def action_execution_failed(self):
        self.log_action_failed=True
        self.failed_action=self.action_history[-1]
        self.action_history=self.action_history[:-1]
    def _reset(self):
        self.trace = {}
        self.action_proposal.reset()
        self.value.reset()
        self.log_action_failed=False
        self.predict.reset()
        self.failed_action=None
        self.action_history=[]
        self.task=None

        
    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        #here we only need to use the language description of the task
        self._reset()
        self.task=str(task)
    
    def finalize(self, total_reward: float):
        #nothing todo
        ...
    def _log_action(self,meta_info,action,critic):
        self.log_action_failed =False
        self.failed_action=None
        self.action_history.append({'metadata':meta_info,'action':action,'critic':critic})
        
        self.trace['metadata']=meta_info
        self.trace['action']=action
        self.trace['critic']=critic
    def _choose_action(self,act_his,sample_pernode,depth):
        #Tobe Done
        ...
    def _get_start_idx(self,sample_pernode,depth):
        return 0 if depth == 0 else  (sample_pernode**depth -1)/(sample_pernode-1)
    
    def _RAFA(self,metadata,
              depth=3,
              sample_pernode=3):
        
        #convert the action into str form,as we only need it 
        root_act_his= [{'metadata':action['metadata'],'action':str(action['action']),'critic':action['critic'] }for action in self.action_history]
        #there should be atmost sample_pernode**(depth+1) nodes
        tmp_act_his = [None]*((sample_pernode** (depth+1)))
        # this refers to the metadata after taking action
        tmp_metadata=[None]*((sample_pernode** (depth+1)))
        tmp_metadata[0]=metadata
        tmp_act_his[0]=copy.deepcopy(root_act_his)
        # depth-1 is the true thought depth
        #lots of basic calculate here
        for dep in range(depth):
            #get the parent node number, note that dep start from 0
            layer_samples =sample_pernode ** dep
            parent_start_idx = self._get_start_idx(sample_pernode,dep)
            
            #the failed information should be given to action_proposal model
            if dep ==0 and self.log_action_failed:
                failed_info =self.failed_action
            else:
                failed_info=None

            for parent_num in range(layer_samples):
                parent_idx =parent_start_idx+parent_num
                # the model should return sample_pernode actions as a list
                actions = self.action_proposal.get_actions(self.task,tmp_act_his[parent_idx],tmp_metadata[parent_idx],sample_pernode,failed_info)
                child_start_idx = self._get_start_idx(sample_pernode,dep+1)+sample_pernode*parent_num

                for child_num in range(sample_pernode):
                    child_idx = child_start_idx+child_num
                    #the metadata before taking action
                    child_metadata=tmp_metadata[parent_idx]
                    #giving predict model action history, current metadata and current action, should return a new metadata
                    tmp_metadata[child_idx]=self.predict.get_obs(tmp_act_his[parent_idx],child_metadata,actions[child_num])
                    #giving value model action history, current metadata, current action and failed_info, should return a critic
                    tmp_act_his[child_idx]=copy.deepcopy(tmp_act_his[parent_idx])
                    tmp_act_his[child_idx].append({'metadata':child_metadata,'action':actions[child_num],'critic':None})
                    child_critic =self.value.act(self.task,tmp_act_his[child_idx],failed_info)
                    tmp_act_his[child_idx][-1]['critic']=child_critic


        

        return self._choose_action(tmp_act_his,sample_pernode,depth)
    def act(self, observation_or_state_repr: Union[Observation, StateRepr]) -> Action:
        #call below function to create a action, where act_type_str can be create through AlfredSubgoal.action_type_intid_to_str
        # and arg_vector_out is a   torch.Size([1, 125]) one hot vector where 1 indicate which object to interact with
        # more detail could be view   /hlsm/lgp/models/alfred/hlsm/hlsm_subgoal_model.py:in _sample_subgoal
        #AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector_out)
        if isinstance(observation_or_state_repr, Observation):
            observation = observation_or_state_repr
        else:
            observation = observation_or_state_repr.observation
        #the meta data estimated by adaptation_model
        meta_info = self.adaptation_model.act(observation,self.task)
        proposed_action,critic= self._RAFA(meta_info)
        self._log_action(meta_info,proposed_action,critic)
        return proposed_action




       



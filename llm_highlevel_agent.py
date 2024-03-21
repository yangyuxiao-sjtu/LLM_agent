from typing import Dict, List, Type, Union
import sys
from evaluate_value import LLM_critic
from action_proposal import action_proposal
from adapt_model import adap_model
from predict import predict_model
import torch
sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.abcd.task import Task
from lgp.abcd.agent import Agent
from lgp.abcd.subgoal import Subgoal
from lgp.abcd.action import Action
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
from lgp.abcd.repr.state_repr import StateRepr
from lgp.abcd.skill import Skill
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal
from lgp.env.alfred.alfred_action import AlfredAction


import copy
class LlmAgent(Agent):
    def __init__(self,
        device ='cuda',
        gamma=0.9,
        adaptation_model=adap_model(),
        value_model=LLM_critic(),
        action_propsal_model=action_proposal(),
        predict_model=predict_model()
        ):
        super().__init__()
        self.gamma=gamma
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
    def _get_critic_value(self,critic):
        str_value = critic.partition('=')[-1]
        str_value.replace(".","")
        try:
            value =float(str_value)
        except ValueError:
            print("str_value error:",str_value)
        return value
    def _choose_action(self,action_history_list,length):
        tmp_value = [None]*len(action_history_list)
        for i,act_his in enumerate(action_history_list):
            depth =len(act_his)-length
            value = self._get_critic_value(act_his[-1]['critic'])
            tmp_value[i]=value *(self.gamma **depth)
            argmax = tmp_value.index(max(tmp_value))
        # the first critic,action of the best long term rollout
        critic =action_history_list[argmax][length]['critic']
        action =action_history_list[argmax][length]['action']
        acion_type = action.partition(':')[0].replace(" ","")
        action_obj = action.partition(":")[1].replace(" ","")
        obj_id =AlfredSubgoal.object_string_to_intid(action_obj)
        obj_tensor =torch.zeros([1,125]).to(self.device)
        if(obj_id==125):
            print('object name error:',action_obj)
        obj_tensor[obj_id]=1.0
        return AlfredSubgoal.from_type_str_and_arg_vector(acion_type, obj_tensor)

        


     
    def _get_start_idx(self,sample_pernode,depth):
        return 0 if depth == 0 else  (sample_pernode**depth -1)/(sample_pernode-1)
    
    def _RAFA(self,metadata,
              depth=3,
              sample_pernode=3):
        
        #convert the action into str form,as we only need it 
        root_act_his= [{'metadata':action['metadata'],'action':str(action['action']).replace("HLA: ", ""),'critic':action['critic'] }for action in self.action_history]
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
                    tmp_act_his[child_idx]=copy.deepcopy(tmp_act_his[parent_idx])
                    tmp_act_his[child_idx].append({'metadata':child_metadata,'action':actions[child_num],'critic':None})
                    #giving predict model action history, current metadata and current action, should return a new metadata
                    tmp_metadata[child_idx]=self.predict.act(self.task,tmp_act_his[child_idx])
                    #giving value model action history, current metadata, current action and failed_info, should return a critic

                    child_critic =self.value.act(self.task,tmp_act_his[child_idx],failed_info)
                    tmp_act_his[child_idx][-1]['critic']=child_critic



        #tmp_act_his[0] is the root action history, so we choose the highest cumulative reward start from 1
        return self._choose_action(tmp_act_his[1:],len(root_act_his))
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




       



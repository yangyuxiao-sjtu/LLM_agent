from typing import Dict, List, Type, Union
import sys
from .evaluate_value import LLM_critic
from .action_proposal import action_proposal
from .adapt_model import adap_model
from .predict import predict_model
from .llm_hlsm_subgoal import LLMHlsmSubgoalModel
import torch
import os
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
from lgp.env.alfred.segmentation_definitions import object_string_to_intid
from lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskRepr
from lgp.agents.agent_state import AgentState
from lgp.models.alfred.hlsm.hlsm_model_factory import HlsmModelFactory
from .process_predict import predict_processor
import re
import copy
class LlmAgent(Agent):
    def __init__(self,
        proposal,
        obs_func,
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
        self.predict_processor = predict_processor()
        self.value= value_model
        self.action_proposal=action_propsal_model
        self.predict=predict_model
        self.proposal=proposal
        self.obs_func = obs_func
        self.device = device
        self.task=None
        self.TaskReprCls = HlsmTaskRepr
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
        self.proposal.clear_trace()
    def action_execution_failed(self):
        self.log_action_failed=True
        if(len(self.action_history)>0):
            self.failed_action=self.action_history[-1]
            self.action_history=self.action_history[:-1]
        self.proposal.action_execution_failed()
    def _reset(self):
        self.trace = {}
        self.action_proposal.reset()
        self.value.reset()
        self.log_action_failed=False
        self.predict.reset()
        self.failed_action=None
        self.action_history=[]
        self.task=None
        self.proposal.reset_state()

        
    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        #here we only need to use the language description of the task
        self._reset()
        task_repr = self.TaskReprCls.from_task([task], device=self.device)
        self.agent_state = AgentState(task_repr)
        self.task=str(task)
    
    def finalize(self, total_reward: float):
        #nothing todo
        ...
    def _log_action(self,meta_info,action,critic):
        self.log_action_failed =False
        self.failed_action=None
        self.action_history.append({'metadata':meta_info,'action':action,'critic':critic})
        print('logged action:',str(action))
        self.trace['metadata']=meta_info
        self.trace['action']=action
        self.trace['critic']=critic
    def _get_critic_value(self,critic):
        pattern = r"=\s*(\d+\.\d+|\d+|\d+/\d+)"
        match = re.search(pattern, critic)
        if match:
            value = float(match.group(1))
        else:
            print('value error!!')
            return 0

        return value
    def _choose_action(self,action_history_list,length):
        tmp_value = [None]*len(action_history_list)
        for i,act_his in enumerate(action_history_list):
            depth =len(act_his)-length
            print('CIRIRITIC:::',act_his[-1]['critic'])
            value = self._get_critic_value(act_his[-1]['critic'])
            print(value)
            tmp_value[i]=value *(self.gamma **depth)
        argmax = tmp_value.index(max(tmp_value))
        # the first critic,action of the best long term rollout
        critic =action_history_list[argmax][length]['critic']
        action =action_history_list[argmax][length]['action']
        acion_type = action.partition(':')[0].replace(" ","")
        action_obj = action.partition(":")[2].replace(" ","")
        action_obj=self.predict_processor.process(action_obj)
        #0 base to 1 base
        obj_id =object_string_to_intid(action_obj)+1
        obj_tensor =torch.zeros([1,125]).to(self.device)
        if(obj_id==125):
            print('object name error:',action_obj)
        obj_tensor[0][obj_id]=1.0
        print('here is the subgoal')
        print(obj_tensor.shape)
        return AlfredSubgoal.from_type_str_and_arg_vector(acion_type, obj_tensor),critic

        


     
    def _get_start_idx(self,sample_pernode,depth):
        return 0 if depth == 0 else  int((sample_pernode**depth -1)/(sample_pernode-1))
    
    def _RAFA(self,metadata,
              depth=1,
              sample_pernode=2):
        valid_idx =0
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
                
                # print("parent_idx:",parent_idx)
                # print("meta_ifo:", tmp_metadata[parent_idx])
                actions = self.action_proposal.get_actions(self.task,tmp_act_his[parent_idx],tmp_metadata[parent_idx],sample_pernode,failed_info)
                child_start_idx = self._get_start_idx(sample_pernode,dep+1)+sample_pernode*parent_num

                for child_num in range(sample_pernode):
                    child_idx = child_start_idx+child_num
                    #the metadata before taking action
                    child_metadata=tmp_metadata[parent_idx]
                    tmp_act_his[child_idx]=copy.deepcopy(tmp_act_his[parent_idx])
                    tmp_act_his[child_idx].append({'metadata':child_metadata,'action':actions[child_num],'critic':None})
                    #giving predict model action history, current metadata and current action, should return a new metadata
                    tmp_metadata[child_idx]=self.predict_processor.process( self.predict.act(self.task,tmp_act_his[child_idx])) 
                    #giving value model action history, current metadata, current action and failed_info, should return a critic

                    child_critic =self.value.act(self.task,tmp_act_his[child_idx],failed_info)
                    tmp_act_his[child_idx][-1]['critic']=child_critic
                    valid_idx = max(valid_idx,child_idx)
                    print('child_critic:',child_critic)

        #tmp_act_his[0] is the root action history, so we choose the highest cumulative reward start from 1
        return self._choose_action(tmp_act_his[1:valid_idx+1],len(root_act_his))
    def act(self, observation_or_state_repr: Union[Observation, StateRepr]) -> Action:
        #call below function to create a action, where act_type_str can be create through AlfredSubgoal.action_type_intid_to_str
        # and arg_vector_out is a   torch.Size([1, 125]) one hot vector where 1 indicate which object to interact with
        # more detail could be view   /hlsm/lgp/models/alfred/hlsm/hlsm_subgoal_model.py:in _sample_subgoal
        #AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector_out)
        if isinstance(observation_or_state_repr, Observation):
            observation = observation_or_state_repr
            observation = observation.to(self.device)
            s_0 = self.obs_func(observation, self.agent_state.prev_state,goal=None)
        else:
            observation = observation_or_state_repr.observation
            s_0 = observation_or_state_repr

        #the meta data estimated by adaptation_model
        meta_info = self.adaptation_model.act(observation,self.task)
        meta_info =self.predict_processor.process(meta_info)
        proposed_action,critic= self._RAFA(meta_info)
        proposed_action=proposed_action.to(self.device)
        # to get mask
        proposed_action =self.proposal.forward_inference(proposed_action,
        s_0,
        self.agent_state.task_repr,
        self.proposal.get_state())


        self.proposal.log_action(proposed_action)
        self.agent_state.prev_state = s_0

        self._log_action(meta_info,proposed_action,critic)
        return proposed_action

import compress_pickle as pickle
if __name__ == "__main__":   
    predict =adap_model()
    from lgp.parameters import Hyperparams, load_experiment_definition
    exp_def = Hyperparams(load_experiment_definition('alfred/eval/hlsm_full/eval_hlsm_valid_unseen'))
    model_factory = HlsmModelFactory(exp_def.Hyperparams)
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/"
    actprop =  LLMHlsmSubgoalModel(exp_def.Hyperparams)
    from lgp.models.alfred.hlsm.hlsm_observation_function import HlsmObservationFunction
    obs_func =HlsmObservationFunction(exp_def.Hyperparams)
    subgoal_model_path = "/mnt/sda/yuxiao_code/hlsm/models/alfred_hlsm_subgoal_model_e5.pytorch"
    if subgoal_model_path:
        sd = torch.load(subgoal_model_path)
        actprop.load_state_dict(sd, strict=False)    
    actprop.eval()
    actprop.to("cuda")
    obs_func.eval()
    agent=LlmAgent(actprop,obs_func)

    file_path=os.path.join(read_path,f'rollout_{5400}.gz')
    
    roll=pickle.load(file_path)
    # acion_type="PickupObject"
    # action_obj="Pencil"
    # # from 0 base to 1 base
    # obj_id =object_string_to_intid(action_obj)+1
    # obj_tensor =torch.zeros([1,125])
    # print("id",obj_id)
    # if(obj_id==125):
    #     print('object name error:',action_obj)
    # obj_tensor[0][obj_id]=1.0
    # ac= AlfredSubgoal.from_type_str_and_arg_vector(acion_type, obj_tensor)
    # print(str(ac))
    # action="PickupObject : Pencil"
    # acion_type = action.partition(':')[0].replace(" ","")
    # action_obj = action.partition(":")[2].replace(" ","")
    # print(acion_type)
    # print(action_obj)

    nw_ls = []
    for i,sg in enumerate(roll):
        task =sg['task']
        agent.start_new_rollout(task)
        obs = sg['observation']
        action = agent.act(obs)
        print(str(action))
        break

       



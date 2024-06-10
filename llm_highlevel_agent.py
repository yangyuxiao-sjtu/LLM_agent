from typing import Dict, List, Type, Union
import sys
from .evaluate_value import LLM_critic
from .action_proposal import action_proposal
from .adapt_model import adap_model
from .predict import predict_model
from .reflection import reflection
from .llm_hlsm_subgoal import LLMHlsmSubgoalModel
import torch
import os
import datetime
import json
from .utils.LLM_utils import his_to_str, get_used_token
from collections import deque

# sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.env.alfred.segmentation_definitions import (
    _INTERACTIVE_OBJECTS,
    _OPENABLES,
    _TOGGLABLES,
    _PICKABLES,
    _RECEPTACLE_OBJECTS,
)

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
from lgp.env.alfred.segmentation_definitions import (
    object_string_to_intid,
    object_intid_to_string,
)
from lgp.models.alfred.hlsm.hlsm_task_repr import HlsmTaskRepr
from lgp.agents.agent_state import AgentState
from lgp.models.alfred.hlsm.hlsm_model_factory import HlsmModelFactory
from .process_predict import predict_processor
from .BeamSearch import Beam_Node, Beam
import re
import copy


class LlmAgent(Agent):
    def __init__(
        self,
        proposal,
        obs_func,
        device="cuda",
        gamma=0.9,
    ):
        super().__init__()
        current_time = datetime.datetime.now()
        module_path = os.path.abspath(__file__)
        self.stored_action = deque()
        module_directory = os.path.dirname(module_path)
        self.log = (
            module_directory
            + "/log/"
            + current_time.strftime("%Y-%m-%d_%H-%M-%S")
            + ".log"
        )
        config_path = os.path.join(module_directory, "config.json")
        with open(config_path, "r") as con:
            self.config = json.load(con)
        self.use_predict = self.config["use_predict"]
        self.gamma = gamma
        # the trined LLaVa model

        if self.use_predict == True:
            self.adaptation_model = adap_model(config=self.config)
        # the llm used in RAFA
        self.reflect_model = None
        self.reflection = None

        if self.config["use_reflection"]:
            self.reflect_model = reflection()
        self.predict_processor = predict_processor()
        self.value = LLM_critic(config=self.config)
        self.value.set_log(self.log)
        self.action_proposal = action_proposal(config=self.config)
        self.action_proposal.set_log(self.log)
        self.predict = predict_model(config=self.config)
        self.proposal = proposal
        self.obs_func = obs_func
        self.device = device
        self.task = None
        self.TaskReprCls = HlsmTaskRepr
        self.failure_times = 0
        self.prev_failed = False

        self.fail_info = None
        ##/the origin max_fail_times in hlsm/lgp/experiment_definitions/alfred/eval/hlsm_full/eval_hlsm_full_base.json  10 . since LLM inference is quite slow we change it into 5 for fair compete
        self.max_fail_times = 5
        self.keep_act = False
        # for debugging
        self.trace = {}
        # handle failed action
        self.log_action_failed = False
        # maintain obs,action,predict and critic
        self.action_history = []
        self.failed_action = None

    def get_trace(self, device="cpu"):
        return {k: v.to(device) if v is not None else v for k, v in self.trace.items()}

    def clear_trace(self):
        self.trace = {}
        if self.proposal:
            self.proposal.clear_trace()
        self.keep_act = False

    def action_execution_failed(self, md=None):
        # the action has been logged fail, we don't need to it again
        if self.log_action_failed == True:
            return
        self._log("action failed", self.action_history)
        if md != None and "message" in md:
            self._log("fail_info", md["message"])
            self.fail_info = md["message"]
        else:
            self.fail_info = None
        if self.reflect_model != None and len(self.action_history) > 0:

            self.reflection = self.reflect_model.generate_reflection(
                self.task,
                self.action_history,
                self.action_history[-1]["critic"],
                failure_info=md,
            )
            self._log("reflection:", self.reflection)

            if (
                "The errors were not caused by the agent, and it is advised to continue previous actions."
                in self.reflection
            ):
                self.keep_act = True

        if self.prev_failed:
            self.failure_times += 1
        else:
            self.failure_times = 1
        self.log_action_failed = True
        if len(self.action_history) > 0:
            self._log(
                "action his before fail:",
                [act["action"] for act in self.action_history],
            )
            self.failed_action = self.action_history[-1]
            self.action_history = self.action_history[:-1]
            self._log("failed action", self.failed_action["action"])

        if self.proposal:
            self.proposal.action_execution_failed()

    def _reset(self):
        self.trace = {}
        self.action_proposal.reset()
        self.value.reset()
        self.log_action_failed = False
        self.predict.reset()
        self.failed_action = None
        self.action_history = []
        self.task = None
        self.failure_times = 0
        self.fail_info = None
        self.prev_failed = False
        self.stored_action.clear()
        if self.proposal:
            self.proposal.reset_state()
        self.reflection = None
        self.keep_act = False

    def _log(self, name, obj=None):
        with open(self.log, "a") as f:
            if obj != None:
                f.write(f"{name}: {obj}\n")
            else:
                f.write(f"{name}\n")

    def _process_act(self, action, obs, critic):
        if isinstance(obs, str):
            obs = obs.split()
        is_picked = False
        is_opened = False
        is_sliceable = False
        new_cric = ""
        new_action = None
        obj = action.split(":")[1].strip()
        act = action.split(":")[0].strip()
        for item in self.action_history:
            if "PickupObject" in item["action"]:
                is_picked = True
                if "Knife" in item["action"]:
                    is_sliceable = True
            elif "PutObject" in item["action"]:
                is_picked = False
                is_sliceable = False
            if "OpenObject" in item["action"] and obj in item["action"]:
                is_opened = True
        if is_picked == True and "PickupObject" in action:
            ##in this situation, pickup is not allowed, we should putdown obj first
            self.stored_action.appendleft((action, critic))

            if obj in _RECEPTACLE_OBJECTS:
                new_action = f"PutObject : {obj}"
            else:
                for item in obs:
                    if item in _RECEPTACLE_OBJECTS and item not in _OPENABLES:
                        new_action = f"PutObject : {item}"
                        new_cric = "should put before pick"
        if is_sliceable == False and "SliceObject" in action:
            self.stored_action.appendleft((action, critic))
            for obj in obs:
                if "Knife" in obj:
                    new_action = f"PickupObject : {obj}"
                    self.stored_action.appendleft((action, critic))
                    new_cric = "should pick knife before slice"
        if is_opened == False and "PutObject" in action and obj in _OPENABLES:
            self.stored_action.appendleft((action, critic))
            new_action = f"OpenObject : {obj}"
            new_cric = "should open container before put"
        if new_action == None:
            return action, critic
        return new_action, new_cric

    def start_new_rollout(self, task: Task, state_repr: StateRepr = None):
        # here we only need to use the language description of the task
        self._reset()
        self._log("current used token", get_used_token())
        task_repr = self.TaskReprCls.from_task([task], device=self.device)
        self.agent_state = AgentState(task_repr)
        self.task = str(task)
        self._log("\n\n\n new task", self.task)
        if isinstance(task, str) == False:
            traj = task.get_task_id()
            traj_path = os.path.join("/mnt/sda/yuxiao_code/subgoal_gt", traj + ".json")
            if os.path.exists(traj_path):
                with open(traj_path, "r") as f:
                    sg_data = json.load(f)
                    gt_sg = "\n".join(sg_data)
                    self._log("ground truth subgoal", gt_sg)
            self._log(" new task id", task.get_task_id())

    def finalize(self, total_reward: float):
        # nothing todo
        ...

    def _log_action(self, meta_info, action, critic, predict):
        self.prev_failed = self.log_action_failed
        self.log_action_failed = False
        self.failed_action = None
        self.reflection = None
        self.action_history.append(
            {
                "metadata": meta_info,
                "action": action,
                "critic": critic,
                "predict": predict,
            }
        )
        self._log("metadata", meta_info)
        self._log("predict", predict)
        self._log("action", str(action))
        self._log("critic", critic)
        acts = [act["action"] for act in self.action_history]
        self._log("action history:", ", ".join(acts))
        self.trace["metadata"] = meta_info
        self.trace["action"] = action
        self.trace["critic"] = critic

    def _get_critic_value(self, critic):
        pattern = r"=\s*(\d+\.\d+|\d+|\d+/\d+)"
        match = re.search(pattern, critic)
        if match:
            value = float(match.group(1))
        else:
            print("value error!!", critic)
            return 0

        return value

    def _convert_straction(self, action: str):
        action_type = action.partition(":")[0].replace(" ", "")
        action_obj = action.partition(":")[2].replace(" ", "")
        action_obj = self.predict_processor.process(action_obj)
        # 0 base to 1 base
        obj_id = object_string_to_intid(action_obj) + 1
        obj_tensor = torch.zeros([1, 125]).to(self.device)
        if obj_id == 125:
            self._log("obj name error", action_obj)

        obj_tensor[0][obj_id] = 1.0
        # if self.failure_times >= self.max_fail_times:
        #     action_type = "Stop"
        return AlfredSubgoal.from_type_str_and_arg_vector(action_type, obj_tensor)

    def _choose_action(self, action_history_list, length):
        tmp_value = [None] * len(action_history_list)
        for i, act_his in enumerate(action_history_list):
            depth = len(act_his) - length
            # print('CIRIRITIC:::',act_his[-1]['critic'])
            value = self._get_critic_value(act_his[-1]["critic"])
            # print(value)
            tmp_value[i] = value * (self.gamma**depth)
        argmax = tmp_value.index(max(tmp_value))

        # the first critic,action of the best long term rollout
        critic = action_history_list[argmax][length]["critic"]
        action = action_history_list[argmax][length]["action"]
        self._log("max_trail_critic:", action_history_list[argmax][-1]["critic"])
        Alfred_action = self._convert_straction(action)
        return (Alfred_action, critic)

    def _get_start_idx(self, sample_pernode, depth):
        return (
            0 if depth == 0 else int((sample_pernode**depth - 1) / (sample_pernode - 1))
        )

    def _RAFA(self, metadata: str, predict=None, depth=2, sample_pernode=2):
        valid_idx = 0
        # convert the action into str form,as we only need it
        root_act_his = [
            {
                "metadata": action["metadata"],
                "action": str(action["action"]).replace("HLA: ", ""),
                "critic": action["critic"],
            }
            for action in self.action_history
        ]
        # there should be atmost sample_pernode**(depth+1) nodes
        tmp_act_his = [None] * ((sample_pernode ** (depth + 1)))
        # this refers to the metadata after taking action
        tmp_metadata = [None] * ((sample_pernode ** (depth + 1)))
        tmp_metadata[0] = metadata
        tmp_act_his[0] = copy.deepcopy(root_act_his)
        # depth-1 is the true thought depth
        # lots of basic calculate here
        for dep in range(depth):
            # get the parent node number, note that dep start from 0
            layer_samples = sample_pernode**dep
            parent_start_idx = self._get_start_idx(sample_pernode, dep)

            # the failed information should be given to action_proposal model
            if dep == 0 and self.log_action_failed:
                failed_info = self.failed_action
            else:
                failed_info = None
            tmp_act_his_list = []
            tmp_metadata_list = []
            for parent_num in range(layer_samples):
                parent_idx = parent_start_idx + parent_num
                # the model should return sample_pernode actions as a list
                tmp_act_his_list.append(tmp_act_his[parent_idx])
                tmp_metadata_list.append(tmp_metadata[parent_idx])

            actions_list = self.action_proposal.get_actions_threads(
                self.task,
                tmp_act_his_list,
                tmp_metadata_list,
                sample_pernode,
                self.predict_processor,
                failed_info,
                self.reflection,
                predict,
            )

            for parent_num in range(layer_samples):
                child_start_idx = (
                    self._get_start_idx(sample_pernode, dep + 1)
                    + sample_pernode * parent_num
                )
                actions = actions_list[parent_num]
                value_act_his_list = []
                for child_num in range(sample_pernode):
                    child_idx = child_start_idx + child_num
                    # the metadata before taking action
                    child_metadata = tmp_metadata[parent_idx]
                    tmp_act_his[child_idx] = copy.deepcopy(tmp_act_his[parent_idx])
                    tmp_act_his[child_idx].append(
                        {
                            "metadata": child_metadata,
                            "action": actions[child_num],
                            "critic": None,
                        }
                    )
                    # giving predict model action history, current metadata and current action, should return a new metadata
                    tmp_metadata[child_idx] = self.predict_processor.process(
                        self.predict.act(
                            self.task, tmp_act_his[child_idx], self.predict_processor
                        )
                    )
                    # giving value model action history, current metadata, current action and failed_info, should return a critic
                    value_act_his_list.append(tmp_act_his[child_idx])

                child_critic_ls = self.value.act_threads(
                    self.task, value_act_his_list, failed_info, self.reflection, predict
                )
                for child_num in range(sample_pernode):
                    child_idx = child_start_idx + child_num
                    child_critic = child_critic_ls[child_num]

                    tmp_act_his[child_idx][-1]["critic"] = child_critic
                    valid_idx = max(valid_idx, child_idx)

        # tmp_act_his[0] is the root action history, so we choose the highest cumulative reward start from 1
        return self._choose_action(tmp_act_his[1 : valid_idx + 1], len(root_act_his))

    def _beam_search(self, metadata: str, predict=None, depth=2, sample_per_node=2):
        root_act_his = [
            {
                "metadata": action["metadata"],
                "action": str(action["action"]).replace("HLA: ", ""),
                "critic": action["critic"],
            }
            for action in self.action_history
        ]
        root_node = Beam_Node(root_act_his, 0)
        beam = Beam(sample_per_node)
        if self.log_action_failed:
            failed_info = self.failed_action
        else:
            failed_info = None
        tmp_act_his_list = []
        tmp_metadata_list = []
        ##the first layer
        actions_list = self.action_proposal.get_actions_threads(
            self.task,
            [root_act_his],
            [metadata],
            sample_per_node,
            self.predict_processor,
            failed_info,
            self.reflection,
            predict,
        )
        actions_list = list(set(actions_list[0]))
        for act in actions_list:
            tmp_act = copy.deepcopy(root_act_his)
            tmp_act.append({"metadata": metadata, "action": act})
            tmp_act_his_list.append(tmp_act)
        child_critic_ls = self.value.act_threads(
            self.task, tmp_act_his_list, failed_info, self.reflection, predict
        )
        for i in range(len(child_critic_ls)):
            acts = tmp_act_his_list[i][-1]
            acts["critic"] = child_critic_ls[i]
            score = self._get_critic_value(acts["critic"])
            beam.add(root_node, acts, score)

        for dep in range(depth - 1):
            new_beam = Beam(sample_per_node)
            for node in beam.get():

                new_metadata = self.predict.act(
                    self.task, node.action_history, self.predict_processor
                )
                acts = self.action_proposal.get_actions_threads(
                    self.task,
                    [node.action_history],
                    [new_metadata],
                    sample_per_node,
                    self.predict_processor,
                    failed_info,
                    self.reflection,
                    predict,
                )
                acts = list(set(acts[0]))
                tmp_act_his_list = []
                for act in acts:
                    tmp_act = node.get_history()
                    tmp_act.append({"action": act, "metadata": new_metadata})
                    tmp_act_his_list.append(tmp_act)
                critics = self.value.act_threads(
                    self.task, tmp_act_his_list, failed_info, self.reflection, predict
                )
                for i in range(len(critics)):
                    act = tmp_act_his_list[i][-1]
                    act["critic"] = critics[i]
                    score = self._get_critic_value(critics[i])
                    new_beam.add(node, act, score)
            beam = new_beam
        max_score = -1
        max_node = None
        for i, node in enumerate(beam.get()):

            if node.get_score() > max_score:
                max_score = node.get_score()
                max_node = node

        self._log("max_trail_critic:", max_node.action_history[-1]["critic"])
        length = len(root_act_his)
        action = max_node.action_history[length]["action"]
        critic = max_node.action_history[length]["critic"]

        return (action, critic)

    def act(
        self, observation_or_state_repr: Union[Observation, StateRepr], md=None
    ) -> Action:
        # call below function to create a action, where act_type_str can be create through AlfredSubgoal.action_type_intid_to_str
        # and arg_vector_out is a   torch.Size([1, 125]) one hot vector where 1 indicate which object to interact with
        # more detail could be view   /hlsm/lgp/models/alfred/hlsm/hlsm_subgoal_model.py:in _sample_subgoal
        # AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector_out)

        self._log(
            "action his  at begin:",
            [act["action"] for act in self.action_history],
        )
        if isinstance(observation_or_state_repr, Observation):
            observation = observation_or_state_repr
            observation = observation.to(self.device)
            s_0 = self.obs_func(observation, self.agent_state.prev_state, goal=None)
        else:
            observation = observation_or_state_repr.observation
            s_0 = observation_or_state_repr

        # the meta data estimated by adaptation_model

        rep = s_0.data.data
        meta_info = []
        # use hlsm obj detector to get what have seen
        for i in range(124):
            if torch.any(rep[0][i] >= 1):
                obj_name = object_intid_to_string(i)
                if obj_name in _INTERACTIVE_OBJECTS:
                    meta_info.append(obj_name)
        # use adapt_model to predict useful obj
        meta_info = ", ".join(meta_info)
        predict = None
        if self.failed_action != None:
            proposed_action = self.failed_action["action"]
            proposed_action = self._convert_straction(proposed_action)
            critic = "use prev failed action"
        elif len(self.action_history) < 20:
            if self.stored_action:
                proposed_action, critic = self.stored_action.popleft()
                proposed_action = self._convert_straction(proposed_action)
            else:
                if self.use_predict:
                    predict = self.adaptation_model.act(meta_info, self.task)
                    self._log("orin prid", predict)
                    if self.config['predict_type']=='object':
                        predict = self.predict_processor.process_with_metadata(
                            predict, meta_info
                        )
                proposed_action, critic = self._beam_search(meta_info, predict)
                tmp_action, critic = self._process_act(
                    proposed_action, meta_info, critic
                )
                while tmp_action != proposed_action:
                    proposed_action = tmp_action
                    tmp_action, critic = self._process_act(
                        proposed_action, meta_info, critic
                    )
                proposed_action = self._convert_straction(tmp_action)

        else:
            proposed_action = self._convert_straction("Stop: NIL")
            critic = "Stop due to act history too long"
        # elif self.keep_act==True:
        #     proposed_action = self._convert_straction(self.failed_action["action"])
        #     critic = self.failed_action["critic"]
        #     self.keep_act = False
        proposed_action = proposed_action.to(self.device)
        # to get mask
        proposed_action = self.proposal.forward_inference(
            proposed_action, s_0, self.agent_state.task_repr, self.proposal.get_state()
        )

        self.proposal.log_action(proposed_action)
        self.agent_state.prev_state = s_0

        self._log_action(
            meta_info, str(proposed_action).replace("HLA:", ""), critic, predict
        )
        return proposed_action

    def get_prompt(self, predict):

        prompt = "Your task is: " + self.task + "\n"
        if predict != None:
            prompt += "The objects might be useful in the tasks are:" + predict + "\n"
        prompt += his_to_str(self.action_history)
        return prompt

    def act_debug(self, meta_info, predict=None, md=None):
        # call below function to create a action, where act_type_str can be create through AlfredSubgoal.action_type_intid_to_str
        # and arg_vector_out is a   torch.Size([1, 125]) one hot vector where 1 indicate which object to interact with
        # more detail could be view   /hlsm/lgp/models/alfred/hlsm/hlsm_subgoal_model.py:in _sample_subgoal
        # AlfredSubgoal.from_type_str_and_arg_vector(act_type_str, arg_vector_out)

        # the meta data estimated by adaptation_model

        if predict == None:
            predict = self.adaptation_model.act(meta_info, self.task)
        predict = self.predict_processor.process(predict)
        if self.keep_act == False:
            proposed_action, critic = self._RAFA(meta_info, predict)
        else:
            proposed_action = self._convert_straction(self.failed_action["action"])
            critic = self.failed_action["critic"]
            self.keep_act = False

        # to get mask

        self._log_action(
            meta_info, str(proposed_action).replace("HLA:", ""), critic, predict
        )
        return str(proposed_action), self.get_prompt(predict)
        # proposed_action = self.proposal.forward_inference(
        #     proposed_action, s_0, self.agent_state.task_repr, self.proposal.get_state()
        # )

        # self.proposal.log_action(proposed_action)
        # self.agent_state.prev_state = s_0

        # self._log_action(
        #     meta_info, str(proposed_action).replace("HLA:", ""), critic, predict
        # )
        # return proposed_action


import compress_pickle as pickle

if __name__ == "__main__":
    predict = adap_model()
    from lgp.parameters import Hyperparams, load_experiment_definition

    exp_def = Hyperparams(
        load_experiment_definition("alfred/eval/hlsm_full/eval_hlsm_valid_unseen")
    )
    model_factory = HlsmModelFactory(exp_def.Hyperparams)
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/"
    actprop = LLMHlsmSubgoalModel(exp_def.Hyperparams)
    from lgp.models.alfred.hlsm.hlsm_observation_function import HlsmObservationFunction

    obs_func = HlsmObservationFunction(exp_def.Hyperparams)
    subgoal_model_path = (
        "/mnt/sda/yuxiao_code/hlsm/models/alfred_hlsm_subgoal_model_e5.pytorch"
    )
    if subgoal_model_path:
        sd = torch.load(subgoal_model_path)
        actprop.load_state_dict(sd, strict=False)
    actprop.eval()
    actprop.to("cuda")
    obs_func.eval()
    agent = LlmAgent(actprop, obs_func)

    file_path = os.path.join(read_path, f"rollout_{5400}.gz")

    roll = pickle.load(file_path)
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
    for i, sg in enumerate(roll):
        task = sg["task"]
        agent.start_new_rollout(task)
        obs = sg["observation"]
        action = agent.act(obs)
        print(str(action))
        break

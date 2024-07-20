import json
import os
import sys
with open('/mnt/sda/yuxiao_code/hlsm/data/rollouts/progress_log.json','r') as f:
    data = json.load(f)
new_data={"collected_rollouts":[]}
root = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts"
for item in data["collected_rollouts"]:
    file_p = os.path.join(root,f'rollout_{item}.gz')
    
    #     sys.exit()
    if os.path.exists(file_p):
        new_data["collected_rollouts"].append(item)
with  open('/mnt/sda/yuxiao_code/hlsm/data/rollouts/progress_log_1.json','w') as d:
    json.dump(new_data,d)
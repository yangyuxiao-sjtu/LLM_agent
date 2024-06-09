import re

with open("/mnt/sda/yuxiao_code/hlsm/results/output100.txt", "r") as f:
    text = f.read()

pattern = r"'goal_conditions_met':\s*\((\d+),\s*(\d+)\)"
matches = re.findall(pattern, text)
GC = 0
AG = 0
SR = 0
TK = 0
# 打印找到的所有匹配项
for match in matches:
    # print(f"First number: {match[0]}, Second number: {match[1]}")
    GC += int(match[0])
    AG += int(match[1])
    if match[0] == match[1]:
        SR += 1
    TK += 1
print(GC / AG)
print(SR / TK)

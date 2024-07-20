import copy


class Beam_Node:
    def __init__(self, action_history, score=0, state=0):
        if isinstance(action_history, list) and len(action_history) > 0:
            self.action_history = copy.deepcopy(action_history)
        else:
            self.action_history = []
        self.score = score
        self.state = 0

    def get_state(self):
        return self.state

    def get_history(self):
        return copy.deepcopy(self.action_history)

    def get_score(self):
        return self.score

    def __eq__(self, other):
        if isinstance(other, Beam_Node):
            return self.action_history == other.action_history
        return False

    def __str__(self):
        return f"action_history:{self.action_history}\nscore{self.score}"


class Beam:

    def __init__(self, sample_per_node=3):
        self.sample_per_node = sample_per_node
        self.beam = []

        self.gamma = 0.9

    def get(self):
        return self.beam

    def is_full(self):
        return len(self.beam) == self.sample_per_node

    def add(self, old_node: Beam_Node, act, state):
        old_acts = old_node.get_history()
        old_score = old_node.get_score()
        last_state = old_node.get_state()
        new_acts = old_acts + [act]
        reward = state - last_state
        new_score = old_score + reward * (self.gamma ** (len(old_acts)))

        new_node = Beam_Node(new_acts, new_score, state)
        for item in self.beam:
            if item == new_node:
                item.score = (item.score + new_node.score) / 2
                return
        if self.is_full() == False:
            self.beam.append(new_node)
        else:
            self.beam.append(new_node)
            self.beam.sort(key=lambda x: x.get_score(), reverse=True)
            self.beam.pop()

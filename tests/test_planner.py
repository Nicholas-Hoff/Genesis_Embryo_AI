import planner

class DummyWorldModel:
    def __init__(self, rewards):
        self.rewards = rewards

    def prepare_input(self, state, action):
        return (state, action)

    def __call__(self, inp):
        return inp

    def post_process(self, state, delta):
        # delta is (state, action)
        return delta[1]

    def estimate_reward(self, state):
        # here state represents the action chosen
        return self.rewards[state]


def test_plan_selects_highest_reward_action():
    rewards = {'a': 1.0, 'b': 5.0, 'c': 2.0}
    wm = DummyWorldModel(rewards)
    planner_obj = planner.MCTSPlanner(wm, rollout_depth=3, samples=4)
    best = planner_obj.plan(state=None, actions=list(rewards))
    assert best == 'b'

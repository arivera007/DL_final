import torch
import numpy as np
# import .utils as extract_features
from torch.distributions import Bernoulli
from .player import Team

class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # return torch.nn.Linear(2*5*3, 1, bias=False)
        # return torch.nn.Linear(17, 1, bias=False)
        self.classifier = torch.nn.Linear(17, 3, bias=False)
    
    def forward(self, x):
        # f = self.network(x)
        # return self.classifier(f.mean(dim=(2,3)))
        return self.classifier(x)

class DumbActor:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
    
    def __call__(self, features):
        output = self.action_net(torch.as_tensor(features).view(1,-1))[0]
        # output = self.action_net(torch.as_tensor(f).view(1,-1))[0]
        print(output)
        return output

from os import path
class Actor(Team):
    def __init__(self, action_net, device):
        self.team = None
        self.num_players = None
        self.step = 0
        if action_net is None:
            self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'jurgen_agent.pt'))
            # self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'adriana_agent.pt'))
            print("Loaded adriana_agent.pt")
        else:
            self.model = action_net.to(device).eval()
        self.trayectories_data = []  # List of list of dicts. Each dict is a trajectory data. Each list is a player. Each list of dicts is a match.
    
    # def __call__(self, track_info, kart_info, **kwargs):
    def act(self, player_state, opponent_state, soccer_state):
        # actions = super().act(player_state, opponent_state, soccer_state)
        actions = []
        # for pstate in player_state:
        #     features = extract_features(pstate, soccer_state, opponent_state, self.team)
        #     acceleration, steer, brake = self.model(features)
        #     actions.append(dict(acceleration=acceleration, steer=steer, brake=brake))

        trayectories = []
        for player_idx, pstate in enumerate(player_state):
            features = extract_features(pstate, soccer_state, opponent_state, self.team)
            acceleration, steer, brake = self.model(features)
            greedy_actions_player = dict(acceleration=acceleration, steer=steer, brake=brake)
            # actions.append(actions_player)
            # features = extract_features(pstate, soccer_state, opponent_state, self.team)
            # action_player = actions[player_idx]
            # Sample from each Bernoulli distribution
            acc_dist = Bernoulli(logits=greedy_actions_player['acceleration']) # Should just be 1
            acceleration = acc_dist.sample()
            steer_dist = Bernoulli(logits=greedy_actions_player['steer'])
            # steer = steer_dist.sample()*2-1
            steer = steer_dist.sample()
            break_dist = Bernoulli(logits=greedy_actions_player['brake'])
            break_b = break_dist.sample() > 0.5
            actions_player = dict(acceleration=acceleration, steer=steer, brake=break_b)
            actions.append(actions_player)
            trajectory_data = {}
            # Score should be dephased by T-1
            trajectory_data['score'] = soccer_state['score'][self.team] # TODO: is this correct? the team is the same as team_id? 0 vs 1?
            trajectory_data['actions'] = actions_player 
            trajectory_data['features'] = features
            trayectories.append(trajectory_data)
        self.trayectories_data.append(trayectories)
        return actions 



# class GreedyActor:
#     def __init__(self, action_net):
#         self.action_net = action_net.cpu().eval()
    
#     def __call__(self, track_info, kart_info, **kwargs):
#         f = state_features(track_info, kart_info)
#         output = self.action_net(torch.as_tensor(f).view(1,-1))[0]

#         action = pystk.Action()
#         action.acceleration = 1
#         action.steer = output[0]
#         return action


def save_model(model):
    from os import path
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, path.join(path.dirname(path.abspath(__file__)), 'adriana_agent.pt'))
    torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


# def save_model(model):
#     from os import path
#     return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


# def save_model(model):
#     from torch import save
#     from os import path
#     if isinstance(model, Planner):
#         return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
#     raise ValueError("model type '%s' not supported!" % str(type(model)))


# def load_model():
#     from torch import load
#     from os import path
#     r = Planner()
#     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
#     return r

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

# Using same feature as per eDX: https://edstem.org/us/courses/43106/discussion/3548655
def extract_features(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of opponents 
    opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]]
    opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]]

    kart_to_opponent0 = (opponent_center0 - kart_center) / torch.norm(opponent_center0-kart_center)
    kart_to_opponent1 = (opponent_center1 - kart_center) / torch.norm(opponent_center1-kart_center)

    kart_to_opponent0_angle = torch.atan2(kart_to_opponent0[1], kart_to_opponent0[0]) 
    kart_to_opponent1_angle = torch.atan2(kart_to_opponent1[1], kart_to_opponent1[0]) 

    kart_to_opponent0_angle_difference = limit_period((kart_angle - kart_to_opponent0_angle)/np.pi)
    kart_to_opponent1_angle_difference = limit_period((kart_angle - kart_to_opponent1_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0]) 
    kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle)/np.pi)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, opponent_center0[0],
        opponent_center0[1], opponent_center1[0], opponent_center1[1], kart_to_opponent0_angle, kart_to_opponent1_angle, 
        goal_line_center[0], goal_line_center[1], puck_to_goal_line_angle, kart_to_puck_angle_difference, 
        kart_to_opponent0_angle_difference, kart_to_opponent1_angle_difference, 
        kart_to_goal_line_angle_difference], dtype=torch.float32)

    return features 

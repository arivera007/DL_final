# Im_jurgen_vsAI4

import numpy as np
import torch
from grader.runner import TeamRunner, Match, MatchException #, MAX_TIME_IMAGE, MAX_TIME_STATE, STEPS_PER_MATCH
STEPS_PER_MATCH = 1200
MAX_TIME_IMAGE = 0.05 * STEPS_PER_MATCH
MAX_TIME_STATE = 0.01 * STEPS_PER_MATCH

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 


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





def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


# def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
#     dataset = SuperTuxDataset(dataset_path, transform=transform)
#     return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

# def load_data(dataset_path=DATASET_PATH, batch_size=128):
def load_data(dataset_path, batch_size=128):
    data = []
    # dataset = SuperTuxDataset(dataset_path, transform=transform)
    # return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    # for chunk in load_recording('recordings/recordings.pkl'):

    # data_path = 'data/train/AIvsIm_jurgen4.pkl'
    my_team_id = 1 # Red team (Blue=0, red=1)
    my_team_str = 'team2_state' # Red team = 1 = team2_state, Blue team = 0 = team1_state
    opponent_str = 'team1_state'
    data = data + form_data(dataset_path, my_team_id, my_team_str, opponent_str)

    # hardcoded_path = 'data/train/Im_jurgen_vsAI4.pkl'
    # my_team_id = 0 # Blue team (Blue=0, red=1)
    # my_team_str = 'team1_state' # Red team = 1 = team2_state, Blue team = 0 = team1_state
    # opponent_str = 'team2_state'
    # data = data + form_data(hardcoded_path, my_team_id, my_team_str, opponent_str)


    # # TODO: Hack for now
    # red_dataset_paths = ['AI_Jurgen_Red1.pkl', 'geoffry_Jurgen_Red.pkl', 'yann_Jurgen_Red.pkl']
    # blue_dataset_paths = ['Jurgen_AI_Blue.pkl', 'Jurgen_YannBlue.pkl', 'Jurgen_yoshuaBlue.pkl']
    
    # # Load when I am RED team
    # for path in red_dataset_paths:
    #     base_path = 'data/train_truth/red/'
    #     # I have control of calling this fn so opponent will always be team1
    #     my_team_id = 1 # Red team (Blue=0, red=1)
    #     my_team_str = 'team2_state' # Red team = 1 = team2_state, Blue team = 0 = team1_state
    #     opponent_str = 'team1_state'
    #     data = data + form_data(base_path+path, data, my_team_id, my_team_str, opponent_str)

    # # Load when I am BLUE team
    # for path in blue_dataset_paths:
    #     base_path = 'data/train_truth/blue/'
    #     # I have control of calling this fn so opponent will always be team1
    #     my_team_id = 0 # Red team (Blue=0, red=1)
    #     my_team_str = 'team1_state' # Red team = 1 = team2_state, Blue team = 0 = team1_state
    #     opponent_str = 'team2_state'
    #     data = data + form_data(base_path+path, data, my_team_id, my_team_str, opponent_str)
    
    return data

# def form_data(dataset_path, data, my_team_id, my_team_str, opponent_str):
def form_data(dataset_path, my_team_id, my_team_str, opponent_str):
    data = []
    for chunk in load_recording(dataset_path):
        players_state = chunk[my_team_str]
        for num_player, player in enumerate(players_state):
            player_state = player
            # opponent_state = chunk['team1_state']
            opponent_state = chunk[opponent_str]
            soccer_state = chunk['soccer_state']
            actions = chunk['actions']
            # print(actions)

            features = extract_features(player_state, soccer_state, opponent_state, my_team_id) # Tensor of shape (17,)

            # For labels mus return: acceleration, steer, brake
            # labels = torch.tensor(actions, dtype=torch.float32)
            player_actions = actions[num_player*2 + my_team_id] # TODO: check if this is correct, I only see 2 sets of actions in the recording, shouldn't there be 4?
            # print(team_actions)
            if len(player_actions) == 0: # TODO, check if this is the right way to handle this (continue? vs maybe is something I need to codify)
                print('skipping this chunk')
                continue
            labels = (player_actions['acceleration'], player_actions['steer'], player_actions['brake'])
            data.append((features, labels))

    return data


# import pystk
# import ray        
# To implemente RL agent 
# @ray.remote
# class Rollout:
#     def __init__(self, screen_width, screen_height, hd=True, track='lighthouse', render=True, frame_skip=1):
#         # Init supertuxkart
#         if not render:
#             config = pystk.GraphicsConfig.none()
#         elif hd:
#             config = pystk.GraphicsConfig.hd()
#         else:
#             config = pystk.GraphicsConfig.ld()
#         config.screen_width = screen_width
#         config.screen_height = screen_height
#         pystk.init(config)
        
#         self.frame_skip = frame_skip
#         self.render = render
#         race_config = pystk.RaceConfig(track=track)
#         self.race = pystk.Race(race_config)
#         self.race.start()
    
#     def __call__(self, agent, n_steps=200):
#         torch.set_num_threads(1)
#         self.race.restart()
#         self.race.step()
#         data = []
#         track_info = pystk.Track()
#         track_info.update()

#         for i in range(n_steps // self.frame_skip):
#             world_info = pystk.WorldState()
#             world_info.update()

#             # Gather world information
#             kart_info = world_info.players[0].kart

#             agent_data = {'track_info': track_info, 'kart_info': kart_info}
#             if self.render:
#                 agent_data['image'] = np.array(self.race.render_data[0].image)

#             # Act
#             action = agent(**agent_data)
#             agent_data['action'] = action

#             # Take a step in the simulation
#             for it in range(self.frame_skip):
#                 self.race.step(action)

#             # Save all the relevant data
#             data.append(agent_data)
#         return data

# class Rollout_Runner:
#     def __init__(self, num_rollouts=10) -> None:
#         self.rollouts = [Rollout.remote(50, 50, hd=False, render=False, frame_skip=5) for i in range(num_rollouts)]

#     def rollout_many(self, many_agents, **kwargs):
#         ray_data = []
#         for i, agent in enumerate(many_agents):
#             ray_data.append(self.rollouts[i % len(self.rollouts)].__call__.remote(agent, **kwargs) )
#         return ray.get(ray_data)

# # def dummy_agent(**kwargs):
# #     action = pystk.Action()
# #     action.acceleration = 1
# #     return action

class Unique_Match:
    def __init__(self):
        self.match = Match(use_graphics=False)

class HockyRunner(TeamRunner):
    """
        Similar to TeamRunner but this module takes Team object as inputs instead of the path to module
    """
    def __init__(self, team):
        self._team = team
        self.agent_type = self._team.agent_type

class Rollout_Runner:
    def __init__(self, my_team, unique_match, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.student_model = HockyRunner(self.module.Team())
        self.student_model = HockyRunner(my_team)
        # self.match = Match(use_graphics=self.student_model.agent_type == 'image')
        self.match = unique_match
        self.verbose = False

    def _test(self, agent_name):
        time_limit = MAX_TIME_STATE if self.student_model.agent_type == 'state' else MAX_TIME_IMAGE

        test_model = TeamRunner(agent_name)
        ball_locations = [
            [0, 1]#,
            # [0, -1],
            # [1, 0],
            # [-1, 0],
        ]
        scores = []
        results = []

        try:
            # for bl in ball_locations:
            #     result = self.match.run(self.student_model, test_model, 2, STEPS_PER_MATCH, max_score=3,
            #                        initial_ball_location=bl, initial_ball_velocity=[0, 0],
            #                        record_fn=None, timeout=time_limit, verbose=self.verbose)
            #     scores.append(result[0])
            #     results.append(f'{result[0]}:{result[1]}')

            for bl in ball_locations:
                result = self.match.run(test_model, self.student_model, 2, STEPS_PER_MATCH, max_score=3,
                                   initial_ball_location=bl, initial_ball_velocity=[0, 0],
                                   record_fn=None, timeout=time_limit, verbose=self.verbose)
                scores.append(result[1])
                results.append(f'{result[1]}:{result[0]}')
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)
            assert 0
        return sum(scores), results

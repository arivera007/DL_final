# from .planner import Planner, save_model 
from .adriana_agent import ActionNet, save_model, DumbActor, Actor
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data, Rollout_Runner, Unique_Match
import copy
from torch.distributions import Bernoulli

def train(args):
    from os import path
    # train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_path = 'data/train/AIvsIm_jurgen4.pkl' # Best so far and submitted.
    # data_path = 'data/train/AIvsIm_jurgen2400.pkl'
    n_epochs = args.num_epoch
    lr = args.learning_rate
    wd = args.weight_decay
    batch_size = args.batch_size
    opponent_agent = 'yann_agent'


    # RL Initialization. Get best outcome from dumb agents.
    n_trajectories = args.num_trayectories #teacher 10
    n_iterations = args.num_iterations #teacher ?
    n_steps = 1200 # 600 ?
    T = 20
    num_players = 2

    unique_match = Unique_Match()
    # many_action_nets = [ActionNet() for i in range(n_trajectories)]
    # many_actors = [Actor(action_net) for action_net in many_action_nets]
    # many_rollouts = [Rollout_Runner(my_team=actor, unique_match=unique_match.match) for actor in many_actors]
    # roll_data = [rollout._test(opponent_agent) for rollout in many_rollouts]
    # # roll_data = [rollout._test('geoffrey_agent') for rollout in many_rollouts]
    # print(f'len(roll_data): {len(roll_data)}')
    # print(f'roll_data[TOP THREE]: {roll_data[0:3]}')
    # idx_max_score = np.argmax([d[0] for d in roll_data])
    # # print(f'max_score: {roll_data[idx_max_score]}')
    # good_initialization = many_action_nets[idx_max_score]

    # best_init_model = copy.deepcopy(good_initialization)
    # model = copy.deepcopy(good_initialization)

    
    
    # SECOND ATTEMPT TO BEST INITIALIZATION
    # Start training
    data = load_data(data_path)
    train_data = data
    global_step = 0
    init_epochs = 50
    model = ActionNet()
    model.train().to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iteration, loss_val = None, None
    for epoch in range(init_epochs):
        for iteration in range(0, len(train_data), batch_size):
            batch_ids = torch.randint(0, len(train_data), (batch_size,), device=device)
            batch = [train_data[i] for i in batch_ids] # TODO: Do this right with tensors
            batch_features = [data_point[0] for data_point in batch]
            batch_features_tensor = torch.stack(batch_features, dim=0)
            batch_labels = [torch.tensor((data_point[1][0].item(), data_point[1][1].item(), data_point[1][2].item())) for data_point in batch]
            batch_labels_tensor = torch.stack(batch_labels, dim=0)         
            o = model(batch_features_tensor)
            loss_val = loss(o, batch_labels_tensor)
            global_step += 1
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss_val}')






    # best_init_actor = Actor(None, device)
    model = copy.deepcopy(model)
    model.to(device).train()

    # # if args.continue_training:
    # #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
    # # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # data = load_data(data_path)
    # train_data = data
    # train_data = roll_data # TODO: split data into train and val ?

    loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(n_epochs):
        eps = 1e-2
        
        # Roll out the policy, compute the Expectation
        # best_init_actor = Actor(model)  #model = best_init_model
        actors = [Actor(model, device) for i in range(n_trajectories)]
        many_rollouts = [Rollout_Runner(my_team=actor, unique_match=unique_match.match) for actor in actors]
        trajectories = [rollout._test(opponent_agent) for rollout in many_rollouts]
        print('epoch = %d   best_dist = '%epoch, np.max([d[0] for d in trajectories]))

        # trajectories = rollout_many([Actor(action_net)]*n_trajectories, n_steps=600)
        # print('epoch = %d   best_dist = '%epoch, np.max([t[-1]['kart_info'].overall_distance for t in trajectories]))
        
        # Compute all the reqired quantities to update the policy
        features = []
        returns = []
        actions = []
        for rollout_idx, rollout in enumerate(many_rollouts):
            team_data = actors[rollout_idx].trayectories_data
            for chunk in team_data:
                # print(f'chunk: {chunk}')
                score_dephased = 0
                for player_id in range(num_players):
                    chunk_player = chunk[player_id]
                    # Compute the returns
                    returns.append( score_dephased )  # We only use the score as reward. TODO: record a trayectory and its intermediate states
                    score_dephased = chunk_player['score']
                    # Compute the features
                    features.append( torch.as_tensor(chunk_player['features'], dtype=torch.float32).to(device).view(-1) )
                    # Store the action that we took
                    # [torch.tensor((data_point[1][0].item(), data_point[1][1].item(), data_point[1][2].item())) for data_point in batch]
                    labels = chunk_player['actions']
                    labels_actions = torch.tensor((labels['acceleration'].item(), labels['steer'].item(), labels['brake'].item()))
                    actions.append(labels_actions)
        
        # Upload everything to the GPU
        returns = torch.as_tensor(returns, dtype=torch.float32).to(device)
        # actions = torch.as_tensor(actions, dtype=torch.float32).to(device)
        actions = torch.stack(actions).to(device)
        # print(f'shape actions: {actions.shape}')
        # print(f'actions: {actions}')
        features = torch.stack(features).to(device)
        
        if returns.std() != 0:
            returns = (returns - returns.mean()) / returns.std()
        
        model.train().to(device)
        avg_expected_log_return = []
        for it in range(n_iterations):
            batch_ids = torch.randint(0, len(returns), (batch_size,), device=device)
            batch_returns = returns[batch_ids]
            batch_actions = actions[batch_ids]
            batch_features = features[batch_ids]
            
            output = model(batch_features)
            # print(f'shape output: {output.shape}')
            # print(f'output: {output}')
            # pi = Bernoulli(logits=output[:,0])
            pi = Bernoulli(logits=output)
            # print('pi: ', pi)
            # print(f'shape pi: {pi.shape}')
            
            # expected_log_return = (pi.log_prob(batch_actions)*batch_returns).mean()
            expected_log_return = pi.log_prob(batch_actions)
            # print(f'shape expected_log_return: {expected_log_return.shape}')
            # print(f'shape batch_returns: {batch_returns.shape}')
            # expected_log_return = expected_log_return * torch.transpose(batch_returns)
            expected_log_return = expected_log_return * torch.unsqueeze(batch_returns, dim=1)
            # expected_log_return = torch.bmm(expected_log_return, batch_returns)
            # print(f'matmul shape expected_log_return: {expected_log_return.shape}')
            # print(f'expected_log_return: {expected_log_return}')
            # print(f'batch_returns: {batch_returns}')
            expected_log_return = expected_log_return.mean()
            # expected_log_return = (pi.log_prob(batch_actions)*batch_returns).mean()
            optim.zero_grad()
            (-expected_log_return).backward()
            optim.step()
            avg_expected_log_return.append(float(expected_log_return))

    save_model(model)

        # best_performance, current_performance = rollout_many([GreedyActor(best_action_net), GreedyActor(action_net)], n_steps=600)
        # if best_performance[-1]['kart_info'].overall_distance < current_performance[-1]['kart_info'].overall_distance:
        #     best_action_net = copy.deepcopy(action_net)



# WROKING
    # # Start training
    # global_step = 0
    # model.train().to(device)
    # # logger = tb.SummaryWriter(log_dir+'/'+str(datetime.now()), flush_secs=1)
    # iteration, loss_val = None, None
    # for epoch in range(n_epochs):
    #     for iteration in range(0, len(train_data), batch_size):
    #         batch_ids = torch.randint(0, len(train_data), (batch_size,), device=device)
    #         # print(f'batch_ids: {batch_ids}')
    #         # print(f'len(train_data): {len(train_data)}')
    #         # print(f'train_data sample: {train_data[0:2]}')
    #         # batch_ids = batch_ids.tolist()
    #         batch = [train_data[i] for i in batch_ids] # TODO: Do this right with tensors
    #         # batch = train_data[batch_ids]
    #         # batch_features = [(data_point[0]) for data_point in batch]
    #         batch_features = [data_point[0] for data_point in batch]
    #         # print(f'batch_features: {batch_features[0:3]}')
    #         batch_features_tensor = torch.stack(batch_features, dim=0)
    #         batch_labels = [torch.tensor((data_point[1][0].item(), data_point[1][1].item(), data_point[1][2].item())) for data_point in batch]
    #         # print(f'batch_labels: {batch_labels[0:3]}')
    #         batch_labels_tensor = torch.stack(batch_labels, dim=0)         

    #         # batch_features = train_data[batch_ids]
    #         # batch_labels = train_data[batch_ids]

    #         o = model(batch_features_tensor)
    #         loss_val = loss(o, batch_labels_tensor)
    #         # print(f'loss_val: {loss_val}')

    #         # logger.add_scalar('train/loss', loss_val, global_step)
    #         global_step += 1

    #         optimizer.zero_grad()
    #         loss_val.backward()
    #         optimizer.step()
    #     print(f'epoch: {epoch}, iteration: {iteration}, loss: {loss_val}')
    # save_model(model)

        # model.eval()
        # for _ in range(val_num_batches):
        #     batch = make_random_batch(val_data, batch_size, seq_len+1)
        #     batch_data_valid = batch[:, :, :-1].to(device)
        #     batch_label_valid = batch.argmax(dim=1).to(device)
        #     o = model(batch_data_valid)
        #     loss_valid = loss(o, batch_label_valid)
        #     # if valid_logger is not None and global_step % 100 == 0:
        #     if valid_logger is not None:
        #         valid_logger.add_scalar('val loss', loss_valid, global_step)

        # print(f'epoch: {epoch}, loss: {loss_val}, validation loss: {loss_valid}')



# def log(logger, img, label, pred, global_step):
#     """
#     logger: train_logger/valid_logger
#     img: image tensor from data loader
#     label: ground-truth aim point
#     pred: predited aim point
#     global_step: iteration
#     """
#     import matplotlib.pyplot as plt
#     import torchvision.transforms.functional as TF
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(TF.to_pil_image(img[0].cpu()))
#     WH2 = np.array([img.size(-1), img.size(-2)])/2
#     ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
#     ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
#     logger.add_figure('viz', fig, global_step)
#     del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=120)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    # parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-t', '--num_trayectories', type=int, default=3)
    parser.add_argument('-it', '--num_iterations', type=int, default=3)  # Batch iterations
    # parser.add_argument('-t', '--transform',
    #                     # default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    #                     default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    # parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)

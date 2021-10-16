#!/usr/bin/env python3
from typing import Tuple, Optional

import numpy as np

from agent import Fish
from communicator import Communicator
from shared import SettingLoader


class FishesModelling:
    def init_fishes(self, n):
        fishes = {}
        for i in range(n):
            fishes["fish" + str(i)] = Fish()
        self.fishes = fishes


class PlayerController(SettingLoader, Communicator):
    def __init__(self):
        SettingLoader.__init__(self)
        Communicator.__init__(self)
        self.space_subdivisions = 10
        self.actions = None
        self.action_list = None
        self.states = None
        self.init_state = None
        self.ind2state = None
        self.state2ind = None
        self.alpha = 0
        self.gamma = 0
        self.episode_max = 300

    def init_states(self):
        ind2state = {}
        state2ind = {}
        count = 0
        for row in range(self.space_subdivisions):
            for col in range(self.space_subdivisions):
                ind2state[(col, row)] = count
                state2ind[count] = [col, row]
                count += 1
        self.ind2state = ind2state
        self.state2ind = state2ind

    def init_actions(self):
        self.actions = {
            "left": (-1, 0),
            "right": (1, 0),
            "down": (0, -1),
            "up": (0, 1)
        }
        self.action_list = list(self.actions.keys())

    def allowed_movements(self):
        self.allowed_moves = {}
        for s in self.ind2state.keys():
            self.allowed_moves[self.ind2state[s]] = []
            if s[0] < self.space_subdivisions - 1:
                self.allowed_moves[self.ind2state[s]] += [1]
            if s[0] > 0:
                self.allowed_moves[self.ind2state[s]] += [0]
            if s[1] < self.space_subdivisions - 1:
                self.allowed_moves[self.ind2state[s]] += [3]
            if s[1] > 0:
                self.allowed_moves[self.ind2state[s]] += [2]

    def player_loop(self):
        pass


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


def epsilon_greedy(Q,
                   state,
                   all_actions,
                   current_total_steps=0,
                   epsilon_initial=1.0,
                   epsilon_final=0.2,
                   epsilon_scheduler: Optional['ScheduleLinear'] = None,
                   eps_type="constant") -> int:
    if eps_type == 'constant':
        epsilon = epsilon_final
        # ADD YOUR CODE SNIPPET BETWEEN EX 3.1
        # Implemenmt the epsilon-greedy algorithm for a constant epsilon value
        # Use epsilon and all input arguments of epsilon_greedy you see fit
        # It is recommended you use the np.random module
        dice_value = np.random.rand()
        if dice_value <= epsilon:
            # random action
            action = np.random.choice(all_actions)
        else:
            # greedy choice
            action = np.nanargmax(Q[state, all_actions])
            action = all_actions[action]
        # ADD YOUR CODE SNIPPET BETWEEN EX 3.1

    elif eps_type == 'linear':
        # ADD YOUR CODE SNIPPET BETWEEN EX  3.2
        epsilon = epsilon_scheduler.value(current_total_steps)
        # Implement the epsilon-greedy algorithm for a linear epsilon value
        # Use epsilon and all input arguments of epsilon_greedy you see fit
        # use the ScheduleLinear class
        # It is recommended you use the np.random module
        dice_value = np.random.rand()
        if dice_value <= epsilon:
            # random action
            action = np.random.choice(all_actions)
        else:
            # greedy choice
            action = np.nanargmax(Q[state, all_actions])
            action = all_actions[action]
        # ADD YOUR CODE SNIPPET BETWEEN EX  3.2

    else:
        raise "Epsilon greedy type unknown"

    return action


# noinspection PyAttributeOutsideInit
class PlayerControllerRL(PlayerController, FishesModelling):
    def __init__(self):
        super().__init__()

    def player_loop(self):
        # send message to game that you are ready
        self.init_actions()
        self.init_states()
        self.alpha = self.settings.alpha
        self.gamma = self.settings.gamma
        self.epsilon_initial = self.settings.epsilon_initial
        self.epsilon_final = self.settings.epsilon_final
        self.annealing_timesteps = self.settings.annealing_timesteps
        self.threshold = self.settings.threshold
        self.episode_max = self.settings.episode_max

        Q = self.q_learning()

        # compute policy
        policy = self.get_policy(Q)

        # send policy
        msg = {"policy": policy, "exploration": False}
        self.sender(msg)

        msg = self.receiver()
        print("Q-learning returning")
        return

    def q_learning(self):
        ns = len(self.state2ind.keys())
        na = len(self.actions.keys())
        # initialization
        self.allowed_movements()
        # ADD YOUR CODE SNIPPET BETWEEN EX. 2.1
        # Initialize a numpy array with ns state rows and na state columns with float values from 0.0 to 1.0.
        Q = 0.01 * np.random.randn(ns, na)
        # ADD YOUR CODE SNIPPET BETWEEN EX. 2.1

        for s in range(ns):
            list_pos = self.allowed_moves[s]
            for i in range(4):
                if i not in list_pos:
                    Q[s, i] = np.nan

        # Q_old = Q.copy()

        diff = np.infty
        end_episode = False

        init_pos_tuple = self.settings.init_pos_diver
        init_pos = self.ind2state[(init_pos_tuple[0], init_pos_tuple[1])]
        episode = 0

        Q_best = np.copy(Q)
        R_best = -np.inf
        R_best_count = 0
        current_total_steps = 0

        # Initialize schedulers
        epsilon_scheduler = ScheduleLinear(schedule_timesteps=self.annealing_timesteps, final_p=self.epsilon_final,
                                           initial_p=self.epsilon_initial, curve_smoothness=10.0)
        alpha_scheduler = ScheduleLinear(schedule_timesteps=10 * self.annealing_timesteps, final_p=self.alpha,
                                         initial_p=self.alpha / 2, curve_smoothness=10.0)
        gamma_scheduler = ScheduleLinear(schedule_timesteps=10 * self.annealing_timesteps, final_p=self.gamma,
                                         initial_p=self.gamma, curve_smoothness=10.0)

        # Change the while loop to incorporate a threshold limit, to stop training when the mean difference
        # in the Q table is lower than the threshold
        while episode <= self.episode_max and diff > self.threshold and R_best_count < 5:

            s_current = init_pos
            R_total = 0
            steps = 0
            while not end_episode:
                # selection of action
                # Disable not allowed moves
                allowed_actions = self.allowed_moves[s_current]

                # Use the epsilon greedy algorithm to retrieve an action
                if np.all(np.isnan(Q[s_current, :])):
                    action = np.random.choice(allowed_actions)
                else:
                    action = epsilon_greedy(Q=Q, state=s_current, all_actions=allowed_actions,
                                            current_total_steps=current_total_steps,
                                            epsilon_initial=self.epsilon_initial, epsilon_final=self.epsilon_final,
                                            epsilon_scheduler=epsilon_scheduler, eps_type='linear')

                assert action in allowed_actions, f'action={action} | allowed_actions={allowed_actions} | ' + \
                                                  f'Q[s_current]={Q[s_current]}'

                # compute reward
                action_str = self.action_list[action]
                # print(f'state={self.state2ind[s_current]} | action={action_str}')
                msg = {"action": action_str, "exploration": True}
                self.sender(msg)

                # wait response from game
                msg = self.receiver()
                R = msg["reward"]
                R_total += R
                s_next_tuple = msg["state"]
                end_episode = msg["end_episode"]
                s_next = self.ind2state[s_next_tuple]

                # ADD YOUR CODE SNIPPET BETWEEN EX. 2.2
                # Get current learning rate
                lr = alpha_scheduler.value(t=current_total_steps)
                gamma = gamma_scheduler.value(t=current_total_steps)
                # Implement the Bellman Update equation to update Q
                current_q = Q[s_current, action] if not np.isnan(Q[s_current, action]) else 0.
                next_q = np.nanmax(Q[s_next, :])
                if np.isnan(next_q):
                    next_q = 0.
                Q[s_current, action] = (1 - lr) * current_q + lr * (R + gamma * next_q)
                # ADD YOUR CODE SNIPPET BETWEEN EX. 2.2

                s_current = s_next
                current_total_steps += 1
                steps += 1

            # Check Episode's total Rewards
            if R_total >= R_best:
                diff = np.nansum(np.abs(Q_best[:] - Q[:]))
                Q_best = np.copy(Q)
                if R_total == R_best:
                    R_best_count += 1
                else:
                    R_best = R_total
                    R_best_count = 0
            else:
                R_best_count = 0

            print("Episode: {}, Steps {}, Diff: {:6e}, Total Reward: {}, Total Steps {}"
                  .format(episode, steps, diff, R_total, current_total_steps))
            episode += 1
            end_episode = False

        return Q_best

    # noinspection PyBroadException
    def get_policy(self, Q):
        nan_max_actions_proxy = [None for _ in range(len(Q))]
        for s in range(len(Q)):
            try:
                # noinspection PyTypeChecker
                nan_max_actions_proxy[s] = np.nanargmax(Q[s])
            except:
                nan_max_actions_proxy[s] = np.random.choice([0, 1, 2, 3])

        nan_max_actions_proxy = np.array(nan_max_actions_proxy)

        policy = {}
        for s in self.state2ind.keys():
            policy[tuple(self.state2ind[s])] = self.action_list[nan_max_actions_proxy[s]]
        return policy


class ScheduleLinear(object):
    SCHEDULER_INDEX = 0

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0, curve_smoothness: float = 1.0,
                 plot_curve: bool = False):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.__class__.SCHEDULER_INDEX += 1
        if plot_curve:
            ex, self.epsilon_curve = ScheduleLinear.get_alpha_curve(num_iters=schedule_timesteps, return_x=True,
                                                                    alpha_multiplier=curve_smoothness,
                                                                    y_start=initial_p, y_end=final_p)
            import matplotlib.pyplot as plt
            plt.plot(ex, self.epsilon_curve)
            plt.xlabel('iter')
            plt.ylabel('epsilon')
            plt.title(f'Epsilon curve during the annealing steps (={schedule_timesteps})')
            plt.savefig(f'scheduler_{ScheduleLinear.SCHEDULER_INDEX}.pdf')
            plt.show()
        else:
            self.epsilon_curve = ScheduleLinear.get_alpha_curve(num_iters=schedule_timesteps, y_start=initial_p,
                                                                y_end=final_p, alpha_multiplier=curve_smoothness)

    def value(self, t):
        # ADD YOUR CODE SNIPPET BETWEEN EX 3.2
        # Return the annealed linear value
        return self.epsilon_curve[t if t < self.schedule_timesteps else -1]
        # ADD YOUR CODE SNIPPET BETWEEN EX 3.2

    @staticmethod
    def get_alpha_curve(num_iters: int, alpha_multiplier: float = 10.0, y_start: float = 0.0, y_end: float = 1.0,
                        return_x: bool = False) -> Tuple[np.ndarray, np.ndarray] or np.ndarray:
        """
        Return the sigmoid curve fro StyleGAN's alpha parameter.
        Source: https://github.com/achariso/gans-thesis
        :param (int) num_iters: total number of iterations (equals the number of points in curve)
        :param (float) alpha_multiplier: parameter which controls the sharpness of the curve (1=linear, 1000=delta at half
                                         the interval - defaults to 10 that a yields a fairly smooth transition)
        :param (float) y_start: initial y value in the resulting curve
        :param (float) y_end: final y value in the resulting curve
        :param (bool) return_x: set to True to have the method also return the x-values
        :return: either a tuple with x,y as np.ndarray objects  or y as np.ndarray object
        """
        if num_iters < 2:
            return np.array([y_end, ]) if not return_x else [0, ], np.array([y_end, ])
        x = np.arange(num_iters)
        c = num_iters // 2
        a = alpha_multiplier / num_iters
        y = 1. / (1 + np.exp(-a * (x - c)))
        y = y / (y[-1] - y[0])
        # Fix values
        y_diff = y_end - y_start
        if y_start > y_end:
            y = y[::-1]
            y_start, y_end = y_end, y_start
        if y_diff != 1.0:
            y = y * abs(y_diff) + y_start
        y += (y_end - max(y)) + 1e-14
        # Return
        if return_x:
            return x, y
        return y

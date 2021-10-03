#!/usr/bin/env python3
from random import choices
from abc import abstractmethod
from collections import defaultdict

import random
from sys import stderr

from constants import *
from hmm import HMM
from itertools import islice, count

from hmm_utils import argmax
from player_controller_hmm import PlayerControllerHMMAbstract


class PlayerControllerHMMTemp(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """

        # A list in which every observation is appended
        self.fish_observations = []
        # Set of guessed fish ids
        self.already_guessed = set()
        # Each type has a matrix. Each matrix is a list of observations for fish of that type
        self.observations_per_type = [ [] for _ in range(N_SPECIES)]

        self.total_guesses = 0
        self.total_guesses_correct = 0

    @abstractmethod
    def guess_impl(self, step, observations):
        """
        Method to be overridden by subsclasses, perform the guess
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        pass

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        """
        for n in range(N_FISH):
            if n not in self.already_guessed:
                self.already_guessed.add(n)
                return n, argmax(self.fish_counts)[1]
        return None
        """

        self.fish_observations.append(observations)
        guess = self.guess_impl(step, observations)
        if guess is not None:
            self.already_guessed.add(guess[0])
        self.total_guesses += 1
        if self.total_guesses >= N_FISH:
            print("Game over" , file=stderr)
        if guess is not None:
            print(f"Model guessing fish id {guess[0]} to be {guess[1]}", file=stderr)
        else:
            print("Model did not guess", file=stderr)
        return guess

        #####
        if step < 20:
            print("step ", step, file=stderr)
            return None
        if len(self.hmms) <= 1:
            # Return a random guess
            guess = step % N_FISH, random.randint(0, N_SPECIES - 1)
            print("Random Guessing ", guess, file=stderr)
            self.already_guessed.add(guess[0])
            return guess
        else:
            guess_fish_id = -1
            guess_fish_type = -1
            guess_val = -999999
            for n in range(N_FISH):
                if n in self.already_guessed:
                    continue
                max_i = -1
                max_val = -999999999
                for k, v in self.hmms.items():
                    # print(f"Calculating likelihood at type {k} for fish id {n}")
                    ll, _, _ = v.alpha_pass(observations)
                    if ll > max_val:
                        max_val = ll
                        max_i = k
                if max_val > guess_val:
                    guess_val = max_val
                    guess_fish_id = n
                    guess_fish_type = max_i

            guess = guess_fish_id, guess_fish_type
            print("Model Guessing ", guess, file=stderr)
            self.already_guessed.add(guess_fish_id)
            return guess

    @abstractmethod
    def reveal_impl(self, correct, fish_id, true_type, current_observations):
        """
        Method to be overriden by subclasses
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        pass

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        """
        self.fish_counts[true_type] += 1
        return
        """
        # Record fish data
        print(f"Guess for {fish_id} was {correct}", file=stderr)
        # Transpose fish observations
        fish_observations = zip(*self.fish_observations)
        current_observations: list = next(islice(fish_observations, fish_id, fish_id + 1))
        # Call reveal implementation
        self.reveal_impl(correct, fish_id, true_type, current_observations)
        if correct:
            self.total_guesses_correct += 1
        print(f"Model accuracy is {self.total_guesses_correct/ self.total_guesses:.2f}\\%", file=stderr)
        return


class RandomPlayerController(PlayerControllerHMMTemp):
    """
    Player controller that makes completely random guesses
    Scores history : 53 (0.92)
                     50 (0.94)

    local accuracies: 0.04
                      0.06
    """
    def guess_impl(self, step, observations):
        return step % N_FISH, random.randint(0, N_SPECIES - 1)

    def reveal_impl(self, correct, fish_id, true_type, current_observations):
        pass

class MostCommonPlayerController(PlayerControllerHMMTemp):
    """
    Player controller which always guesses the most probable fish
    Scores history:  21 (0.94)
                     21 (0.93)
                     25 (0.94) Change fish id from correct to steps % N_FISH
                     41 (0.94)       with random choice
                     52 (0.96)

    local accuracites: 0.12
                       0.12
                       0.20 with random choise
    """

    RANDOM_CHOICE = True
    def __init__(self):
        PlayerControllerHMMTemp.__init__(self)
        self.fish_counts = [0] * N_SPECIES

    def guess_impl(self, step, observations):
        print("counts ", self.fish_counts, file=stderr)

        if self.RANDOM_CHOICE:
            total_fish = sum(self.fish_counts) + 1
            max_i = choices(range(N_SPECIES), weights=[x / total_fish for x in self.fish_counts])[0]
        else:
            max, max_i = argmax(self.fish_counts)
        #for n in range(N_FISH):
            #if n not in self.already_guessed:
        return step % N_FISH, max_i
        #return None

    def reveal_impl(self, correct, fish_id, true_type, current_observations):
        self.fish_counts[true_type] += 1


class SimpleHMMPlayerController(PlayerControllerHMMTemp):
    """
    Kattis 45 (3.24), num_hidden=3, noise=0.01
           47 (3.65), num_hidden=3, noise=0.0001

    Local accuracies: 0.06
    """
    def __init__(self):
        PlayerControllerHMMTemp.__init__(self)
        # One hmm for each type of fish
        #self.hmms = defaultdict(lambda: HMM(3, N_EMISSIONS))
        self.hmms = [HMM(3, N_EMISSIONS) for _ in range(N_SPECIES)]

    def guess_impl(self, step, observations):
        if step < 20:
            return None

        guess_fish_id = -1
        guess_fish_type = -1
        guess_val = -999999
        for n in range(N_FISH):
            if n in self.already_guessed:
                continue
            max_i = -1
            max_val = -999999999
            for k, v in enumerate(self.hmms):
                # print(f"Calculating likelihood at type {k} for fish id {n}")
                ll, _, _ = v.alpha_pass(observations)
                if ll > max_val:
                    max_val = ll
                    max_i = k
            if max_val > guess_val:
                guess_val = max_val
                guess_fish_id = n
                guess_fish_type = max_i

        guess = guess_fish_id, guess_fish_type
        self.already_guessed.add(guess_fish_id)
        return guess

    def reveal_impl(self, correct, fish_id, true_type, current_observations):
        # Record current fish type's observations
        self.observations_per_type[true_type].append(current_observations)
        print(f"Starting training model {true_type} for fish ", fish_id, file=stderr)
        self.hmms[true_type].baum_welch(current_observations)
        print(f"Finished training model {true_type} for fish ", fish_id, file=stderr)


PlayerControllerHMM = SimpleHMMPlayerController

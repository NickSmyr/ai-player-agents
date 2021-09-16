# Q1

 Initial state is the position of all the fishes, 
 of the players (and hooks) as well as the time remaining. 
 
 The transition function is the function that with input a 
 state (which information about which player is playing next)
 outputs a set of states that can be next depending on the players
 move. For fishing derby the transition function will move the 
 fish according to the observations, move the player who has the
 turn, and reject states that cannot occur (such as a player moving
 right when the opponent is always right)
 
 The possible states are then the states that can be output by any previous 
 output states by the transition function, with the initial state being the
 only state at the start.
 
# Q2

The terminal states are all the states which satisfy either of the 
following conditions

- Time has run out
- All fish have been caught

# Q3

The heuristic ν(A, s) = Score(Green boat) − Score(Red boat) mentioned 
is a good heuristic because, given enough search depth, the player A 
will look to catch fish to increase his score, and also to do it faster
than the other player (player A will act so he increases his score or reduce 
the opponent's score.

# Q4

V will best approximate the utility function if, whenever it is applied on a state,
it will return the utility of the terminal state, if player A searches the 
entire possible tree and uses the minimax strategy.

# Q5

An example would be that player A has has caught fish with scores 4 and 5, 
so his score is 9, but player B is one move away from catching a fish with
score 10

# Q6

The heuristic in question only counts the winning and losing states at
the leaf nodes. However it does not account for the adversary's actions. 
In the chess example, after moving the queen to a4, black has a lot of 
moves with which they can lose, and only one with which the game can continue. 


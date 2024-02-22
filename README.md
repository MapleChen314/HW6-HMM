# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


## Overview 

This is an implementation of the forward and Viterbi algorithms to calculate 1. the probability of observing a certain sequence, and 2. the most likely sequence of hidden states that may have led to a certain observation sequence.

Its implementation uses the following inputs:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)
* `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 



## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [x] complete the `forward` function in the HiddenMarkovModelClass <br>
  [x] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [x] Ensure functionality on mini and full weather dataset <br>
  [x] Account for edge cases 

[TODO] Packaging <br>
  [x] Update README with description of your methods <br>
  [x] pip installable module (optional)<br>
  [x] github actions (install + pytest) (optional)

import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        forward_probability=None
        n_obs=len(input_observation_states)
        n_hidden_states=len(self.hidden_states)
        alpha=np.zeros((n_hidden_states,n_obs))
        
        if n_obs>0:
            # Step 2. Calculate probabilities
            for obs_idx in range(n_obs): #The 0th column is determined by the prior p
                if obs_idx==0:
                    for i in range(n_hidden_states):
                        if input_observation_states[0] not in self.observation_states:
                            raise ValueError("Given observed states are not in the initialized set!!")
                        alpha[i,0]=self.prior_p[i]*self.emission_p[i,self.observation_states_dict[input_observation_states[0]]]
                else:
                    alpha_sum=0
                    for hidden_state_iter in range(n_hidden_states):
                        for prev_hidden_state in range(n_hidden_states):
                            if input_observation_states[obs_idx] not in self.observation_states:
                                raise ValueError("Given observed states are not in the initialized set!!")
                            alpha_sum+=alpha[prev_hidden_state,obs_idx-1]\
                                *self.transition_p[prev_hidden_state,hidden_state_iter]\
                                *self.emission_p[hidden_state_iter,self.observation_states_dict[input_observation_states[obs_idx]]]
                        alpha[hidden_state_iter,obs_idx]=alpha_sum
            
            # Step 3. Return final probability 
            forward_probability=sum(alpha[:,-1])
        return forward_probability
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(self.hidden_states),len(decode_observation_states)))
        #store best path for traceback
        best_path = np.zeros((len(self.hidden_states),len(decode_observation_states)))
        n_obs=len(decode_observation_states)
        n_hidden_states=len(self.hidden_states)
        for i in range(n_hidden_states):
            if decode_observation_states[0] not in self.observation_states:
                raise ValueError("Given observed states are not in the initialized set!!")
            viterbi_table[i,0]=self.prior_p[i]*self.emission_p[i,self.observation_states_dict[decode_observation_states[0]]]
       # Step 2. Calculate Probabilities
        for obs_i in range(1,n_obs):
            for hidden_state_iter in range(n_hidden_states): #Iterate down each column
                term_list=[]
                for prev_hidden_iter in range(n_hidden_states):
                    term=viterbi_table[prev_hidden_iter,obs_i-1]*self.transition_p[prev_hidden_iter,hidden_state_iter]
                    term_list.append(term)
                if decode_observation_states[obs_i] not in self.observation_states:
                    raise ValueError("Given observed states are not in the initialized set!!")
                viterbi_table[hidden_state_iter,obs_i]=max(term_list)*self.emission_p[hidden_state_iter,self.observation_states_dict[decode_observation_states[obs_i]]] 
                best_path[hidden_state_iter,obs_i]=np.argmax(np.array(term_list))
        # Step 3. Traceback 
        best_hidden=[]
        current_hidden_idx=int(np.argmax(np.array(viterbi_table[:,-1])))
        current_hidden=self.hidden_states_dict[current_hidden_idx]
        best_hidden=[current_hidden]+best_hidden
        for j in range(1,n_obs):
            back_i=n_obs-j
            temp_idx=best_path[current_hidden_idx,back_i]
            current_hidden_idx=int(temp_idx)
            current_hidden=self.hidden_states_dict[current_hidden_idx]
            best_hidden=[current_hidden]+best_hidden

        # Step 4. Return best hidden state sequence 
        return best_hidden
        
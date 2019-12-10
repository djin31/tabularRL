# tabularRL
Implementation of tabular methods in reinforcement learning for policy control and evaluation.

We demonstrate different policy evaluation (Monte Carlo, k step Temporal Difference) and control (k step SARSA, TD lambda, Monte Carlo)  algorithms for the game of BlackJack.

## Code
### Environment
`blackjack_simulator.py` is an implementation of BlackJack simulator. It also defines an abstract class for `Environment`. Any suitable class derived from this abstract class can be substituted in place of `BlackJack` to run the RL algorithms for a different setting.

### RL Algorithms
`evaluate_policy.py` contains implementation for different policy evaluation algorithms while `control_policy.py` contains policy control algorithms. It also contains `test_policy` method to evaluate any policy by computing average rewards by acting out in the environment according to the given policy.

### Demo code
`tabular_rl.py` implements a wrapper over the implemented algorithms providing a clean interface for benchmarking the algorithms. 
For a more interactive guide and visualization of algorithms, and observing the effect of different hyperparameters refer to `BlackJackDemo.ipynb` notebook.



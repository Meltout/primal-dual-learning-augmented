import numpy as np
import matplotlib.pyplot as plt

class SkiRentalProblemInstance:
  def __init__(self, N, B):
    self.N = N
    self.B = B
    self.t = 0

  def make_observation(self):
    if self.t == self.N:
      return "not snowing"
    else:
      self.t += 1
      return "snowing"

  def OPT_cost(self):
    return min(self.N, self.B)

class Predictor:
  def __init__(self, problem_instance:SkiRentalProblemInstance, config:dict):
    self.N_pred = None
    if config.get('error_distribution') == 'normal':
      sigma = config.get('sigma')
      self.N_pred = int(max(1, np.round(np.random.normal(problem_instance.N, sigma, 1)).item()))
    elif config.get('error_distribution') == 'optimistic':
      delta = config.get('delta')
      self.N_pred = problem_instance.N + delta
    elif config.get('error_distribution') == 'pessimistic':
      delta = config.get('delta')
      self.N_pred = max(1, problem_instance.N - delta)
    elif config.get('error_distribution') == 'constant-prediction':
      self.N_pred = config.get('constant')

class SkiRentalPDLA:
  """
  instances_params: N and B values on which the algo will be tested
  """

  def __init__(self, NBs):
    self.NBs = NBs

  def run(self, problem_instance:SkiRentalProblemInstance, lam, N_pred):
    B = problem_instance.B
    def e(z):
      return pow((1 + 1 / B), z * B)
    if N_pred >= B:
      # Prrediction suggests buying
      c = e(lam)
      c_prime = 1
    else:
      # Prediction suggests renting
      c = e(1 / lam)
      c_prime = lam
    """
    In order to get a randomized integral solution, we arrange the increments of x on the interval [0, 1] and
    choose uniformly in random a number in [0, 1]. We buy on the day corresponding to the increment of x to
    which the random number belongs. It can be shown that the probability of buying on the jth day is exactly
    the change in the value of x on the jth day and the probability of renting the skis on the jth day is exactly
    f_j. Thus, the expected cost of the randomized algorithm is the same as the cost of the fractional algorithm.

    Quoted from: Niv Buchbinder, Kamal Jain, and Joseph (Seffi) Naor. Online primal-dual algorithms for
    maximizing ad-auctions revenue
    """
    x = 0
    x_threshhold_to_buy = np.random.rand()
    cost = 0
    while problem_instance.make_observation() == 'snowing':
      x = (1 + 1 / B) * x + 1 / ((c - 1) * B)
      if x > x_threshhold_to_buy:
        cost += B
        break
      else:
        cost += 1
    return cost

  def monte_carlo_eval_competitive_ratio(self, N, B, predictor_config, lam, nb_simulations=1000):
    competitive_ratios = []
    for _ in range(nb_simulations):
      problem_instance = SkiRentalProblemInstance(N, B)
      predictor = Predictor(problem_instance, predictor_config)
      cost_alg = self.run(problem_instance, lam, predictor.N_pred)
      competitive_ratios.append(cost_alg / problem_instance.OPT_cost())
    return np.mean(competitive_ratios).item()

  def compute_statistics(self, predictor_configs):
    lams = np.linspace(0.01, 0.99, 100)
    for config in predictor_configs:
      competitive_ratios = []
      for lam in lams:
        competitive_ratios.append(np.mean([self.monte_carlo_eval_competitive_ratio(N, B, config, lam) for (N, B) in self.NBs]).item())
      print(competitive_ratios)
      plt.plot(lams, competitive_ratios, label=config.get('name'))

    plt.xlabel('Lambda')
    plt.ylabel('Expected performance over OPT')
    plt.legend()
    plt.savefig('plots/statistics.png')

if __name__ == '__main__':
  configs = [
    {
      'name':'normal',
      'error_distribution':'normal',
      'sigma':5
    },
    {
      'name':'overly-optimistic',
      'error_distribution':'optimistic',
      'delta':5
    },
    {
      'name':'overly-pessimistic',
      'error_distribution':'pessimistic',
      'delta':5
    },
    {
      'name':'constant-prediction',
      'error_distribution':'constant-prediction',
      'constant':1
    }
  ]

  # generate problem instances with the same cost of OPT
  NBs = [(N, 7) for N in range(1, 21)]

  ski_pdla = SkiRentalPDLA(NBs)
  ski_pdla.compute_statistics(configs)
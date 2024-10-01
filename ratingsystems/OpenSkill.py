import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted
from openskill.models import ThurstoneMostellerFull as OS
from openskill.models import ThurstoneMostellerFullRating as OSRating
from scipy.stats import norm
from copy import deepcopy



# draw_margin = 8.3 * teamsize^0.5 * erfinv(p_draw)
# Doesn't support repeated participants in different teams
class OpenSkill_System(RatingSystem):
    """
    An open source system developed as an alternative to TrueSkill.
    
    It functions very similarly to TrueSkill but without patented mechanisms 
    (mainly the factor graphs).

    https://openskill.me/en/stable/

    As with TrueSkill, this implementation doesn't doesn't fully support rating groups 
    as well as repeated participants on different teams or 0-weight participants.

    
    :param init_mu: Initial performance average for new participants.
    :type init_mu: float
    :param init_sigma: Initial measurement uncertainty.
    :type init_sigma: float
    :param beta: Performance standard deviation of all participants.
    :type beta: float
    :param tau: Dynamic factor which accounts for change in performance over time.
    :type tau: float
    :param kappa: A very small number to limit how low the standards deviations are allowed to fall.
    :type kappa: float
    :param draw_margin: Minimum performance difference required for a victory.
    :type draw_margin: float      
    """
    def __init__(self, init_mu=25, init_sigma=8.3, beta=4.2, tau=0.083, kappa=0.0001, draw_margin=0.1, **_):
        RatingSystem.__init__(self, rating_class=OpenSkill_Rating, init_mu=init_mu,
                              init_sigma=init_sigma)
        self.draw_margin = draw_margin
        self.tau = tau
        self.kappa = kappa
        self.beta = beta
        self.env = OS(mu=init_mu, sigma=init_sigma, beta=beta, tau=tau, kappa=kappa)
        self.add_metric(InformationPreempted("InformationPreempted", draw_margin=draw_margin))

    
    def add_game(self, game):  
        _, _, results_matrix, h2h_ratings = self._add_game_metrics_update(game)
        if len(self.interaction_groups) == 0:
            ratings = []
            weights = []            
            for team_ind, rating_list in enumerate(h2h_ratings):
                ratings.append([rating for rating in rating_list[0 if team_ind!=0 else 1]])
                weights.append([rating.recent_partials[team_ind] for rating in rating_list[0 if team_ind!=0 else 1]])
            try:
                new_ratings = self.env.rate(teams=ratings,scores=game["Results"],weights=weights)
            except (FloatingPointError, ZeroDivisionError):
                new_ratings = ratings
            for team_ind, team in enumerate(new_ratings):
                for rating_ind, rating in enumerate(team):
                    ratings[team_ind][rating_ind].mu = rating.mu
                    ratings[team_ind][rating_ind].sigma = rating.sigma
        else:
            self._update_participants(h2h_ratings, results_matrix=results_matrix, env=self.env)

    def _result_probabilities(self, team1_ratings, team2_ratings, team1_ind, team2_ind):
        total_mu = 0
        total_var = 0
        for rating in team1_ratings:
            total_mu += rating.mu * rating.recent_partials[team1_ind]
            total_var += rating.var * rating.recent_partials[team1_ind]
        for rating in team2_ratings:
            total_mu -= rating.mu * rating.recent_partials[team2_ind]
            total_var += rating.var * rating.recent_partials[team2_ind]
        z_win = (total_mu-self.draw_margin) / (total_var**0.5)
        z_loss = (-total_mu-self.draw_margin) / (total_var**0.5)
        win_prob = norm.cdf(z_win)
        loss_prob = norm.cdf(z_loss)
        return win_prob, 1-win_prob-loss_prob, loss_prob

class OpenSkill_Rating(Rating, OSRating):
    def __init__(self, init_mu, init_sigma):
        OSRating.__init__(self, mu=init_mu, sigma=init_sigma)
        self.rating_components = ["mu", "std"]

    def _pre_update(self, h2h_ratings, team_ind, results_matrix, env, **context):
        self.new_mu = self.mu
        self.new_sigma = self.sigma
        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            team2_ratings = h2h_ratings[team2_ind][team_ind]
            weights = []            
            weights.append([rating.recent_partials[team_ind] for rating in team1_ratings])
            weights.append([rating.recent_partials[team2_ind] for rating in team2_ratings])
            ratings = []
            ratings.append([rating for rating in team1_ratings])
            ratings.append([rating for rating in team2_ratings])
            rating_ind = ratings[0].index(self)
            ranks = list(results_matrix[team_ind,team2_ind,-1::-2])   
            ratings_copy = deepcopy(ratings)
            try:
                new_ratings = env.rate(teams=ratings_copy,ranks=ranks,weights=weights)
            except (FloatingPointError, ZeroDivisionError):
                new_ratings = ratings_copy
            self.new_mu += new_ratings[0][rating_ind].mu - self.mu
            self.new_sigma *= new_ratings[0][rating_ind].sigma/self.sigma

    def _update(self, h2h_ratings, team_ind, **context):
        self.mu = self.new_mu
        self.sigma = self.new_sigma

    def __str__(self):
        return f"Rating: {self.mu:.2f}\tDev: {self.std:.2f}"

    def __repr__(self):
        return f"Rating: {self.mu:.2f}\tDev: {self.std:.2f}"

    
    @property
    def std(self):
        return (self.system.beta**2 + self.sigma**2)**0.5

    @std.setter
    def std(self, value):
        print("WARNING: std is calculated from beta and sigma and can't be set")

    @std.deleter
    def std(self):
        print("WARNING: std is calculated from beta and sigma and can't be deleted")

    @property
    def var(self):
        return self.system.beta**2 + self.sigma**2

    @var.setter
    def var(self, value):
        print("WARNING: var is calculated from beta and sigma and can't be set")

    @var.deleter
    def var(self):
        print("WARNING: var is calculated from beta and sigma and can't be deleted")

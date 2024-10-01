import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted
from trueskill import TrueSkill as TS
from trueskill import Rating as TSRating


# draw_margin = 8.3 * teamsize^0.5 * erfinv(p_draw)
# Doesn't support repeated participants in different teams
class TrueSkill_System(RatingSystem):
    """
    The rating system created by Microsoft in 2005. 
    
    It models matches using factor graphs and uses Gaussian density filtering to 
    estimate the parameters of the graphs. Each rating consists of the average 
    performance and the measurement uncertainty, every player is assumed to have the 
    same performance standard deviation.

    It extended the core principles behind Elo to work for multi-player teams, multi-team games,
    and to predict draw probabilities.

    https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/

    This package uses the implementation by Heungsub Lee and doesn't fully support 
    rating groups as well as repeated participants on different teams or 0-weight participants.
    
    Note: Microsoft has a patent on using factor graphs for rating systems, so 
    make sure you're legally allowed to use this system before using it.
    
    :param init_mu: Initial performance average for new participants.
    :type init_mu: float
    :param init_sigma: Initial measurement uncertainty.
    :type init_sigma: float
    :param beta: Performance standard deviation of all participants.
    :type beta: float
    :param tau: Dynamic factor which accounts for change in performance over time.
    :type tau: float
    :param draw_margin: Minimum performance difference required for a victory.
    :type draw_margin: float      
    """
    def __init__(self, init_mu=25, init_sigma=8.3, beta=4.2, tau=0.083, draw_margin=0, **_):
        RatingSystem.__init__(self, rating_class=TrueSkill_Rating, init_mu=init_mu,
                              init_sigma=init_sigma)
        self.draw_margin = draw_margin
        self.tau = tau
        self.beta = beta
        self.env = TS(mu=init_mu, sigma=init_sigma, beta=beta, tau=tau, draw_probability=self._calc_draw)
        self.add_metric(InformationPreempted("InformationPreempted", draw_margin=draw_margin))
    
    def add_game(self, game):  
        _, _, results_matrix, h2h_ratings = self._add_game_metrics_update(game)
        if len(self.interaction_groups) == 0:
            ratings = []
            weights = {}            
            for team_ind, rating_list in enumerate(h2h_ratings):
                for rating in rating_list[0 if team_ind!=0 else 1]:
                    weights[rating] = rating.recent_partials[team_ind]
                ratings.append({rating:rating for rating in rating_list[0 if team_ind!=0 else 1]})
            try:
                new_ratings = self.env.rate(rating_groups=ratings,ranks=-np.array(game["Results"]),weights=weights)
            except (FloatingPointError, ZeroDivisionError):
                new_ratings = ratings
            for team in new_ratings:
                for rating in team:
                    rating.mu = team[rating].mu
                    rating.sigma = team[rating].sigma
                    # rating.sigma = rating.sigma * self.temp
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
        win_prob = self.env.cdf(z_win)
        loss_prob = self.env.cdf(z_loss)
        return win_prob, 1-win_prob-loss_prob, loss_prob

    def _calc_draw(self, team1_rating, team2_rating, env):
        total_mu = team1_rating.mu - team2_rating.mu
        total_var = team1_rating.sigma**2 + team2_rating.sigma**2
        z_win = (total_mu-self.draw_margin) / (total_var**0.5)
        z_loss = (-total_mu-self.draw_margin) / (total_var**0.5)
        win_prob = self.env.cdf(z_win)
        loss_prob = self.env.cdf(z_loss)
        return 1-win_prob-loss_prob

class TrueSkill_Rating(Rating, TSRating):
    def __init__(self, init_mu, init_sigma):
        TSRating.__init__(self, mu=init_mu, sigma=init_sigma)
        self.rating_components = ["mu", "std"]

    def _pre_update(self, h2h_ratings, team_ind, results_matrix, env, **context):
        self.new_mu = self.mu
        self.new_sigma = self.sigma
        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            team2_ratings = h2h_ratings[team2_ind][team_ind]
            weights = {}
            for rating in team1_ratings:
                weights[rating] = rating.recent_partials[team_ind]
            for rating in team2_ratings:
                weights[rating] = rating.recent_partials[team2_ind]
            ratings = []
            ratings.append({rating:rating for rating in team1_ratings})
            ratings.append({rating:rating for rating in team2_ratings})
            ranks = results_matrix[team_ind,team2_ind,-1::-2]
            try:
                new_ratings = env.rate(rating_groups=ratings,ranks=ranks,weights=weights)
            except (FloatingPointError, ZeroDivisionError):
                new_ratings = ratings
            self.new_mu += new_ratings[0][self].mu - self.mu
            self.new_sigma *= new_ratings[0][self].sigma/self.sigma

    def _update(self, h2h_ratings, team_ind, **context):
        self.mu = self.new_mu
        self.sigma = self.new_sigma

    def __str__(self):
        return f"Rating: {self.mu:.0f}\tDev: {self.std:.0f}"

    def __repr__(self):
        return f"Rating: {self.mu:.0f}\tDev: {self.std:.0f}"
    

    @property
    def mu(self):
        return self.pi and self.tau / self.pi

    @mu.setter
    def mu(self, value):
        self.tau = value * self.pi

    @property
    def sigma(self):
        return (1 / self.pi)**2 if self.pi else np.inf

    @sigma.setter
    def sigma(self, value):
        self.pi = (1 / value)**0.5

    
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

import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted
from datetime import datetime

# Implement Trueskill
PI_SQR = np.pi**2
MAX_ITER = 10
EPSELON = 1e-6

def _g(var):
    return (1 + 3 * var / PI_SQR) ** -0.5

def _E(rating, var, g=None):
    if g is None:
        g = _g(var)
    return (1 + np.e**(-g * rating)) ** -1

def _S(rating, var, draw_margin, g=None):
    if g is None:
        g = _g(var)
    win_prob = _E(rating - draw_margin, var, g)
    loss_prob = _E(-rating - draw_margin, var, g)
    draw_prob = 1 - win_prob - loss_prob
    return win_prob + 0.5 * draw_prob

# just use the defaults for this one
class Glicko2_System(RatingSystem):
    """
    The rating system created by Mark Glickman in the 2000s. 
    
    It builds on Glicko by varying the rate at which RDs decrease over time.

    http://www.glicko.net/glicko.html

    Here it was extended to rate players within teams, allow multi-team game formats, 
    and predict the probability of draws.

    That said, the implementation in this package doesn't perform very well since 
    the ideal rating should include 5-10 games and here ratings are updated after 
    every game. 
    
    As an attempt to compensate for this, two "k" parameters were added that scale the 
    updates in rating and RD.

    :param init_rating: Rating given to new players.
    :type init_rating: float
    :param init_RD: RD given to new players.
    :type init_RD: float
    :param init_sigma: Volatility given to new players.
    :type init_sigma: float
    :param tau: Constraint on how much volatility is allowed to change per game.
    :type tau: float
    :param k_rating: K factor that scales the rating updates.
    :type k_rating: float
    :param k_var: K factor that scales the RD updates (more accurately, the 
        variance updates).
    :type k_var: float
    :param draw_margin: Margin used to predict draws. Can be thought of as the 
        rating of a player that plays against every team (and if it wins it's a draw).
    :type draw_margin: float      
    """
    def __init__(self, init_rating=0, init_RD=2, init_sigma=0.06, tau=0.2, k_rating=1, k_var=1, draw_margin=1, **_):
        RatingSystem.__init__(self, rating_class=Glicko2_Rating, init_rating=init_rating, 
                              init_RD=init_RD, init_sigma=init_sigma, tau=tau, 
                              k_rating=k_rating, k_var=k_var)
        self.draw_margin = draw_margin
        self.add_metric(InformationPreempted("InformationPreempted", draw_margin=draw_margin))

    def add_game(self, game):   
        _, _, results_matrix, h2h_ratings = self._add_game_metrics_update(game)
        game_period = game["Date"] if "Date" in game else self.games
        scores = np.sum(results_matrix[:,:] * [1,0.5,0], axis = 2)
        self._update_participants(h2h_ratings,scores=scores, game_period=game_period)
        
    def _result_probabilities(self, team1_ratings, team2_ratings, team1_ind, team2_ind):
        total_rating = 0
        total_var = 0
        for rating in team1_ratings:
            total_rating += rating.rating * rating.recent_partials[team1_ind]
            total_var += rating.var * rating.recent_partials[team1_ind]
        for rating in team2_ratings:
            total_rating -= rating.rating * rating.recent_partials[team2_ind]
            total_var += rating.var * rating.recent_partials[team2_ind]        

        expected_win = _E(total_rating-self.draw_margin,total_var)
        expected_loss = _E(-total_rating-self.draw_margin,total_var)
        expected_draw = 1 - expected_win - expected_loss

        return expected_win, expected_draw, expected_loss

class Glicko2_Rating(Rating):
    def __init__(self, init_rating, init_RD, init_sigma, tau, k_rating, k_var):
        self.rating_components = ["rating", "RD"]
        self.rating = init_rating
        self.RD = init_RD
        self.init_var = init_RD**2
        self.sigma = init_sigma
        self.tau = tau
        self.k_rating = k_rating
        self.k_var = k_var
        self.last_played = None

    @property
    def RD(self):
        return self.var**0.5
    @RD.setter
    def RD(self, value):
        self.var = value**2

    @property
    def sigma(self):
        return self.varvol**0.5
    @sigma.setter
    def sigma(self, value):
        self.varvol = value**2

    def _pre_update(self, h2h_ratings, team_ind, game_period, **context):
        rating_periods_passed = self._calc_rps_passed(game_period)
        self.var = min(self.init_var, self.var + self.varvol*(rating_periods_passed))
        self.old_rating = self.rating
        self.old_var = self.var

    
    def _update(self, h2h_ratings, team_ind, scores, **context):
        g_array = np.zeros(len(h2h_ratings))
        E_array = np.zeros(len(h2h_ratings))

        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            team2_ratings = h2h_ratings[team2_ind][team_ind]
            total_rating = 0
            total_var = 0
            for rating in team1_ratings:
                total_rating += rating.old_rating * rating.recent_partials[team_ind]
                total_var += rating.old_var * rating.recent_partials[team_ind]
            for rating in team2_ratings:
                total_rating -= rating.old_rating * rating.recent_partials[team2_ind]   
                total_var += rating.old_var * rating.recent_partials[team2_ind]   
            total_var -= self.old_var * self.recent_partials[team_ind]
            g_array[team2_ind] = _g(total_var)
            E_array[team2_ind] = _S(total_rating, total_var, self.system.draw_margin, g_array[team2_ind])
        
        v = np.sum(g_array**2 * E_array * (1-E_array)) ** -1
        g_error_sum = np.sum(g_array * (scores[team_ind] - E_array))
        Delta_sqr = (g_error_sum * v) ** 2 

        self.varvol = self._varvol_iter(Delta_sqr, v)
        # self.var = min(self.init_var, self.old_var + self.varvol)
        self.var = (1/self.var + self.k_var * self.recent_partials[team_ind]/v) ** -1
        self.rating += self.old_var * g_error_sum * self.recent_partials[team_ind] * self.k_rating

    def _varvol_iter(self, Delta_sqr, v):
        a = np.log(self.varvol)
        f = lambda x: self._f(x,Delta_sqr,v,a)
        A = a
        f_A = f(A)
        if Delta_sqr > self.var + v:
            B = np.log(Delta_sqr - self.var - v)
            f_B = f(B)
        else:
            k_iter = 1
            B = a - k_iter * self.tau
            f_B = f(B)
            while f_B < 0 and k_iter < MAX_ITER:
                k_iter += 1
                B = a - k_iter * self.tau
                f_B = f(B)
        iter_counter = 0
        while abs(A-B)>EPSELON and iter_counter < MAX_ITER:
            iter_counter += 1
            C = A + (A-B) * f_A / (f_B-f_A)
            f_C = f(C)
            if f_C*f_B <= 0:
                A = B
                f_A = f_B
            else:
                f_A /= 2
            B = C
            f_B = f_C

        return np.exp(A)

    def _f(self, x, Delta_sqr, v, a):
            ret = np.exp(x) * (Delta_sqr - self.var - v - np.exp(x))
            ret /= 2 * (self.var + v + np.exp(x)) ** 2
            ret -= (x-a) / (self.tau**2)
            return ret
    
    def _calc_rps_passed(self, game_period):
        if isinstance(game_period, str):
            if self.last_played is None:
                self.last_played = datetime.fromisoformat(game_period)
                return 0
            game_date = datetime.fromisoformat(game_period)
            delta = game_date - self.last_played
            periods_passed = delta.days + delta.seconds / 86400
            self.last_played = game_date
            return periods_passed
        if self.last_played is None:
            self.last_played = game_period
            return 0
        periods_passed = game_period - self.last_played
        self.last_played = game_period
        return periods_passed

    def __str__(self):
        return f"Rating: {173.7178*self.rating+1500:.0f}\tRD: {173.7178*self.RD:.0f}"
    
    def __repr__(self):
        return f"Rating: {self.rating:.0f}\tRD: {self.RD:.0f}"
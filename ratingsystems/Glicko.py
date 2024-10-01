import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted
from datetime import datetime

PI_SQR = np.pi**2
Q = np.log(10) / 400
Q_SQR = Q**2

def _g(var):
    return (1 + 3 * Q_SQR * var / PI_SQR) ** -0.5

def _E(rating, var, g=None):
    if g is None:
        g = _g(var)
    return (1 + 10**(-_g(var) * rating / 400)) ** -1

def _S(rating, var, draw_margin, g=None):
    if g is None:
        g = _g(var)
    win_prob = _E(rating - draw_margin, var, g)
    loss_prob = _E(-rating - draw_margin, var, g)
    draw_prob = 1 - win_prob - loss_prob
    return win_prob + 0.5 * draw_prob

# if using dates: c = 346 / sqrt(225 * daysbetweengames)
# else:           c = 346 / sqrt(225 * activeparticipants/playersperteam)
class Glicko_System(RatingSystem):
    """
    The rating system created by Mark Glickman in the 1990s. 
    
    It builds on Elo by accounting for measurement uncertainty.

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
    :param c: Increase in uncertainty between games.
    :type c: float
    :param k_rating: K factor that scales the rating updates.
    :type k_rating: float
    :param k_var: K factor that scales the RD updates (more accurately, the 
        variance updates).
    :type k_var: float
    :param draw_margin: Margin used to predict draws. Can be thought of as the 
        rating of a player that plays against every team (and if it wins it's a draw).
    :type draw_margin: float      
    """
    def __init__(self, init_rating=1500, init_RD=350, c=10, k_rating=1, k_var=1, draw_margin=1, **_):
        RatingSystem.__init__(self, rating_class=Glicko_Rating, init_rating=init_rating,
                              init_RD=init_RD, c_sqr = c**2, k_rating=k_rating, k_var=k_var)
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

class Glicko_Rating(Rating):
    def __init__(self, init_rating, init_RD, c_sqr, k_rating, k_var):
        self.rating_components = ["rating", "RD"]
        self.rating = init_rating
        self.RD = init_RD
        self.init_var = init_RD**2
        self.c_sqr = c_sqr
        self.last_played = None
        self.k_rating = k_rating
        self.k_var = k_var

    @property
    def RD(self):
        return self.var**0.5
    @RD.setter
    def RD(self, value):
        self.var = value**2

    def _pre_update(self, h2h_ratings, team_ind, game_period, **context):
        rating_periods_passed = self._calc_rps_passed(game_period)
        self.var = min(self.init_var, self.var + self.c_sqr*rating_periods_passed)
        self.old_rating = self.rating
        self.old_var = self.var

    def _update(self, h2h_ratings, team_ind, scores, **context):
        g_array = np.zeros(len(h2h_ratings))
        E_array = np.zeros(len(h2h_ratings))
        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            # print(team_ind, team2_ind)
            # print(h2h_ratings)
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
        d2_rec = Q_SQR * np.sum(g_array**2 * E_array * (1-E_array))
        self.var = (1/self.var + self.k_var * d2_rec * self.recent_partials[team_ind]) ** -1
        g_error_sum = np.sum(g_array * (scores[team_ind] - E_array))
        self.rating +=  Q * self.k_rating * self.old_var * g_error_sum * self.recent_partials[team_ind]

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
        return f"Rating: {self.rating:.0f}\tRD: {self.RD:.0f}"
    
    def __repr__(self):
        return f"Rating: {self.rating:.0f}\tRD: {self.RD:.0f}"
import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted

# K = 60 / (players_per_game * (n_teams-1))
class Elo_System(RatingSystem):
    """
    The rating system created by Arpad Elo in the 1960s.

    https://en.wikipedia.org/wiki/Elo_rating_system

    Here it was extended to rate players within teams, allow multi-team game formats, 
    and predict the probability of draws.

    :param init_rating: Rating given to new players.
    :type init_rating: float
    :param k: K factor used to calculate rating updates.
    :type k: float
    :param draw_margin: Margin used to predict draws. Can be thought of as the 
        rating of a player that plays against every team (and if it wins it's a draw).
    :type draw_margin: float      
    """
    def __init__(self, init_rating=1200, k=30, draw_margin=10, **_):
        RatingSystem.__init__(self, rating_class=Elo_Rating, init_rating=init_rating, k=k)
        self.draw_margin = draw_margin
        self.add_metric(InformationPreempted("InformationPreempted", draw_margin=draw_margin))

    def add_game(self, game):
        _, pred_matrix, results_matrix, h2h_ratings = self._add_game_metrics_update(game)
        error_matrix =  results_matrix@[1,0.5,0] - pred_matrix@[1,0.5,0]
        self._update_participants(h2h_ratings,error_matrix=error_matrix)
        

    def _result_probabilities(self, team1_ratings, team2_ratings, team1_ind, team2_ind):
        total_rating = 0
        for rating in team1_ratings:
            total_rating += rating.rating * rating.recent_partials[team1_ind]
        for rating in team2_ratings:
            total_rating -= rating.rating * rating.recent_partials[team2_ind]

        expected_win = 1 / (1 + 10 ** ((-total_rating+self.draw_margin) / 400))
        expected_loss = 1 / (1 + 10 ** ((total_rating+self.draw_margin) / 400))
        expected_draw = 1 - expected_win - expected_loss

        return expected_win, expected_draw, expected_loss


class Elo_Rating(Rating):
    def __init__(self, init_rating, k):
        self.rating_components = ["rating"]
        self.rating = init_rating
        self.k = k
    
    def _pre_update(self, h2h_ratings, team_ind, **context):
        pass
    
    def _update(self, h2h_ratings, team_ind, error_matrix, **context):
        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            self.rating += self.k * error_matrix[team_ind, team2_ind] * self.recent_partials[team_ind]
    def __str__(self):
        return f"{self.rating:.0f}"
    

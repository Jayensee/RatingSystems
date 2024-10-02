import numpy as np
from .RatingSystem import *
from .Metrics import InformationPreempted, GMeanLikelihood
from scipy import special

import datetime as dt

SQRT_2PI = (2*np.pi)**0.5
SQRT_2 = 2**0.5
MIN_VAR = 1e-4
MAX_VAR= 1e6

class JRS_System(RatingSystem):
    """
    The rating system created by Jayensee in 2024. 
    
    It uses gradient descent to 
    estimate both the average and variance of the performance exhibited by rated
    participants. The step size decreases with the number of games player by the 
    participant.

    Ratings are continuously adjusted such that the average active participant has 
    a J-Skill of 0 and J-Consistency of 1 (to avoid the ratings drifting over time).

    New players have their initial average performance set to the average of the rated 
    participants in their first match.

    :param init_mu: Initial performance average for new participants when none of the 
        groups in dynamic_start are present in the match.
    :type init_mu: float
    :param init_sigma: Initial performance standard deviation for new participants.
    :type init_sigma: float
    :param k_mu: Initial step size for updates to the performance average.
    :type k_mu: float
    :param k_lnvar: Initial step size for updates to the log of the performance variance.
    :type k_lnvar: float
    :param final_psi: Multiplier for the step size of a participant with infinite games.
    :type final_psi: float
    :param psi_decay: Rate at which the step size decays.
    :type psi_decay: float
    :param avg_decay: Rate at which the estimation of the average rating changes.
    :type avg_decay: float
    :param dynamic_start: List of groups to include when initializing ratings.
    :type dynamic_start: list[str]         
    :param draw_margin: Minimum performance difference required for a victory.
    :type draw_margin: float      
    """
    def __init__(self, draw_margin=1, init_mu=0, init_sigma=1, k_mu=0.1, k_lnvar=0.1, 
                 final_psi=0.5, psi_decay=0.05, avg_decay=0.01, 
                 dynamic_start = [], **_):
        RatingSystem.__init__(self, rating_class=JRS_Rating, init_mu=init_mu, init_sigma=init_sigma,
                              k_mu=k_mu, k_lnvar=k_lnvar, final_psi=final_psi, 
                              psi_decay=psi_decay, dynamic_start=dynamic_start)
        self.draw_margin = draw_margin
        self.avg_mu = init_mu
        self.avg_lnvar = 2*np.log(init_sigma)
        self.avg_decay = avg_decay
        self.add_metric(InformationPreempted("InformationPreempted", draw_margin=draw_margin))

    def add_game(self, game):  
        _, pred_matrix, results_matrix, h2h_ratings = self._add_game_metrics_update(game)
        outcome_inds = np.argmax(results_matrix, axis=2)
        h2h_pre_update = np.empty((len(results_matrix),len(results_matrix),2))

        dynamic_start_sum = {}
        dynamic_start_count = {}
        game_sum_mu = 0
        game_sum_lnvar = 0
        game_rating_count = 0

        for team1_ind, h2h_row in enumerate(h2h_ratings):
            for team2_ind, team1_ratings in enumerate(h2h_row):
                if team2_ind == team1_ind:
                    continue
                team2_ratings = h2h_ratings[team2_ind][team1_ind]
                for rating in team1_ratings:
                    if rating.new:
                        continue
                    if rating.group.name not in dynamic_start_sum:
                        dynamic_start_sum[rating.group.name] = 0
                        dynamic_start_count[rating.group.name] = 0
                    dynamic_start_sum[rating.group.name] += rating.mu * rating.recent_partials[team1_ind]
                    dynamic_start_count[rating.group.name] += rating.recent_partials[team1_ind]
                for rating in team2_ratings:
                    if rating.new:
                        continue
                    if rating.group.name not in dynamic_start_sum:
                        dynamic_start_sum[rating.group.name] = 0
                        dynamic_start_count[rating.group.name] = 0
                    dynamic_start_sum[rating.group.name] += rating.mu * rating.recent_partials[team2_ind]
                    dynamic_start_count[rating.group.name] += rating.recent_partials[team2_ind]
                    
                game_sum_mu += sum(rating.mu*rating.recent_partials[team1_ind] for rating in team1_ratings) + sum(rating.mu*rating.recent_partials[team2_ind] for rating in team2_ratings)
                game_sum_lnvar += sum(rating.lnvar*rating.recent_partials[team1_ind] for rating in team1_ratings) + sum(rating.lnvar*rating.recent_partials[team2_ind] for rating in team2_ratings)
                game_rating_count += len(team1_ratings) + len(team2_ratings)
                
                h2h_mu = sum(rating.mu * rating.recent_partials[team1_ind] for rating in team1_ratings) 
                h2h_mu -= sum(rating.mu * rating.recent_partials[team2_ind] for rating in team2_ratings)
                h2h_var = sum((rating.var) * rating.recent_partials[team1_ind] for rating in team1_ratings) 
                h2h_var += sum((rating.var) * rating.recent_partials[team2_ind] for rating in team2_ratings)
                # h2h_var = sum((rating.psi**2+rating.var) * rating.recent_partials[team1_ind]**2 for rating in team1_ratings) 
                # h2h_var += sum((rating.psi**2+rating.var) * rating.recent_partials[team2_ind]**2 for rating in team2_ratings)
                h2h_sigma = h2h_var ** 0.5
                outcome = outcome_inds[team1_ind, team2_ind]
                if outcome == 0:
                    z = (self.draw_margin - h2h_mu) / (SQRT_2 * h2h_sigma)
                    dx = np.exp(-z**2) / (SQRT_2PI * h2h_sigma)
                    dy = dx * (self.draw_margin - h2h_mu) / (2 * h2h_var)
                elif outcome == 2:
                    z = (self.draw_margin + h2h_mu) / (SQRT_2 * h2h_sigma)
                    dx = - np.exp(-z**2) / (SQRT_2PI * h2h_sigma)
                    dy = - dx * (self.draw_margin + h2h_mu) / (2 * h2h_var)
                else:
                    z_w = (self.draw_margin - h2h_mu) / (SQRT_2 * h2h_sigma)
                    z_l = (self.draw_margin + h2h_mu) / (SQRT_2 * h2h_sigma)
                    dwx = np.exp(-z_w**2) / (SQRT_2PI * h2h_sigma)
                    dlx = -np.exp(-z_l**2) / (SQRT_2PI * h2h_sigma)
                    dwy = dwx * (self.draw_margin - h2h_mu) / (2 * h2h_var)
                    dly = - dlx * (self.draw_margin + h2h_mu) / (2 * h2h_var)
                    dx = - dlx - dwx
                    dy = - dwy - dly
                grad_log = [dx,dy] / pred_matrix[team1_ind, team2_ind, outcome]
                h2h_pre_update[team1_ind, team2_ind] = grad_log
                # h2h_pre_update[team1_ind, team2_ind,:2] = grad_log / h2h_sigma
                # h2h_pre_update[team1_ind, team2_ind,1] /= h2h_var

        self.avg_mu *= 1 - self.avg_decay
        self.avg_mu += self.avg_decay * game_sum_mu / game_rating_count
        self.avg_lnvar *= 1 - self.avg_decay
        self.avg_lnvar += self.avg_decay * game_sum_lnvar / game_rating_count

        self._update_participants(h2h_ratings, h2h_pre_update=h2h_pre_update, 
                                  dynamic_start_sum=dynamic_start_sum, dynamic_start_count=dynamic_start_count)

    def _result_probabilities(self, team1_ratings, team2_ratings, team1_ind, team2_ind):
        total_mu = 0
        total_var = 0
        for rating in team1_ratings:
            total_mu += rating.mu * rating.recent_partials[team1_ind]
            # total_var += (rating.var + rating.psi**2) * rating.recent_partials[team1_ind]**2
            total_var += (rating.var) * rating.recent_partials[team1_ind]
        for rating in team2_ratings:
            total_mu -= rating.mu * rating.recent_partials[team2_ind]
            # total_var += (rating.var + rating.psi**2) * rating.recent_partials[team2_ind]**2
            total_var += (rating.var) * rating.recent_partials[team2_ind]
        total_var *= 2 
        z_win = (self.draw_margin-total_mu) / (total_var**0.5)
        win_prob = (1 - special.erf(z_win)) / 2
        if self.draw_margin>0:
            z_loss = (self.draw_margin+total_mu) / (total_var**0.5)
            loss_prob = (1 - special.erf(z_loss)) / 2
        else:
            loss_prob = 1 - win_prob
        return win_prob, 1-win_prob-loss_prob, loss_prob

    @property
    def adjusted_draw_margin(self):
        return self.draw_margin / np.exp(self.avg_lnvar/2)

class JRS_Rating(Rating):
    def __init__(self, init_mu, init_sigma, k_mu, k_lnvar, dynamic_start,
                 final_psi, psi_decay):
        self.rating_components = ["J_Skill", "J_Consistency","J_Uncertainty", "J_XE"]
        self.mu = init_mu
        self.sigma = init_sigma
        self.k_mu = k_mu
        self.k_lnvar = k_lnvar

        self.delta_psi = 1 - final_psi
        self.final_psi = final_psi
        self.psi_decay = psi_decay

        self.dynamic_start = dynamic_start

        self.new = True

    def _pre_update(self, h2h_ratings, team_ind, 
                    dynamic_start_sum, dynamic_start_count, **context):
        if self.new:
            mu_sum = 0
            mu_count = 0
            for group_name in self.dynamic_start:
                if group_name in dynamic_start_sum:
                    mu_sum += dynamic_start_sum[group_name]
                    mu_count += dynamic_start_count[group_name]
            self.mu = mu_sum / mu_count if mu_count>0 else self.mu
            self.new = False
        self.old_var = self.var
        self.old_psi = self.psi
        self.old_delta_psi = self.delta_psi

    def _update(self, h2h_ratings, team_ind, h2h_pre_update, **context):
        update = [0,0]
        for team2_ind, team1_ratings in enumerate(h2h_ratings[team_ind]):
            if team2_ind == team_ind or self not in team1_ratings:
                continue
            update += h2h_pre_update[team_ind,team2_ind]

        self.mu += self.k_mu * (self.old_psi) * self.recent_partials[team_ind] * update[0]
        self.lnvar += self.k_lnvar * (self.old_psi) * (self.old_var) * self.recent_partials[team_ind] * update[1]
    
        self.delta_psi = self.old_delta_psi * (1 - self.psi_decay)
        # self.delta_psi = ((self.psi**2)*(1 - self.psi**2 * update[0]**2))**0.5


    def __str__(self):
        # return f"mu: {self.mu:.4f}\tlnvar: {self.lnvar:.4f}\tpsi: {self.psi:.4f}"
        return f"J-Skill: {self.J_Skill:.2f}\tJ-Consistency: {self.J_Consistency:.2f}\tJ-Uncertainty: {self.J_Uncertainty:.2f}\tJ-XE: {100*self.J_XE:.1f}%"
    
    @property
    def psi(self):
        """Current step size multiplier."""
        return self.final_psi + self.delta_psi

    @property
    def lnvar(self):
        """Natural log of the performance variance."""
        return np.log(self.var)
    @lnvar.setter
    def lnvar(self, value):
        self.var = min(max(np.exp(value),MIN_VAR),MAX_VAR)

    @property
    def sigma(self):
        """Performance standard deviation."""
        return self.var**0.5
    @sigma.setter
    def sigma(self, value):
        self.var = min(max(value**2,MIN_VAR),MAX_VAR)
    
    @property
    def J_Skill(self):
        """Average performance adjusted for rating drifts."""
        return (self.mu-self.system.avg_mu)/np.exp(self.system.avg_lnvar/2)
        # return self.mu
    @property
    def J_Consistency(self):
        """Base 2 log of the square root of performance precision adjusted for rating drifts."""
        return np.log2((self.var/np.exp(self.system.avg_lnvar))**-0.5)
        # return self.psi
    @property
    def J_Uncertainty(self):
        """Same as psi."""
        return self.psi
    @property
    def J_XE(self):
        """Probability of outperforming the average active participant."""
        return (1 - special.erf(- self.J_Skill * (2**self.J_Consistency) / SQRT_2)) / 2

        

        



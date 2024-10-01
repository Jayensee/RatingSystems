"""
The Metrics module contains all of the metric classes that can be used with the rating systems.

Every metric class has an error attribute that can be used as the loss for minimization.

Note: every rating system is assumed to predict draw probabilities.
"""

import numpy as np
from math import log2, log

class GMeanLikelihood:
    """
    The geometric mean of the likelihood.
    
    Can be thought of as the likelihood for each match.
    """
    def __init__(self, name="GMeanLikelihood"):
        self.name = name
        self.count = 0
        self.log_likelihood = 0
        self.likelihood = 0
        self.error = 1


    def _update(self, results_matrix, pred_matrix):
        self.log_likelihood *= self.count
        n = len(results_matrix)*(len(results_matrix)-1)
        self.count += n
        total_log_likelihood = np.sum(np.log(pred_matrix, where=results_matrix==1, out=np.zeros_like(pred_matrix)))
        self.log_likelihood += (total_log_likelihood)
        self.log_likelihood /= self.count
        self.likelihood = np.exp(self.log_likelihood)
        self.error = 1 - self.likelihood
        
    def __str__(self):
        if self.count > 0:
            return f"{self.name}: {self.likelihood:.6f}"
        else:
            return f"{self.name}: No games yet"

class PercentInformationPreempted:
    """
    Since every game is split into a series of head to head matches, there are 3 
    possible outcomes for each of these: win, draw, or loss (two if draws aren't possible).

    This means that each outcome can carry up to 1 trit/bit of information. 

    The information preempted by the system is divided by the maximum theoretical 
    information (1 trit/bit) and multiplied by 100 to get the percentage of this 
    maximum that the system preempted.
    """
    def __init__(self, name="PercentInformationPreempted", draw_margin=0):
        GMeanLikelihood.__init__(self, name)
        self.log_base = 3 if draw_margin>0 else 2
        self.percent_info = 0
        self.error = 100 - self.percent_info

    def _update(self, results_matrix, pred_matrix):
        GMeanLikelihood._update(self, results_matrix=results_matrix, pred_matrix=pred_matrix)
        try:
            self.percent_info = 100 * (1+log(self.likelihood, self.log_base))
            if self.percent_info > 98:
                print(self.likelihood)
        except ValueError:
            self.percent_info = 100 * self.likelihood
        self.error = 100 - self.percent_info
    
    def __str__(self):
        if self.count > 0:
            return f"{self.name}: {self.percent_info:.6f}"
        else:
            return f"{self.name}: No games yet"
        
class InformationPreempted:
    """
    The information content not preempted by the system for each head to head match 
    is given by the average negative log likelihood of its predictions.

    By subtracting this amount from the theoretical maximum information limit of 1 
    trit/bit we get the average information the system already had before knowing the 
    outcome of each match.

    The unit used is the bit.
    """
    def __init__(self, name="InformationPreempted", draw_margin=0):
        GMeanLikelihood.__init__(self, name)
        self.max_info = log2(3) if draw_margin>0 else 1
        self.preempted_bits = 0
        self.error = -self.preempted_bits

    def _update(self, results_matrix, pred_matrix):
        GMeanLikelihood._update(self, results_matrix=results_matrix, pred_matrix=pred_matrix)
        try:
            self.preempted_bits = (self.max_info+log2(self.likelihood))
        except ValueError:
            self.preempted_bits = 0.0
        self.error = -self.preempted_bits
    
    def __str__(self):
        if self.count > 0:
            return f"{self.name}: {self.preempted_bits:.6f}"
        else:
            return f"{self.name}: No games yet"

class GMeanLikelihood_Log_Odds(GMeanLikelihood):
    """
    Log-odds of the geometric mean of the likelihood.

    Transforms the geometric mean of the likelihood to have no upper or lower bound.
    """
    def __init__(self, name="GMeanLikelihood_Log_Odds"):
        GMeanLikelihood.__init__(self, name)
        self.log_odds = -np.inf
        
    def _update(self, results_matrix, pred_matrix):
        GMeanLikelihood._update(self, results_matrix=results_matrix, pred_matrix=pred_matrix)
        self.log_odds = 10*np.log2(self.likelihood/(1-self.likelihood))

    def __str__(self):
        if self.count > 0:
            return f"{self.name}: {self.log_odds:.6f}"
        else:
            return f"{self.name}: No games yet"

class RMS_Score_Error:
    """
    Root mean square error of the score.

    The score is defined as 1 for a win, 0.5 for a draw, 0 for a loss.
    """
    def __init__(self, name="RMS_Score_Error"):
        self.name = name
        self.count = 0
        self.error = 0
        self.square_error = 0

    def _matrix_to_scores(self, matrix):
        scores = matrix@[1,0.5,0]
        np.fill_diagonal(scores, 0)
        # scores = np.sum(scores, axis=1) / (len(matrix)-1)
        return scores

    def _update(self, results_matrix, pred_matrix):
        self.square_error *= self.count
        self.count += len(results_matrix) * (len(results_matrix)-1)
        results_scores = self._matrix_to_scores(results_matrix)
        pred_scores = self._matrix_to_scores(pred_matrix)
        self.square_error += np.sum((results_scores - pred_scores)**2)
        # if self.count>22:
        #     print("SE",self.square_error)
        #     print("PREDSCORES",pred_scores)
        #     print("RESSCORES",results_scores)
        #     print("NSE",(results_scores - pred_scores)**2)
        # if self.count>23:
        #     print("SE",self.square_error)
        #     print("PREDSCORES",pred_scores)
        #     print("RESSCORES",results_scores)
        #     print("NSE",(results_scores - pred_scores)**2)
        #     1/0
            
        self.square_error /= self.count
        self.error = self.square_error**0.5

    def __str__(self):
        if self.count > 0:
            return f"{self.name}: {self.error:.6f}"
        else:
            return f"{self.name}: No games yet"


import numpy as np
import pandas as pd
from tqdm import tqdm

class RatingSystem:
    """
    This is the base class every rating system inherits from.
    It includes all the core methods and attributes that are used by most 
    (if not all) systems.

    It's not an actual rating system in itself, it only exists to be inherited.

    :param rating_class: The rating class used by the system. Every system 
        implemented is expected to have its own rating class.
    :type rating_class: Rating
    :param default_params: The default parameters for new ratings. These get 
        passed onto the rating objects when they're created, so make sure the rating 
        class accepts them. 
    :type default_params: dict
    """

    def __init__(self, rating_class, **default_params):
        for param in default_params:
            setattr(self, param, default_params[param])
        self.default_params = default_params
        self.groups = {}
        self.interaction_groups = {}
        self.prediction_metrics = {}
        self.tracked_participants = {}
        self.games = 0
        self.rating_class = rating_class

    def add_metric(self, metric):
        """
        Adds a metric to the rating system.

        :param metric: Metric object (see Metrics module).
        :type metric: Metric
        """
        setattr(self, metric.name, metric)
        self.prediction_metrics[metric.name] = metric

    def group_to_clipboard(self, group_name):
        """
        Copies the ratings of the specified group to the clipboard, formatted as 
        a tsv (to paste into a spreadsheet).

        :param group_name: Name of the rating group.
        :type group_name: str
        """
        if group_name in self.groups:
            self.groups[group_name].to_clipboard()
        
    def interaction_to_clipboard(self, inter_name):
        """
        Copies the ratings of the specified interaction group to the clipboard, 
        formatted as a tsv (to paste into a spreadsheet).

        :param inter_name: Name of the interaction group.
        :type inter_name: str
        """
        if inter_name in self.interaction_groups:
            self.interaction_groups[inter_name].to_clipboard()

    def set_group(self, name, **override_params):
        """
        Creates a new rating group and sets its rating params. If the group 
        already exists it just sets the rating params.

        :param group_name: Name of the rating group.
        :type group_name: str
        :param override_params: The rating parameters of the group.
        :type group_name: dict
        """
        group_params = self.default_params.copy()
        for param in override_params:
            group_params[param] = override_params[param]
        self.groups[name] = ParticipantGroup(system=self, name=name, 
                                             rating_class=self.rating_class, **group_params)

    def _set_groups(self, game, **group_params):
        """
        Sets the rating params for every group in a game and creates ratings for 
        new players in said game.
        
        :param game: Dictionary with the data of a single game (using the standard 
            format).
        :type game: dict
        :param group_params: The rating parameters of the group.
        :type group_name: dict
        """
        for team in game["Teams"]:
            for group_name in team:
                if group_name not in self.groups:
                    self.set_group(group_name, **group_params)
                self.groups[group_name]._create_participants(game)

    def set_interaction(self, group1_name, group2_name, same_team, **override_params):
        """
        Creates a new interaction rating group of the given groups and sets its 
        rating params. If the group already exists it just sets the rating params.

        :param group1_name: The name of the first group.
        :type group1_name: str 
        :param group2_name: The name of the second group.
        :type group2_name: str 
        :param same_team: True if the interaction consists of ally duos or 
            matchups with opponents.
        :type same_team: bool 
        :param override_params: The rating parameters of the group.
        """
        interaction_group_params = self.default_params.copy()
        for param in override_params:
            interaction_group_params[param] = override_params[param]
        inter = InteractionGroup(system=self, 
                            group1_name=group1_name, group2_name=group2_name, same_team=same_team,
                            rating_class=self.rating_class, **interaction_group_params)
        self.interaction_groups[inter.name] = inter
    
    def _set_interactions(self, game, **inter_params):
        for inter_group_name, inter_group_info in game["Interactions"].items():
            if inter_group_name not in self.interaction_groups:
                self.set_interaction(group1_name=inter_group_info["Group1"],
                                     group2_name=inter_group_info["Group2"], 
                                     same_team=inter_group_info["Duo"], **inter_params)
            self.interaction_groups[inter_group_name]._create_participants(game)

    def _standardize_game_format(self, game):
        """
        WARNING: This is almost entirely untested, use at your own risk!

        I wrote this method in a sleep-deprived stupor before realizing the 
        formats of all my datasets were already standardized.

        Creates a dictionary in the standard format with the game data provided.

        Accepts the following formats:
        - {"Teams":[{"group1name":{"p1name":weight, "p2name":weight}, "group2name":{"p3name":weight, "p4name":weight}}, ...], "Results": []}
            This is the only format tested. More keys can be included if needed (ie. "Date").
        - {"Teams":[{"group1name":["p1name", "p2name"], "group2name":["p1name", "p2name"]}, ...], "Results": []}
            No weights.
        - {"Teams":[{"p1name":weight, "p2name":weight}, ...], "Results": []}
            No rating groups.
        - {"Teams":[["p1name", "p2name"], ...], "Results": []}
            No rating groups, no weights.
        - {"team1":score, "team2": score} 
            Single-player teams, no rating groups, no weights.
        
        :param game: Dictionary with the game data.
        :type game: dict 
        """
        # Formats allowed (most of this is untested):
        # {"team1":score, "team2": score} not multi-player teams, no groups, bare
        # {"Teams":[["p1name", "p2name"], ...], "Results": []}
        # {"Teams":[{"p1name":weight, "p2name":weight}, ...], "Results": []}
        # {"Teams":[{"group1name":["p1name", "p2name"], "group2name":["p1name", "p2name"]}, ...], "Results": []}
        # {"Teams":[{"group1name":{"p1name":weight, "p2name":weight}, "group2name":{"p3name":weight, "p4name":weight}}, ...], "Results": []}

        # Bare
        if "Teams" not in game:
            formatted_game = {"Teams":[{"default_group":{team:1}} for team in game], 
                              "Results":[result for result in game.values()]}
        # No groups no weights
        elif not isinstance(game["Teams"][0], dict):
            formatted_game = {"Teams":[{"default_group":[{player:1} for player in team]} for team in game["Teams"]], 
                              "Results":game["Results"]}
            for i in game:
                if i not in formatted_game:
                    formatted_game[i] = game[i]
        # No groups
        elif not isinstance(next(iter(game["Teams"][0].values())), dict):
            formatted_game = {"Teams":[{"default_group":[{player:team[player]} for player in team]} for team in game["Teams"]], 
                              "Results":game["Results"]}
            for i in game:
                if i not in formatted_game:
                    formatted_game[i] = game[i]
        # Other two formats
        else:
            formatted_game = game
            # Beyond cursed, I should have gone to sleep hours ago
            formatted_game["Teams"] = [{group:{player:team[group][player] if isinstance(team[group], dict) else 1 
                                               for player in team[group]} for group in team} for team in formatted_game["Teams"]]

        return formatted_game 

    def _add_game_metrics_update(self, game):
        """
        Executes all of the core steps of adding a game:

        1. Standardizes the game format
        2. Finds all of the interactions that are present in the game.
        3. Ensures every group and participant exists.
        4. Creates the lists of ratings for head to head matchups.
        5. Calculates the prediction matrix.
        6. Calculates the results matrix.
        7. Updates all of the metrics in the system.

        :param game: Dictionary with the data of a single game (using the standard 
            format).
        :type game: dict
        :return: 
            - game_full - Game data in standard format with the interactions added in.
            - pred_matrix - NumPy matrix with head to head predictions.
            - results_matrix - NumPy matrix with head to head results.
            - h2h_ratings - Nested lists (i, j) with the ratings of team i vs team j.
                (this is needed because of the interaction groups)
        """
        self.games += 1
        game = self._standardize_game_format(game)
        game_full = self._fill_interactions(game)
        self._set_groups(game_full, **self.default_params)
        self._set_interactions(game_full, **self.default_params)
        h2h_ratings = self._get_h2h_ratings(game_full=game_full)
        pred_matrix = self.calc_prediction_matrix(game_full, h2h_ratings)
        results_matrix = self._calc_results_matrix(game_full)
        self._update_metrics(results_matrix=results_matrix, pred_matrix=pred_matrix)
        return game_full, pred_matrix, results_matrix, h2h_ratings
    
    def add_game(self, game):
        """
        All of the systems implemented attempt to standardize the game dict provided, 
        but to be safe it's recommended to follow the standard format:
        ::

            {
            "Teams":[{
                "group1name":{
                    "p1name":weight, 
                    "p2name":weight
                }, 
                "group2name":{
                    "p3name":weight, 
                    "p4name":weight
                }
                }, 
                {
                "group1name":{
                    "p5name":weight, 
                    "p6name":weight
                }, 
                "group2name":{
                    "p7name":weight, 
                    "p9name":weight
                }
            }, 
            ...], 
            "Results": [1,0,...],
            }

        More keys are allowed, although as of now the only system using any beyond 
        these ones are Glicko and Glicko2.

        :param game: Dictionary with the data of a single game.
        :type game: dict
        """
        pass

    def add_games(self, games, verbose=False):
        """
        Calls the add_game method for every game in a list.
        
        :param games: List of dictionaries with game data.
        :type games: list[dict]
        """
        progress_bar = lambda x: tqdm(x) if verbose else x
        for game in progress_bar(games):
            self.add_game(game)

    def _update_metrics(self, results_matrix, pred_matrix):
        """
        Calls the _update method of every Metric in the system.

        :param results_matrix: NumPy matrix with head to head results.
        :type results_matrix: numpy.ndarray
        :param pred_matrix: NumPy matrix with head to head predictions.
        :type pred_matrix: numpy.ndarray
        """
        for metric in self.prediction_metrics.values():
            metric._update(results_matrix, pred_matrix)

    def _result_probabilities(self, team1_ratings, team2_ratings, team1_ind, team2_ind):
        """
        Empty method that needs to be implemented by the inheriting rating system.

        Calculates the win, draw, loss probabilities given the ratings of two teams. 
        The indices are used to track the weights and are needed to allow for 
        repeated participants on different teams in the same game.

        :param team1_ratings: List with ratings on team 1.
        :type team1_ratings: list[Rating]
        :param team2_ratings: List with ratings on team 2.
        :type team2_ratings: list[Rating]
        :param team1_ind: Indice of team 1 within the game.
        :type team1_ind: int
        :param team2_ind: Indice of team 2 within the game.
        :type team2_ind: int
        """
        pass

    def print_metrics(self, metric_names=None):
        """
        Prints all metrics in the system; alternatively a list of metric names 
        can be specified.

        :param metric_names: List with the names of metrics to print, if None 
            then every metric is printed.
        :type metric_names: list[str], Optional
        """
        if metric_names is not None:
            for metric_name in metric_names:
                print(self.prediction_metrics[metric_name])
        else:
            for metric in self.prediction_metrics.values():
                print(metric)

    def _calc_results_matrix(self, game):
        """
        Creates a matrix with head to head results from the team scores in a given game.

        :param game: Dictionary with the data of a single game.
        :type game: dict
        :return: NumPy matrix with the head to head results.
        :rtype: numpy.ndarray
        """
        res_matrix = np.empty((len(game["Teams"]),len(game["Teams"]),3))
        for team1_ind, _ in enumerate(game["Teams"]):
            for team2_ind, _ in enumerate(game["Teams"]):
                if team1_ind > team2_ind:
                    continue
                if team1_ind == team2_ind:
                    res_matrix[team1_ind,team2_ind] = [0,0,0]
                    continue
                res_matrix[team1_ind,team2_ind,0] = 1 if game["Results"][team1_ind] > game["Results"][team2_ind] else 0
                res_matrix[team1_ind,team2_ind,1] = 1 if game["Results"][team1_ind] == game["Results"][team2_ind] else 0
                res_matrix[team1_ind,team2_ind,2] = 1 if game["Results"][team1_ind] < game["Results"][team2_ind] else 0
                res_matrix[team2_ind,team1_ind,0] = 1 if game["Results"][team1_ind] < game["Results"][team2_ind] else 0
                res_matrix[team2_ind,team1_ind,1] = 1 if game["Results"][team1_ind] == game["Results"][team2_ind] else 0
                res_matrix[team2_ind,team1_ind,2] = 1 if game["Results"][team1_ind] > game["Results"][team2_ind] else 0
        return res_matrix
    
    def _update_participants(self, h2h_ratings, **context):
        """
        Runs the _pre_update and _update methods of every rating in a game.
        Clears the recent_partials of every rating after it is done (these are 
        the weights of the current match, separated by team index)

        :param h2h_ratings: Nested lists (i, j) with the ratings of team i vs team j.
        :type h2h_ratings: list[list[Rating]]
        :param context: Keyword arguments to pass to the _pre_update and _update methods
        :type context: dict
        """
        ratings_team_ind = {}
        for team1_ind, _ in enumerate(h2h_ratings):
            for team2_ind, _ in enumerate(h2h_ratings):
                if team1_ind == team2_ind:
                    continue
                for rating in h2h_ratings[team1_ind][team2_ind]:
                    if rating not in ratings_team_ind:
                        ratings_team_ind[rating] = {}
                    ratings_team_ind[rating][team1_ind] = None
        for rating, team_inds in ratings_team_ind.items():
            for team_ind in team_inds:
                rating._pre_update(h2h_ratings=h2h_ratings, team_ind=team_ind, **context)
        for rating, team_inds in ratings_team_ind.items():
            for team_ind in team_inds:
                rating._update(h2h_ratings=h2h_ratings, team_ind=team_ind, **context)
        for rating, team_inds in ratings_team_ind.items():
            for team_ind in team_inds:
                del rating.recent_partials[team_ind]
                
    def _fill_interactions(self, game):
        """
        Finds the interactions in a given game, and returns a copy of the game 
        data dictionary with the interactions added.

        :param game: Dictionary with the data of a single game.
        :type game: dict
        :return: Copy of the input dictionary but with the interactions added.
        :rtype: dict
        """
        game_full = game.copy()
        game_full["Interactions"] = {}
        game_full["Interactions_by_team"] = {}
        for team1_ind, team1 in enumerate(game["Teams"]):
            for team2_ind, team2 in enumerate(game["Teams"]):
                if team1_ind > team2_ind:
                    continue
                for group1_name in team1:
                    for group2_name in team2:
                        inter_name = f"{'Duo' if team1_ind == team2_ind else 'Op'}_{group1_name}_{group2_name}"
                        inter_name_inv = f"{'Duo' if team1_ind == team2_ind else 'Op'}_{group2_name}_{group1_name}"
                        if inter_name in self.interaction_groups:
                            if inter_name not in game_full["Interactions"]:
                                game_full["Interactions"][inter_name] = {"Group1": group1_name,
                                                                         "Group2": group2_name,
                                                                         "Duo": team1_ind == team2_ind,
                                                                         "Teams": []}
                            game_full["Interactions"][inter_name]["Teams"].append([team1_ind, team2_ind])
                            if team1_ind not in game_full["Interactions_by_team"]:
                                game_full["Interactions_by_team"][team1_ind] = {}
                            if team2_ind not in game_full["Interactions_by_team"][team1_ind]:
                                game_full["Interactions_by_team"][team1_ind][team2_ind] = set()
                            game_full["Interactions_by_team"][team1_ind][team2_ind].add(inter_name)
                        if inter_name_inv in self.interaction_groups:
                            if inter_name_inv not in game_full["Interactions"]:
                                game_full["Interactions"][inter_name_inv] = {"Group1": group2_name,
                                                                             "Group2": group1_name,
                                                                             "Duo": team1_ind == team2_ind,
                                                                             "Teams": []}
                            game_full["Interactions"][inter_name_inv]["Teams"].append([team2_ind, team1_ind])
                            if team2_ind not in game_full["Interactions_by_team"]:
                                game_full["Interactions_by_team"][team2_ind] = {}
                            if team1_ind not in game_full["Interactions_by_team"][team2_ind]:
                                game_full["Interactions_by_team"][team2_ind][team1_ind] = set()
                            game_full["Interactions_by_team"][team2_ind][team1_ind].add(inter_name_inv)
        return game_full
    
    def _get_h2h_ratings(self, game_full):
        """
        Creates a nested list of lists with the ratings of team i vs team j at (i, j).

        :param game_full: Dictionary with the data of a single game (including interactions).
        :type game_full: dict
        :return: Nested list of lists with the ratings of team i vs team j at (i, j).
        :rtype: list[list[Rating]]
        """
        h2h_ratings = [[[] for _ in game_full["Teams"]] for _ in game_full["Teams"]]
        for team1_ind,team1 in enumerate(game_full["Teams"]):
            for team2_ind,team2 in enumerate(game_full["Teams"]):
                if team1_ind == team2_ind:
                    continue
                for group_name, participant_name_dict in team1.items():
                    for participant_name, participant_partial in participant_name_dict.items():
                        rating = self.groups[group_name].participants[participant_name].rating
                        rating.recent_partials[team1_ind] = participant_partial
                        h2h_ratings[team1_ind][team2_ind].append(rating)

        for team1_ind,team1 in enumerate(game_full["Teams"]):
            for team2_ind,team2 in enumerate(game_full["Teams"]):
                if team1_ind == team2_ind:
                    continue
                if team1_ind in game_full["Interactions_by_team"]:
                    if team1_ind in game_full["Interactions_by_team"][team1_ind]:
                        interaction_names = game_full["Interactions_by_team"][team1_ind][team1_ind]
                        for interaction_name in interaction_names:
                            interaction = self.interaction_groups[interaction_name]
                            for participant1_ind, participant1_name in enumerate(team1[interaction.group1]):
                                for participant2_ind, participant2_name in enumerate(team1[interaction.group2]):
                                    if interaction.group1 == interaction.group2:
                                        if participant1_ind == participant2_ind or participant1_name > participant2_name:
                                            continue
                                    interaction_name = f"{participant1_name}_{participant2_name}"
                                    rating = interaction.participants[interaction_name].rating
                                    p1_rating = self.groups[interaction.group1].participants[participant1_name].rating
                                    p2_rating = self.groups[interaction.group2].participants[participant2_name].rating
                                    rating.recent_partials[team1_ind] = p1_rating.recent_partials[team1_ind] * p2_rating.recent_partials[team1_ind]
                                    h2h_ratings[team1_ind][team2_ind].append(rating)
                    if team2_ind in game_full["Interactions_by_team"][team1_ind]:
                        interaction_names = game_full["Interactions_by_team"][team1_ind][team2_ind]
                        for interaction_name in interaction_names:
                            interaction = self.interaction_groups[interaction_name]
                            for participant1_ind, participant1_name in enumerate(team1[interaction.group1]):
                                for participant2_ind, participant2_name in enumerate(team2[interaction.group2]):
                                    if interaction.group1 == interaction.group2:
                                        if participant1_name > participant2_name:
                                            continue
                                    interaction_name = f"{participant1_name}_{participant2_name}"
                                    rating = interaction.participants[interaction_name].rating
                                    p1_rating = self.groups[interaction.group1].participants[participant1_name].rating
                                    p2_rating = self.groups[interaction.group2].participants[participant2_name].rating
                                    rating.recent_partials[team1_ind] = p1_rating.recent_partials[team1_ind] * p2_rating.recent_partials[team2_ind]
                                    h2h_ratings[team1_ind][team2_ind].append(rating)
        return h2h_ratings


    def calc_prediction_matrix(self, game_full, h2h_ratings=None):
        """
        Calculates the predictions for a given game.

        Returns a square matrix with the [win, draw, loss] probabilities at of 
        team i vs team j at (i,j).

        :param game_full: Dictionary with the data of a single game (including interactions).
        :type game_full: dict
        :param h2h_ratings: Nested lists (i, j) with the ratings of team i vs team j.
        :type h2h_ratings: list[list[Rating]]
        :return: Prediction matrix.
        :rtype: numpy.ndarray
        """
        if "Interactions" not in game_full:
            game_full = self._fill_interactions(game_full)
        pred_matrix = np.empty((len(game_full["Teams"]),len(game_full["Teams"]),3))
        if h2h_ratings is None:
            h2h_ratings = self._get_h2h_ratings(game_full)
        for team1_ind,_ in enumerate(game_full["Teams"]):
            for team2_ind,_ in enumerate(game_full["Teams"]):
                if team1_ind == team2_ind:
                    pred_matrix[team1_ind,team2_ind] = [0,0,0]
                    continue
                team1_ratings = [*h2h_ratings[team1_ind][team2_ind]]
                team2_ratings = [*h2h_ratings[team2_ind][team1_ind]]
                probs = self._result_probabilities(team1_ratings=team1_ratings, team2_ratings=team2_ratings,
                                                   team1_ind=team1_ind, team2_ind=team2_ind)
                pred_matrix[team1_ind,team2_ind] = probs
        return pred_matrix
    
    def to_dataframe(self, include_groups=True, include_interaction_groups=True):
        """
        Creates a pandas DataFrame with participant ratings.

        :param include_groups: Whether to include the regular groups in the DataFrame.
        :type include_groups: bool
        :param include_interaction_groups: Whether to include the interaction groups in the DataFrame.
        :type include_interaction_groups: bool
        :return: pandas DataFrame with participant ratings.
        :rtype: pandas.DataFrame
        """
        dataframes = []
        sort_cols = []
        if include_groups:
            for _, group in self.groups.items():
                if len(group.participants) == 0:
                    continue
                df_og = group.to_dataframe()
                df = df_og.rename(columns={col:f"{group.name}_{col}" for col in df_og})
                sort_cols.extend([f"{group.name}_{col}" for col in df_og])
                dataframes.append(df)
        if include_interaction_groups:
            for _, inter_group in self.interaction_groups.items():
                if len(inter_group.participants) == 0:
                    continue
                df_og = inter_group.to_dataframe()
                df = df_og.rename(columns={col:f"{inter_group.name}_{col}" for col in df_og})
                sort_cols.extend([f"{group.name}_{col}" for col in df_og])
                dataframes.append(df)      
        full_df = pd.DataFrame()
        for df in dataframes:
            full_df = full_df.combine(df, lambda c1,c2:c2, fill_value="", overwrite=False)
        full_df = full_df[sort_cols]

        return full_df
    
    def to_csv(self, include_groups=True, include_interaction_groups=True, **kwargs):
        """
        Creates a csv string with participant ratings.

        :param include_groups: Whether to include the regular groups in the DataFrame.
        :type include_groups: bool
        :param include_interaction_groups: Whether to include the interaction groups in the DataFrame.
        :type include_interaction_groups: bool
        :param kwargs: Keyword arguments to pass to pandas.DataFrame.to_csv
        :type kwargs: dict
        :return: Csv string with participant ratings.
        :rtype: str
        """
        return self.to_dataframe(include_groups, include_interaction_groups).to_csv(**kwargs)

    def to_clipboard(self, include_groups=True, include_interaction_groups=True, excel=True, index=False, **kwargs):
        """
        Copies tables with participant ratings to the clipboard (to paste in a spreadsheet).

        :param include_groups: Whether to include the regular groups in the DataFrame.
        :type include_groups: bool
        :param include_interaction_groups: Whether to include the interaction groups in the DataFrame.
        :type include_interaction_groups: bool
        :param kwargs: Keyword arguments to pass to pandas.DataFrame.to_clipboard
        :type kwargs: dict
        :return: Csv string with participant ratings.
        :rtype: str
        """
        return self.to_dataframe(include_groups, include_interaction_groups).to_clipboard(excel=excel,index=index, **kwargs)    
    
    def print_participants(self):
        "Prints the ratings of all participants to stdout."
        print("GROUPS:")
        for group_name in self.groups:
            print(f"{group_name}: ")
            self.groups[group_name].print_participants(prefix="\t")
        print("\nINTERACTIONS:")
        for inter_name in self.interaction_groups:
            print(f"{inter_name}: ")
            self.interaction_groups[inter_name].print_participants(prefix="\t")
        

class ParticipantGroup:
    """
    This is used for regular participant groups.

    It shouldn't need to be extended, and ParticipantGroup objects should only 
    be created through the methods in the RatingSystem class.
    """
    def __init__(self, system, name, rating_class, **group_params):
        for param in group_params:
            setattr(self, param, group_params[param])
        self.params = group_params
        self.system = system
        self.name = name
        self.participants = {}
        self.rating_class = rating_class
    
    def create_participant(self, name):
        "Creates a participant with the given name."
        self.participants[name] = Participant(system=self.system, group=self, name=name, 
                                              rating_class=self.rating_class, **self.params)
        
    def _create_participants(self, game):
        for team in game["Teams"]:
            if self.name not in team: 
                continue
            for participant_name in team[self.name]:
                if participant_name not in self.participants:
                    self.create_participant(name=participant_name)

    def print_participants(self, prefix=""):
        "Prints the ratings of participants in the group to stdout."
        for participant in self.participants.values():
            print(f"{str(prefix)}{participant}")
    
    def to_dataframe(self):
        "Creates a pandas DataFrame with the ratings of participants in the group."
        dict_tocopy = {"Names":[]}
        for participant in self.participants.values():
            dict_tocopy["Names"].append(participant.name)
            for component in participant.rating.rating_components:
                if component not in dict_tocopy:
                    dict_tocopy[component] = []
                dict_tocopy[component].append(getattr(participant.rating, component))
        return pd.DataFrame(dict_tocopy)

    def to_csv(self, **kwargs):
        "Creates a csv string with the ratings of participants in the group."
        return self.to_dataframe().to_csv(**kwargs)

    def to_clipboard(self, excel=True, **kwargs):
        "Copies a table with the ratings of participants in the group to the clipboard."
        return self.to_dataframe().to_clipboard(excel=excel, **kwargs)

class InteractionGroup(ParticipantGroup):
    """
    This is used for groups consisting of interactions between participants.
    Mostly the same as ParticipantGroup.
    """
    def __init__(self, system, group1_name, group2_name, same_team, rating_class, **group_params):
        self.group1 = group1_name
        self.group2 = group2_name
        self.duo = same_team
        self.name = f"{'Duo' if same_team else 'Op'}_{group1_name}_{group2_name}"
        ParticipantGroup.__init__(self, system, self.name, rating_class, **group_params)

    def create_participant(self, name):
        self.participants[name] = Participant(system=self.system, group=self, name=name, 
                                              rating_class=self.rating_class, **self.params)

    def _create_participants(self, game):
        for team1_ind, team2_ind in game["Interactions"][self.name]["Teams"]:
            for participant1_ind, participant1_name in enumerate(game["Teams"][team1_ind][self.group1]):
                for participant2_ind, participant2_name in enumerate(game["Teams"][team2_ind][self.group2]):
                    if self.group1 == self.group2:
                        if (participant1_ind == participant2_ind and self.duo) or participant1_name > participant2_name:
                            continue
                    interaction_name = f"{participant1_name}_{participant2_name}"
                    if interaction_name not in self.participants:
                        self.create_participant(name=interaction_name)    

class Participant:
    """
    This is the class used for every participant (regular or interaction).

    It shouldn't need to be extended, and ParticipantGroup objects should only 
    be created through the methods in the RatingSystem class.
    """
    def __init__(self, system, group, name, rating_class, **params):
        for param in params:
            setattr(self, param, params[param])
        self.system = system
        self.group = group
        self.name = name
        self.rating = rating_class(**params)
        self.rating.system = system
        self.rating.group = group
        self.rating.participant = self
        self.rating.recent_partials = {}

    def __str__(self):
        return f"{self.name}: {self.rating}"
    
class Rating:
    """
    This is the base Rating class that needs to be extended for every rating system.
    
    Rating objects are created by the Participant class and have their parent 
    Participant, ParticipantGroup, and System objects automatically added as attributes.

    The recent_partials attribute is also automatically updated and keeps track of the 
    weights associated with the rating in the current game (it's a dictionary with 
    teamind: weight pairs for every team that the Participant is a part of).

    Ratings have a __hash__ method so they can be used as keys in dictionaries.

    :param rating_params: Keyword parameters used by the Rating.
    :type rating_params: dict
    """
    def __init__(self, **rating_params):
        pass

    def _pre_update(self, h2h_ratings, team_ind, **context):
        """
        _pre_update and _update take in the same arguments and are called in sequence.

        _update is guaranteed to always run after the _pre_update of every participant.
    
        :param h2h_ratings: Nested lists (i, j) with the ratings of team i vs team j.
        :type h2h_ratings: list[list[Rating]]
        :param team_ind: The index of the team the Participant is part of (if they're 
            part of several teams this method is called once for each)
        :type team_ind: int
        """
        pass

    def _update(self, h2h_ratings, team_ind, **context):
        """
        _pre_update and _update take in the same arguments and are called in sequence.

        _update is guaranteed to always run after the _pre_update of every participant.
    
        :param h2h_ratings: Nested lists (i, j) with the ratings of team i vs team j.
        :type h2h_ratings: list[list[Rating]]
        :param team_ind: The index of the team the Participant is part of (if they're 
            part of several teams this method is called once for each)
        :type team_ind: int
        """        
        pass

    def __hash__(self):
        return hash((self.group.name, self.participant.name))
    
    def __eq__(self, to_compare):
        try:
            return True if hash(self) == hash(to_compare) else False
        except TypeError:
            return False

    
    


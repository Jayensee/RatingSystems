# Rating Systems

In the 60s, the Elo rating system was invented with the goal of accurately rating chess players. 

Although it was a massive improvement on what came before it, Elo left much to be desired particularly for those seeking to use it outside of chess and so over the following decades other systems were created to compete with Elo.

This package implements 6 rating systems with the same code structure, allowing for direct comparison between them. 

Beyond that, it extends these systems to work for multi-player, multi-team games, and allows different parameters to be used for different groups of ratings.

It also implements interaction groups which are rating groups that assign ratings to pairs of participants, allowing for the rating of participant synergies and counters.

## Example usage

```
>>> from ratingsystems.Elo import Elo_System
>>> system = Elo_System(init_rating=1500, k=42)
>>> system.set_group("OtherPlayers", init_rating=100, k=10)
>>> system.set_interaction("OtherPlayers","OtherPlayers", same_team=False, init_rating=0)
>>> example_game = {
...     "Teams": [
...         {
...             "Players": {"p1":0.5, "p2":1},
...             "OtherPlayers": {"p3":1},
...         },
...         {
...             "Players": {"p4":1, "p5":0.5},
...             "OtherPlayers": {"p6":1},
...         }
...     ],
...     "Results": [2,1]
... }
>>> system.add_game(example_game)
>>> system.print_participants()
GROUPS:
OtherPlayers:
        p3: 105
        p6: 95
Players:
        p1: 1510
        p2: 1521
        p4: 1479
        p5: 1490

INTERACTIONS:
Op_OtherPlayers_OtherPlayers:
        p3_p6: 21
```

## [Documentation](docs/_build/html/index.html)

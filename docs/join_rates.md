# BDB 2027 -> nflverse join rates

_Generated 2026-10-15 by `scripts/verify_joins.py`._


## gameId -> nflverse schedules (`old_game_id`)

| season | seasonType | n | matched | rate |
|---|---|---|---|---|
| 2023 | POST | 13 | 13 | 1.000 |
| 2023 | PRE | 49 | 0 | 0.000 |
| 2023 | REG | 272 | 272 | 1.000 |
| 2024 | POST | 13 | 13 | 1.000 |
| 2024 | PRE | 49 | 0 | 0.000 |
| 2024 | REG | 272 | 272 | 1.000 |
| 2025 | POST | 13 | 13 | 1.000 |
| 2025 | PRE | 49 | 0 | 0.000 |
| 2025 | REG | 272 | 272 | 1.000 |


## (gameId, playId) of player_play -> pbp / participation / FTN (share of plays)

| season | seasonType | plays | pbp | participation | ftn |
|---|---|---|---|---|---|
| 2023 | POST | 1113 | 1.000 | 1.000 | 1.000 |
| 2023 | PRE | 5512 | 0.000 | 0.000 | 0.000 |
| 2023 | REG | 25095 | 1.000 | 1.000 | 1.000 |
| 2024 | POST | 1687 | 1.000 | 1.000 | 1.000 |
| 2024 | PRE | 6202 | 0.000 | 0.000 | 0.000 |
| 2024 | REG | 32616 | 1.000 | 1.000 | 1.000 |
| 2025 | POST | 1777 | 1.000 | 1.000 | 1.000 |
| 2025 | PRE | 6345 | 0.000 | 0.000 | 0.000 |
| 2025 | REG | 34607 | 1.000 | 1.000 | 1.000 |


## nflId -> nflverse players

- players.csv rows: 510; matched to nflverse `nfl_id`: 485 (95.1%)
- draft overall pick agrees (both drafted): 100.0% of 382
- draft year agrees (where nflverse has one): 100.0%
- display name identical: 89.5% (rest are suffix/nickname variants)
- with a `pfr_id`: 482; of those in nflverse combine: 403


Unmatched nflIds:

| nflId | displayName | draftYear | nflPosition |
|---|---|---|---|
| 56238 | Jadon Haselwood | 2023 | WR |
| 56518 | Alan Ali | 2023 | C |
| 56474 | Arquon Bush | 2023 | CB |
| 56228 | Habakkuk Baldonado | 2023 | DE |
| 56365 | D.J. Dale | 2023 | DT |
| 56183 | Emil Ekiyor | 2023 | G |
| 56662 | Rashad Torrence | 2023 | FS |
| 57428 | Brady Latham | 2024 | G |
| 57678 | Josh Proctor | 2024 | DB |
| 57581 | Keaton Bills | 2024 | G |
| 57874 | Trey Knox | 2024 | TE |
| 57637 | Myles Murphy | 2024 | DT |
| 57441 | Nelson Ceaser | 2024 | DE |
| 57577 | Keith Randolph | 2024 | DT |
| 57854 | Charles Turner | 2024 | C |
| 58445 | Kaden Prather | 2025 | WR |
| 58514 | Elijhah Badger | 2025 | WR |
| 58931 | Jah Joyner | 2025 | DE |
| 58497 | Zy Alexander | 2025 | DB |
| 58880 | O'Donnell Fortune | 2025 | DB |


## (gameId, playId, nflId) -> player listed on field in participation (plays present in participation only)

| season | seasonType | n | matched | rate |
|---|---|---|---|---|
| 2023 | POST | 1560 | 1560 | 1.000 |
| 2023 | REG | 38788 | 38788 | 1.000 |
| 2024 | POST | 4351 | 4351 | 1.000 |
| 2024 | REG | 85562 | 85562 | 1.000 |
| 2025 | POST | 7258 | 7258 | 1.000 |
| 2025 | REG | 124919 | 124865 | 1.000 |

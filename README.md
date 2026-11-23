# Efbet Predictor

Run `uv run python main.py` from the project directory. Choose a home and an away team to see the most likely 90-minute score, its probability, and the home-win/draw/away-win probabilities. Team names are case-insensitive. Enter `quit` at either prompt to exit.

The model estimates each team's scoring rate from its attack strength, its opponent's defence, and whether it plays home or away. It uses those rates to generate a Poisson score distribution. Exact scores are uncertain: even the most likely one can have a low probability. The model fits the local data at startup and does not use the old `model.pth` checkpoint.

## Experimental pre-match features

Try `uv run python main.py --prematch --date 2025-05-24`. Omit `--date` to use today. The fixture date determines which results can be used for training and for recent form; same-day results are excluded because kickoff times are incomplete.

For each team, this mode uses short rest (fewer than seven days since its last recorded match) and goals scored and conceded over its last five recorded matches within 60 days. Missing or stale history falls back to neutral. The data end in May 2025, so today's form and rest may be unavailable until newer results are added. The files do not include confirmed lineups or xG.

The pre-match variant scored worse on the current holdout, so the baseline remains the default.

## Match data and evaluation

`2022-23.json` adds 283 Bulgarian First League results (regular season, league playoff groups, and the European qualification playoff). Source: [Karel Stokkermans, RSSSF, *Bulgaria 2022/23*](https://www.rsssf.org/tablesb/bulg2023.html). Club names were normalized to the existing files. The neutral-site promotion/relegation playoff is excluded. Including the original 2023/24 and 2024/25 league and 2024/25 cup data, the model fits 890 completed matches.

Startup compares the original data, the expanded baseline, and the expanded model with pre-match features on the same 123 matches from February 26, 2025 onward. Earlier holdout results can inform later fixtures' form but are not used to fit the holdout models. The selected model is then refitted on results strictly before the requested prediction date. Lower score and outcome log loss are better. This is one historical split, not a guarantee of future performance. The next data priority is verified 2025/26 match-level results.

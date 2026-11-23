import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import date
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ORIGINAL_FILES = [ROOT / name for name in (
    "2023-24.json", "2024-25.json", "2024-25_bgcup.json",
)]
HISTORICAL_FILES = [ROOT / "2022-23.json"]


def load_match_data(files):
    matches = {}
    for file in files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        for match in data["matches"]:
            score = match.get("score")
            if isinstance(score, dict):
                score = score.get("ft")
            if (
                not isinstance(score, list)
                or len(score) != 2
                or not all(type(g) is int and g >= 0 for g in score)
            ):
                continue

            key = (match["date"], match["team1"].strip(), match["team2"].strip())
            goals = tuple(score)
            if key in matches and matches[key] != goals:
                raise ValueError(f"Conflicting full-time scores for {key}")
            matches[key] = goals

    return [(*key, goals) for key, goals in sorted(matches.items())]


class MatchHistory:
    def __init__(self, matches):
        self.by_team = defaultdict(list)
        for day, home, away, (home_goals, away_goals) in sorted(matches):
            day = date.fromisoformat(day)
            self.by_team[home].append((day, home_goals, away_goals))
            self.by_team[away].append((day, away_goals, home_goals))

    def team_features(self, team, day):
        results = self.by_team.get(team, ())
        end = bisect_left(results, (day,))
        recent = [result for result in results[max(0, end - 5):end]
                  if (day - result[0]).days <= 60]
        if not recent:
            return (0.0, 0.0, 0.0)

        days_rest = (day - recent[-1][0]).days
        short_rest = max(0, 7 - days_rest) / 7
        # Pad short histories with neutral form
        scored = sum(gf - 1.25 for _, gf, _ in recent) / 5
        conceded = sum(ga - 1.25 for _, _, ga in recent) / 5
        return (short_rest, scored, conceded)

    def fixture_features(self, fixtures):
        rows = []
        for day, home, away in fixtures:
            day = date.fromisoformat(day)
            home_form = self.team_features(home, day)
            away_form = self.team_features(away, day)
            rows.append((home_form + away_form, away_form + home_form))
        return torch.tensor(rows, dtype=torch.float32)


class ScoreModel(torch.nn.Module):
    def __init__(self, teams, use_prematch=False):
        super().__init__()
        # Index 0 is the league-average fallback for teams unseen in training.
        self.team_to_id = {team: i for i, team in enumerate(sorted(teams), start=1)}
        self.effects = torch.nn.Embedding(len(teams) + 1, 2, padding_idx=0)
        torch.nn.init.zeros_(self.effects.weight)
        self.log_base = torch.nn.Parameter(torch.zeros(2))
        self.prematch_weights = torch.nn.Parameter(torch.zeros(6)) if use_prematch else None

    def encode(self, pairs):
        return torch.tensor([[self.team_to_id.get(team, 0) for team in pair]
                             for pair in pairs], dtype=torch.long)

    def inputs(self, fixtures, history):
        teams = self.encode((home, away) for _, home, away in fixtures)
        if self.prematch_weights is None:
            return teams, None
        if history is None:
            raise ValueError("Pre-match features require match history")
        return teams, history.fixture_features(fixtures)

    def forward(self, features, prematch=None):
        home, away = self.effects(features).unbind(dim=1)
        log_rates = self.log_base + torch.stack(
            (home[:, 0] - away[:, 1], away[:, 0] - home[:, 1]), dim=1,
        )
        if self.prematch_weights is not None:
            log_rates = log_rates + prematch @ self.prematch_weights
        return log_rates


def train_model(matches, history=None):
    teams = {team for _, home, away, _ in matches for team in (home, away)}
    model = ScoreModel(teams, use_prematch=history is not None)
    inputs = model.inputs([match[:3] for match in matches], history)
    goals = torch.tensor([goals for _, _, _, goals in matches], dtype=torch.float32)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.poisson_nll_loss(model(*inputs), goals)
        # Shrink sparse teams
        loss = loss + 0.02 * model.effects.weight.square().mean()
        if model.prematch_weights is not None:
            loss = loss + 0.02 * model.prematch_weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model.eval()


def outcome(home, away):
    return 0 if home > away else 1 if home == away else 2


def score_forecast(home_rate, away_rate):
    def poisson_probs(rate):
        # Include the tail out to at least 12 standard deviations above
        limit = math.ceil(rate + 12 * math.sqrt(rate + 1))
        probs = [math.exp(-rate)]
        for goals in range(1, limit + 1):
            probs.append(probs[-1] * rate / goals)
        return probs

    home_probs, away_probs = poisson_probs(home_rate), poisson_probs(away_rate)
    probs = [0.0, 0.0, 0.0]
    for home, hp in enumerate(home_probs):
        for away, ap in enumerate(away_probs):
            probs[outcome(home, away)] += hp * ap

    total = sum(probs)
    score = (math.floor(home_rate), math.floor(away_rate))
    return score, home_probs[score[0]] * away_probs[score[1]], [p / total for p in probs]


@torch.no_grad()
def predict_match(model, home, away, match_day=None, history=None):
    match_day = match_day or date.today().isoformat()
    rates = model(*model.inputs([(match_day, home, away)], history)).exp()[0].tolist()
    return score_forecast(*rates)


@torch.no_grad()
def evaluate(model, matches, history=None):
    goals = torch.tensor([goals for _, _, _, goals in matches], dtype=torch.float32)
    log_rates = model(*model.inputs([match[:3] for match in matches], history))
    rates = log_rates.exp()
    log_loss = (rates - goals * log_rates + torch.lgamma(goals + 1)).sum(dim=1).mean().item()
    score_hits = outcome_hits = 0
    outcome_loss = 0.0
    for (home, away), rate in zip(goals.tolist(), rates.tolist()):
        score, _, probs = score_forecast(*rate)
        actual = outcome(home, away)
        score_hits += score == (home, away)
        outcome_hits += max(range(3), key=probs.__getitem__) == actual
        outcome_loss -= math.log(probs[actual])
    n = len(matches)
    return (f"score log loss: {log_loss:.3f} | exact score: {score_hits / n:.1%} | "
            f"outcome log loss: {outcome_loss / n:.3f} | outcome accuracy: {outcome_hits / n:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Predict Bulgarian football match scores.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="fixture date (YYYY-MM-DD, defaults to today)")
    parser.add_argument("--prematch", action="store_true",
                        help="enable experimental rest and recent-form features")
    args = parser.parse_args()
    match_day = args.date.isoformat()
    original = load_match_data(ORIGINAL_FILES)
    expanded = load_match_data(HISTORICAL_FILES + ORIGINAL_FILES)
    history = MatchHistory(expanded)
    available = [match for match in expanded if match[0] < match_day]
    if not available:
        parser.error("no match results before the requested date")
    # Use the same recent holdout for both fits and keep matchdays together
    cutoff = original[int(0.8 * len(original))][0]
    test = [match for match in original if match[0] >= cutoff]
    print(f"Chronological holdout: {len(test)} matches from {cutoff}")
    for name, matches in [("Original data", original), ("With 2022/23", expanded)]:
        train = [match for match in matches if match[0] < cutoff]
        print(f"{name} ({len(train)} training matches): {evaluate(train_model(train), test)}")

    train = [match for match in expanded if match[0] < cutoff]
    # Earlier holdout scores can provide form for later dates, but never fit
    print(f"With pre-match features ({len(train)} training matches): "
          f"{evaluate(train_model(train, history), test, history)}")

    prediction_history = history if args.prematch else None
    model = train_model(available, prediction_history)
    model_name = "pre-match features" if args.prematch else "baseline"
    print(f"Predicting fixtures on {match_day} with {model_name}, using {len(available)} "
          f"earlier results (latest: {available[-1][0]}).")
    teams_by_name = {team.casefold(): team for team in model.team_to_id}
    print("Available teams: " + ", ".join(model.team_to_id))
    print("Enter 'quit' at either prompt to exit.")

    while True:
        try:
            home_name = input("Home team: ").strip().casefold()
            if home_name == "quit":
                break
            home = teams_by_name.get(home_name)
            if home is None:
                print("Unknown home team. Choose a name from the list above.")
                continue

            away_name = input("Away team: ").strip().casefold()
            if away_name == "quit":
                break
            away = teams_by_name.get(away_name)
            if away is None:
                print("Unknown away team. Choose a name from the list above.")
                continue
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if home == away:
            print("Choose two different teams.")
            continue

        score, chance, (home_win, draw, away_win) = predict_match(
            model, home, away, match_day, prediction_history,
        )
        home_goals, away_goals = score
        print(f"{home} win: {home_win:.1%} | Draw: {draw:.1%} | {away} win: {away_win:.1%}")
        print(f"Most likely 90-minute score: {home} {home_goals}-{away_goals} {away} ({chance:.1%})")


if __name__ == "__main__":
    main()

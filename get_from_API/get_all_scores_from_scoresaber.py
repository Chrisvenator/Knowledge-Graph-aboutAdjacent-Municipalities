#!/usr/bin/env python3
"""
fetch_scores.py

Download every paginated score entry for a ScoreSaber player.

Example:
    python fetch_scores.py 76561198274713084 --limit 100 --outfile my_scores.json
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

import requests


def fetch_scores(player_id: str, limit: int = 100, delay: float = 0.01):
    """Return a list containing every score page for the given ScoreSaber player."""
    base_url = f"https://scoresaber.com/api/player/{player_id}/scores"
    headers = {"accept": "application/json"}
    page = 1
    scores_accum = []

    while True:
        params = {"limit": limit, "page": page, "withMetadata": "true"}
        resp = requests.get(base_url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        # The payload can be a list or a dict containing "playerScores".
        if isinstance(payload, dict):
            records = payload.get("playerScores", [])
        else:
            records = payload

        if not records:
            break

        scores_accum.extend(records)

        # Stop when fewer records arrive than requested or metadata signals completion.
        if len(records) < limit:
            break

        page += 1
        time.sleep(delay)
        # if page > 100:
        #     break

    return scores_accum


def main():
    parser = argparse.ArgumentParser(
        description="Download all ScoreSaber scores for a player."
    )
    parser.add_argument("--player_id", default="76561198274713084", help="SteamID64 or ScoreSaber player ID")
    parser.add_argument("--limit", type=int, default=10, help="Items per page (max 100)")
    parser.add_argument("--outfile", default=None, help="Destination JSON file")
    args = parser.parse_args()

    scores = fetch_scores(args.player_id, limit=args.limit)

    print(f"Retrieved {len(scores)} scores.")
    if args.outfile:
        path = pathlib.Path(args.outfile)
    else:
        path = pathlib.Path(__file__).parent.parent / "data" / f"scores_{args.player_id}.json"

    path.write_text(json.dumps(scores, indent=2))
    print(f"Scores written to {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

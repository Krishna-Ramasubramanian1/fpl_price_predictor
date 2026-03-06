import requests
import pandas as pd
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url, headers=headers).json()

players = pd.DataFrame(data['elements'])
teams = pd.DataFrame(data['teams'])

all_history = []

for player_id in players["id"]:

    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        data = requests.get(url, headers=headers).json()

        history = pd.DataFrame(data["history"])

        if history.empty:
            continue

        history["player_id"] = player_id

        all_history.append(history)

        print(f"Player {player_id} scraped")

        time.sleep(0.3)

    except:
        print(f"Error with player {player_id}")
        continue


dataset = pd.concat(all_history, ignore_index=True)

print(dataset.head())

dataset.to_csv("player_match_history.csv", index=False)
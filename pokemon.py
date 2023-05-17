# get first 151 pokemon from pokeapi.co
from requests import get
import json

def main():
    print("Starting")

def download_pokemon_to_file():
    pokemon = get("https://pokeapi.co/api/v2/pokemon?limit=151").json()
    # write pokemon data to json file
    with open("pokemon.json", "w") as f:
        json.dump(pokemon, f)

def enrich_pokemon_entries():
    # read pokemon data from json file
    with open("pokemon.json", "r") as f:
        pokemon = json.load(f)
    # enrich pokemon data with additional data from pokeapi.co
    for entry in pokemon["results"]:
        url = entry["url"]
        pokemon_data = get(url).json()
        entry["data"] = pokemon_data
    # write enriched pokemon data to json file
    with open("pokemon.json", "w") as f:
        json.dump(pokemon, f)
        
main()
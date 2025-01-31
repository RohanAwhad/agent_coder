import json
import os
from tavily import TavilyClient

def load_text(fn: str) -> str:
  with open(fn, "r") as f: return f.read()

def save_text(text: str, fn: str) -> None:
  with open(fn, "w") as f: f.write(text)

def append_text(text: str, fn: str) -> None:
  with open(fn, "a") as f: f.write(text)

def save_json(data: dict, fn: str) -> None:
  with open(fn, "w") as f: json.dump(data, f, indent=2)

def load_json(fn: str) -> dict:
  with open(fn, 'r') as f: return json.load(f)


def search_web(query: str) -> str:
  client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
  response = client.search(query)['results']
  return json.dumps(response, indent=2)
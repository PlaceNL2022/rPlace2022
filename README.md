# Reddit /r/place 2022 headless bot

This headless Python bot will automatically login to reddit, obtain access 
tokens (and refreshes them when they expire), obtain orders from the C&C server
and automatically place pixels at the desired locations.

## Requirements

- Python >= 3.8
- NumPy
- Matplotlib
- Rich
- aiohttp

## Installation

```bash
pip install https://github.com/PlaceNL/rPlace2022
```

## Usage

```bash
python PlaceNL -u "USERNAME" "PASSWORD"
```

The bot supports multiple users:
```bash
python PlaceNL -u "USERNAME1" "PASSWORD1" -u "USERNAME2" "PASSWORD2"
```

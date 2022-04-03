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

## Installation & updating to a new version

```bash
pip install --force https://github.com/PlaceNL/rPlace2022
```

## Usage

```bash
PlaceNL -u "USERNAME" "PASSWORD"
```

The bot supports multiple users:
```bash
PlaceNL -u "USERNAME1" "PASSWORD1" -u "USERNAME2" "PASSWORD2"
```

## Docker image

For people experienced with Docker, there's also a docker image you can run:

```bash
docker run -t ghcr.io/placenl/placenl-python -u "USERNAME" "PASSWORD"
```

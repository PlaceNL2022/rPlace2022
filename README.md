# Reddit /r/place 2022 headless bot

This headless Python bot will automatically login to reddit, obtain access
tokens (and refreshes them when they expire), obtain orders from the C&C server
and automatically place pixels at the desired locations.

## Requirements

- Python >= 3.8
- NumPy
- Pillow
- Rich
- aiohttp

## Installation

For now, just download the script and run it with Python. More easy installation methods will come soon.

## Usage

```bash
python PlaceNL.py -u "USERNAME" "PASSWORD"
```

The bot supports multiple users:
<<<<<<< HEAD
=======
```bash
PlaceNL -u "USERNAME1" "PASSWORD1" -u "USERNAME2" "PASSWORD2"
```

**IMPORTANT**: Do you have a $ in your password? Please escape it as follows:

```bash
PlaceNL -u "USERNAME" "PA\$\$WORD"
```

## Docker image

For people experienced with Docker, there's also a docker image you can run:
>>>>>>> b0af102e0571d43ee109201d6664e4cb05efbc7c

```bash
python PlaceNL.py -u "USERNAME1" "PASSWORD1" -u "USERNAME2" "PASSWORD2"
```

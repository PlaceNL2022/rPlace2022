"""
Headless reddit /r/place 2022 updater.

This Python bot is a terminal application that will automatically update
r/place pixels from our command and control server. It supports multiple
reddit accounts and automatically obtains and refreshed access tokens.

Authors:
- /u/tr4ce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations
import re
import json
import random
import argparse
import logging
import asyncio
import string
import warnings
from io import BytesIO
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import aiohttp
from PIL import Image
from rich.logging import RichHandler

__version__ = '2'

logger = logging.getLogger()
logging.basicConfig(format=r"[%(name)s] %(message)s", handlers=[RichHandler()])

MAP_WIDTH = 2000
MAP_HEIGHT = 2000

REDDIT_LOGIN_GET = (
    "https://www.reddit.com/login/?experiment_d2x_2020ify_buttons=enabled&"
    "experiment_d2x_sso_login_link=enabled&experiment_d2x_google_sso_gis_parity=enabled&"
    "experiment_d2x_onboarding=enabled")
REDDIT_LOGIN_POST = "https://www.reddit.com/login"
REDDIT_PLACE_URL = "https://www.reddit.com/r/place/"
REDDIT_PLACE_SET_PIXEL_URL = "https://gql-realtime-2.reddit.com/query"
PLACE_WEBSOCKET = "wss://gql-realtime-2.reddit.com/query"
BACKEND_DOMAIN = "placenl.noahvdaa.me"
CNC_WEBSOCKET = f"wss://{BACKEND_DOMAIN}/api/ws"
BACKEND_MAPS_URL = f"https://{BACKEND_DOMAIN}/maps"
DEFAULT_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:98.0) "
                      "Gecko/20100101 "
                      "Firefox/98.0")

GRAPHQL_CANVAS_QUERY = """
subscription replace($input: SubscribeInput!) {
    subscribe(input: $input) {
        id
        ... on BasicMessage {
            data {
                __typename
                ... on FullFrameMessageData {
                    __typename
                    name
                    timestamp
                }
            }
            __typename
        }
        __typename
    }
}""".strip()

SET_PIXEL_QUERY = """
mutation setPixel($input: ActInput!) {
    act(input: $input) {
        data {
            ... on BasicMessage {
                id
                data {
                    ... on GetUserCooldownResponseMessageData {
                        nextAvailablePixelTimestamp
                        __typename
                    }
                    ... on SetPixelResponseMessageData {
                        timestamp
                        __typename
                    }
                    __typename
                }
                __typename
            }
            __typename
        }
        __typename
    }
}""".strip()


COLOR_MAPPINGS = {
    '#6D001A': 0,
    '#BE0039': 1,
    '#FF4500': 2,
    '#FFA800': 3,
    '#FFD635': 4,
    '#FFF8B8': 5,
    '#00A368': 6,
    '#00CC78': 7,
    '#7EED56': 8,
    '#00756F': 9,
    '#009EAA': 10,
    '#00CCC0': 11,
    '#2450A4': 12,
    '#3690EA': 13,
    '#51E9F4': 14,
    '#493AC1': 15,
    '#6A5CFF': 16,
    '#94B3FF': 17,
    '#811E9F': 18,
    '#B44AC0': 19,
    '#E4ABFF': 20,
    '#DE107F': 21,
    '#FF3881': 22,
    '#FF99AA': 23,
    '#6D482F': 24,
    '#9C6926': 25,
    '#FFB470': 26,
    '#000000': 27,
    '#898D90': 29,
    '#515252': 28,
    '#D4D7D9': 30,
    '#FFFFFF': 31
}

access_token_regexp = re.compile(r'"accessToken":"([a-zA-Z0-9\-_]+)"')
expires_in_regexp = re.compile(r'"expiresIn":(\d+)')
csrf_regexp = re.compile(r'<input type="hidden" name="csrf_token" value="(\w+)">')


class CNCOrderClient:
    """Client for the CNC order server."""

    def __init__(self, session: aiohttp.client.ClientSession):
        """Initialize the client

        Args:
            session (aiohttp.client.ClientSession): The session to use for
        """
        self.session = session
        self.ws = None
        self.logger = logging.getLogger('PlaceNL.cnc')

    async def __aenter__(self) -> CNCOrderClient:
        """Context manager enter

        Returns:
            CNCOrderClient: The client
        """
        self.logger.info("Connecting to Command & Control server...")
        self.ws = await self.session.ws_connect(CNC_WEBSOCKET)
        self.logger.info("Success.")

        asyncio.get_running_loop().create_task(self.ping())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit

        Args:
            exc_type (_type_): _description_
            exc_val (_type_): _description_
            exc_tb (_type_): _description_
        """
        if self.ws:
            await self.ws.close()

        self.ws = None

    async def ping(self):
        """Ping the server every 5 seconds to keep the connection alive."""
        await asyncio.sleep(5)

        if not self.ws:
            return

        try:
            await self.ws.send_str(json.dumps({"type": "ping"}))
        except ConnectionResetError:
            self.logger.exception("Could not send ping, websocket closed?")
            self.ws = None

        asyncio.get_running_loop().create_task(self.ping())

    async def receive_orders(self, new_map_callback=None):
        """Receive orders from the server

        Args:
            new_map_callback (_type_, optional): _description_. Defaults to None.
        """
        if not self.ws:
            return

        await self.ws.send_str(json.dumps({"type": "getmap"}))
        await self.ws.send_str(json.dumps({"type": "brand",
                                           "brand": f"PlaceNLpythonV{__version__}"}))

        async for msg in self.ws:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            if data['type'] == 'map':
                map_url = f"{BACKEND_MAPS_URL}/{data['data']}"
                reason = data.get('reason')
                self.logger.info("Loading new map (reason: %s)", reason
                                 if reason else "connected to server")
                self.logger.info("Map URL: %s", map_url)

                order_map = await self.load_map(map_url)

                if new_map_callback:
                    new_map_callback(order_map)

    async def load_map(self, map_url: str) -> np.ndarray:
        """Load a map from the server.

        Args:
            map_url (str): The URL of the map.

        Returns:
            np.ndarray: np.ndarray of the map of shape(MAP_WIDTH, MAP_HEIGHT, 4) with the RGBA \
                values of the pixels
        """
        async with self.session.get(map_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                self.logger.warning(
                    "Loading the map failed! Got HTTP response %d. Error:\n%s", resp.status, text)
                return

            data = await resp.read()

            order_template = Image.new('RGBA', (MAP_WIDTH, MAP_HEIGHT), (0, 0, 0, 0))
            order_template.paste(Image.open(BytesIO(data)), (0, 0))
            order_map = np.array(order_template)
            self.logger.info("Downloaded orders map, image size: %s (dtype: %s)",
                             order_map.shape, order_map.dtype)

            return order_map

    async def update_pixels(self, queue: deque, event: asyncio.Event):
        """Update the pixels of the map.

        Args:
            queue (deque): update queue.
            event (asyncio.Event): __description__.
        """
        while True:
            await event.wait()
            self.logger.info("Got new pixel event, notifying C&C server...")

            if not self.ws:
                event.clear()
                return

            while queue:
                row, col, color = queue.popleft()
                await self.signal_pixel_drawn(row, col, color)

            event.clear()

    async def signal_pixel_drawn(self, row: int, col: int, color: int):
        """Signal a pixel has been drawn.

        Args:
            row (int): row of the pixel
            col (int): column of the pixel
            color (int): color of the pixel (0-31) see COLOR_MAPPINGS
        """
        if not self.ws:
            return

        await self.ws.send_str(json.dumps({"type": "placepixel",
                                           "x": col,
                                           "y": row,
                                           "color": color}))
        self.logger.info(
            "Notified CNC server of drawn pixel (%d, %d), color: %d.", col, row, color)


class RedditPlaceClient:
    """Client for the Reddit Place server."""

    def __init__(self, session: aiohttp.client.ClientSession,
                 username: str, password: str, user_agent=None, debug=False):
        """Initialize the client.

        Args:
            session (aiohttp.client.ClientSession): __description__
            username (str): Reddit username
            password (str): Reddit password
            user_agent (_type_, optional): _description_. Defaults to None.
            debug (bool, optional): _description_. Defaults to False.
        """
        self.session = session
        self.username = username
        self.password = password
        self.user_agent = user_agent if user_agent else DEFAULT_USER_AGENT

        self.access_token = ""
        self.access_token_expire = None
        self.current_canvas = None

        self.logger = logging.getLogger(f'PlaceNL.reddit.{username}')
        self.debug = debug

    async def __aenter__(self) -> RedditPlaceClient:
        """Context manager enter

        Raises:
            Exception: Reddit login was unsuccessful!
            Exception: Could not obtain access token.

        Returns:
            RedditPlaceClient: self
        """
        self.logger.info("Logging in reddit user %s...", self.username)
        success = await self.login()

        if not success:
            raise Exception("Reddit login was unsuccessful!")

        result = await self.scrape_access_token()

        if not result:
            raise Exception("Could not obtain access token.")

        self.access_token, expires_in = result
        expires = timedelta(seconds=expires_in / 1000)
        self.access_token_expire = datetime.now() + expires

        self.logger.info("Login successful, obtained access token: %s. Expires: %s (%d minutes)",
                         self.access_token, self.access_token_expire,
                         expires.total_seconds() // 60)

        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Context manager exit

        Args:
            exc_type (_type_): _description_
            exc (_type_): _description_
            tb (_type_): _description_
        """
        self.loop = None

    async def login(self) -> bool:
        """Login on reddit.com using the given username and password.

        Returns:
            bool: True if login was successful, False otherwise.
        """

        # First we have to obtain the form CSRF token
        headers = {
            "User-Agent": self.user_agent
        }

        async with self.session.get(REDDIT_LOGIN_GET, headers=headers) as resp:
            if resp.status != 200:
                self.logger.error(
                    "Could not login to reddit, failed to obtain CSRF token. HTTP status: %d.",
                    resp.status)
                return False

            html = await resp.text()
            matches = csrf_regexp.search(html)

            try:
                csrf_token = matches.groups(1)[0]
            except IndexError:
                self.logger.error("Could not login to reddit, failed to obtain CSRF token.")
                return False

        post_data = {
            'csrf_token': csrf_token,
            'username': self.username,
            'password': self.password,
            'dest': "https://www.reddit.com"
        }

        async with self.session.post(REDDIT_LOGIN_POST, data=post_data) as resp:
            if resp.status != 200:
                self.logger.error("Could not login to Reddit! HTTP status %d.", resp.status)
                return False

            cookies = self.session.cookie_jar.filter_cookies(
                "https://www.reddit.com")

            if 'reddit_session' not in cookies:
                self.logger.error("Login unsuccessful! Could not find reddit session cookie.")
                return False

        return True

    async def scrape_access_token(self) -> Optional[tuple[str, int]]:
        """Scrape a few required things from the Reddit Place page.

        We need the `modhash` key(reddit's CSRF protection key) for further
        requests. Furthermore, the Place page contains the websocket URL, which
        we need to obtain updates.

        Returns:
            Optional[tuple[str, int]]: (access_token, expires_in)
        """

        async with self.session.get(REDDIT_PLACE_URL) as resp:
            if resp.status != 200:
                logger.error("Error requesting access token, HTTP status: %d", resp.status)
                return

            data = await resp.text()

            if self.debug:
                fname = f"place_access_token_{self.username}.html"
                logger.debug("Writing HTML to %s", fname)
                with open(fname, "w") as o:
                    o.write(data)

            access_token_matches = access_token_regexp.search(data)
            expires_in_matches = expires_in_regexp.search(data)

            if not access_token_matches or not expires_in_matches:
                logger.error("Could not find access token in received HTML.")
                return

            try:
                access_token = access_token_matches.groups(1)[0]
                expires_in = int(expires_in_matches.groups(1)[0])
            except IndexError:
                self.logger.error("Could not find access token in received HTML.")
                return

            return access_token, expires_in

    async def refresh_access_token(self) -> bool:
        """Refresh the access token.

        Returns:
            bool: True if refresh was successful, False otherwise.
        """
        result = await self.scrape_access_token()
        if not result:
            self.logger.error("Could not refresh access token!")
            return False

        self.access_token, expires_in = result
        expires = timedelta(seconds=expires_in / 1000)
        self.access_token_expire = datetime.now() + expires

        self.logger.info("Refreshed access token: %s. Expires: %s (%d minutes)", self.access_token,
                         self.access_token_expire, expires.total_seconds() // 60)

        return True

    async def load_canvas(self, canvas_id: int) -> Optional[np.ndarray]:
        """Load the canvas with the given ID.

        Args:
            canvas_id (int): Canvas ID

        Returns:
            Optional[np.ndarray]: Canvas image in numpy array format of shape (H, W, 4)
        """
        if datetime.now() > self.access_token_expire:
            result = await self.refresh_access_token()
            if not result:
                return

        headers = {
            "User-Agent": self.user_agent,
            "Origin": "https://hot-potato.reddit.com"
        }

        async with self.session.ws_connect(PLACE_WEBSOCKET, protocols=["graphql-ws"],
                                           headers=headers) as ws:
            await ws.send_str(json.dumps({
                "type": "connection_init",
                "payload": {
                    "Authorization": f"Bearer {self.access_token}"
                }
            }))

            await ws.send_str(json.dumps({
                "id": "1",
                "type": "start",
                "payload": {
                    "variables": {
                        "input": {
                            "channel": {
                                "teamOwner": "AFD2022",
                                "category": "CANVAS",
                                "tag": str(canvas_id)
                            }
                        },
                    },
                    "extensions": {},
                    "operationName": "replace",
                    "query": GRAPHQL_CANVAS_QUERY,
                }
            }))

            async for msg in ws:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    self.logger.debug("Couldn't parse websocket msg: %s", msg.data)
                    continue

                name = data.get(
                    "payload", {}).get(
                    "data", {}).get(
                    "subscribe", {}).get(
                    "data", {}).get("name")

                if name:
                    self.logger.info("Found current canvas URL: %s", name)
                    await ws.close()

                    random_str = "".join(random.choice(string.ascii_letters) for _ in range(15))

                    async with self.session.get(f"{name}?nocache={random_str}") as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            self.logger.error(
                                "Error obtaining current canvas! HTTP Status: %d. Error:\n%s",
                                resp.status, text)

                            return

                        data = await resp.read()
                        canvas = Image.open(BytesIO(data))
                        self.logger.info("Loaded canvas ID %d (image size: %s)",
                                         canvas_id, canvas.size)

                        return canvas

    async def load_full_map(self):
        """Load the full map dynamically based on MAP_WIDTH and MAP_HEIGHT."""
        current_canvas_image = Image.new('RGBA', (MAP_WIDTH, MAP_HEIGHT), (0, 0, 0, 0))
        for i in range(MAP_HEIGHT // 1000):
            for j in range(MAP_WIDTH // 1000):
                map_id = (i * MAP_WIDTH // 1000) + j

                canvas = await self.load_canvas(map_id)  # TODO: implement if None
                current_canvas_image.paste(canvas, (j * 1000, i * 1000))

        self.current_canvas = np.array(current_canvas_image.getdata())
        self.current_canvas.resize((MAP_WIDTH, MAP_HEIGHT, 4))

        self.logger.info("Loaded full canvas (shape: %s, dtype: %s)",
                         self.current_canvas.shape, self.current_canvas.dtype)

    def get_pixels_to_update(self, order_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Get the pixels to update based on the order map.

        Args:
            order_map (np.ndarray): Order map, shape (MAP_WIDTH, MAP_HEIGHT, 4)

        Returns:
            Optional[List[Tuple[int, int]]]: List of pixels to update, shape (N, 2)
        """
        if self.current_canvas is None:
            self.logger.warning(
                "Current canvas not yet loaded, can't figure out pending pixels...")
            return

        to_update = []

        for row in range(order_map.shape[0]):
            for col in range(order_map.shape[1]):
                # Index 3 is alpha channel, ignore pixels set to transparent
                if order_map[row, col, 3] != 0:
                    if not np.array_equal(
                            order_map[row, col, : 3],
                            self.current_canvas[row, col, : 3]):
                        to_update.append((row, col))

        self.logger.info("Found %d pixels incorrectly colored.", len(to_update))

        # Randomize pixels to ensure we're not trying to update the same pixels again and again
        random.shuffle(to_update)

        return to_update

    async def place_pixel(self, row: int, col: int, color: int) -> tuple[bool, float]:
        """Place a pixel at the given row and column.

        Args:
            row (int): Row of the pixel
            col (int): Column of the pixel
            color (int): Color of the pixel

        Returns:
            tuple[bool, float]: Tuple of whether the pixel was placed and the time it took to \
                place it
        """
        if datetime.now() > self.access_token_expire:
            result = await self.refresh_access_token()

            if not result:
                return False, 60.0

        headers = {
            'Accept': '*/*',
            'Connection': 'close',
            'authorization': f"Bearer {self.access_token}",
            'Origin': 'https://hot-potato.reddit.com',
            'Referer': 'https://hot-potato.reddit.com/',
            'apollographql-client-name': 'mona-lisa',
            'apollographql-client-version': '0.0.1',
            'Content-Type': 'application/json',
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate'
        }

        if col < 1000:
            if row < 1000:
                canvas_index = 0
            else:
                canvas_index = 2
        else:
            if row < 1000:
                canvas_index = 1
            else:
                canvas_index = 3

        body = {
            'operationName': 'setPixel',
            'variables': {
                'input': {
                    'actionName': 'r/replace:set_pixel',
                    'PixelMessageData': {
                        'coordinate': {
                            'x': str(col % 1000),
                            'y': str(row % 1000),
                        },
                        'colorIndex': str(color),
                        'canvasIndex': str(canvas_index)
                    }
                }
            },
            'query': SET_PIXEL_QUERY
        }

        self.logger.info(
            "Attempting to place a pixel at (%d, %d) (canvas: %d), with color %d...", col, row,
            canvas_index, color)

        # Create a new session without any existing cookies
        async with aiohttp.ClientSession() as new_session:
            async with new_session.post(REDDIT_PLACE_SET_PIXEL_URL, headers=headers,
                                        json=body) as resp:
                if resp.status != 200:
                    self.logger.error("Error placing pixel! HTTP status %d.", resp.status)
                    text = await resp.text()
                    self.logger.error("%s", text)

                    return False, 60.0

                try:
                    data = await resp.json()
                    errors = data.get('errors')

                    if errors:
                        self.logger.error(
                            "Error placing pixel! Likely placing a new pixel too soon!")
                        next_available = errors[0].get(
                            'extensions', {}).get('nextAvailablePixelTs')

                        if next_available:
                            next_dt = datetime.fromtimestamp(float(next_available) / 1000)
                            delta = next_dt - datetime.now()
                            self.logger.info("Next available possibility: %s (%d seconds)",
                                             next_dt, delta.total_seconds())

                            return False, delta.total_seconds() + random.randint(5, 60)
                        else:
                            return False, 300.0  # wait 5 minutes by default
                    else:
                        next_available = float(
                            data['data']['act']['data'][0]['data']
                            ['nextAvailablePixelTimestamp'])
                        next_dt = datetime.fromtimestamp(next_available / 1000)
                        delta = next_dt - datetime.now()

                        self.logger.info("Success! Next pixel will be set at %s (%d seconds)",
                                         next_dt, delta.total_seconds())

                        return True, delta.total_seconds() + random.randint(5, 60)
                except json.decoder.JSONDecodeError:
                    self.logger.exception("Error placing pixel! Could not read response.")
                    return False, 60.0


async def on_request_start(session, ctx, params) -> None:
    """Handle the request start event.

    Args:
        session (_type_): Session
        ctx (_type_): _description_
        params (_type_): _description_
    """
    logging.getLogger('aiohttp.client').debug(
        "Making %s request to %s", params.method, params.url)
    logging.getLogger('aiohttp.client').debug("%s", params.headers)


class MainRunner:
    """Main runner class."""

    def __init__(self) -> None:
        """Initialize the main runner."""
        self.order_map = None

        self.trace_config = aiohttp.TraceConfig()
        self.trace_config.on_request_start.append(on_request_start)

        self.new_pixels_event = asyncio.Event()
        self.pixels_to_signal = deque()
        self.debug = False

    def set_debug(self) -> None:
        """Set debug mode."""
        self.debug = True

    async def cnc_updater(self) -> None:
        """Update the CNC."""
        while True:
            try:
                async with aiohttp.ClientSession(trace_configs=[self.trace_config]
                                                 ) as cnc_session:
                    async with CNCOrderClient(cnc_session) as cnc_client:
                        tasks = [
                            asyncio.get_running_loop().create_task(
                                cnc_client.receive_orders(self.new_map_callback)),
                            asyncio.get_running_loop().create_task(
                                cnc_client.update_pixels(
                                    self.pixels_to_signal, self.new_pixels_event))]

                        await asyncio.gather(*tasks)
            except Exception:
                logger.exception("Error with C&C updater task..., trying again in 30 seconds.")
            else:
                logger.warning("Lost connection to C&C server, trying again in 30 seconds...")

            await asyncio.sleep(30)

    def new_map_callback(self, orders: np.narray) -> None:
        """Handle a new map callback.

        Args:
            orders (np.narray): New order map
        """
        self.order_map = orders

    async def reddit_client(self, username: str, password: str, user_agent=None) -> None:
        """Run the reddit client.

        Args:
            username (str): Reddit username
            password (str): Reddit password
            user_agent (_type_, optional): _description_. Defaults to None.
        """
        async with aiohttp.ClientSession(trace_configs=[self.trace_config]) as session:
            async with RedditPlaceClient(session, username, password, user_agent,
                                         self.debug) as place_client:
                delay = 0

                while True:
                    if delay > 0:
                        await asyncio.sleep(delay)

                    await place_client.load_full_map()

                    # Compare current canvas to order map
                    to_update = place_client.get_pixels_to_update(self.order_map)

                    if len(to_update) == 0:
                        # No pixels to update, try again in 30 seconds
                        delay = 30
                    else:
                        for pixel in to_update:
                            hex = '#{:02x}{:02x}{:02x}'.format(
                                *self.order_map[pixel[0], pixel[1], :3]).upper()

                            color_index = COLOR_MAPPINGS[hex]

                            success, delay = await place_client.place_pixel(pixel[0], pixel[1],
                                                                            color_index)

                            if success:
                                self.pixels_to_signal.append((*pixel, color_index))
                                self.new_pixels_event.set()

                            break


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-u', '--user', nargs=2, action="append",
        help="Reddit username and password. Use this option multiple times to run with multiple \
            users."
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="Enable verbose output, use multiple times to increase verbosity level.")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)

    runner = MainRunner()

    if args.verbose > 0:
        runner.set_debug()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('aiohttp.client').setLevel(logging.INFO)

    if args.verbose > 1:
        warnings.filterwarnings("always", category=ResourceWarning)
        asyncio.get_running_loop().set_debug(True)
        logging.getLogger('aiohttp.client').setLevel(logging.DEBUG)

    tasks = [asyncio.create_task(runner.cnc_updater())]

    # Wait a few seconds before starting reddit clients to make sure C&C data has downloaded
    await asyncio.sleep(5)

    for username, password in args.user:
        tasks.append(runner.reddit_client(username, password))

    await asyncio.gather(*tasks)


def run() -> None:
    """Run the main function."""
    asyncio.run(main())


if __name__ == '__main__':
    run()

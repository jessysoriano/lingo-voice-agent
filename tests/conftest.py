import asyncio
import base64
import json

import pytest

from lingua_agent.config import Settings


class FakeWS:
    """Scripted stand-in for a websocket. Exceptions in the script are raised in order."""

    def __init__(self, script, hold_open=False):
        self.script, self.sent, self.hold_open = list(script), [], hold_open

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self.script:
            await asyncio.sleep(0.005)  # let the sender task interleave
            if isinstance(item, BaseException):
                raise item
            yield item if isinstance(item, str) else json.dumps(item)
        if self.hold_open:
            await asyncio.Event().wait()

    def types(self):
        return [m["type"] for m in self.sent]


def make_connect(*attempts):
    """Each call returns the next FakeWS, or raises if the next attempt is an exception."""
    queue = list(attempts)

    def connect():
        nxt = queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    return connect


async def chunks(*pieces):
    for p in pieces:
        yield p


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@pytest.fixture
def settings():
    return Settings(api_key="test-key", max_connect_attempts=3, backoff_base_s=0.001, backoff_cap_s=0.002)


@pytest.fixture
def no_sleep():
    delays = []

    async def _sleep(d):
        delays.append(d)

    _sleep.delays = delays
    return _sleep

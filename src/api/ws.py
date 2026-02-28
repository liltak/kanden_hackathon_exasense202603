"""WebSocket connection manager with Redis Pub/Sub backend for multi-instance support."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import WebSocket

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class ConnectionManager:
    """Manages WebSocket connections with Redis Pub/Sub for multi-instance broadcasting.

    When Redis is available, progress updates are published to a Redis channel
    and all API instances subscribe to broadcast to their local WS clients.
    Falls back to in-process dict when Redis is unavailable.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._redis = None
        self._pubsub = None
        self._listener_task: asyncio.Task | None = None

    async def _get_redis(self):
        """Lazy-initialize async Redis client."""
        if self._redis is None:
            try:
                from redis.asyncio import from_url

                self._redis = from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
                logger.info("Redis Pub/Sub connected")
            except Exception:
                logger.warning("Redis unavailable, falling back to in-process WS")
                self._redis = None
        return self._redis

    async def start_listener(self) -> None:
        """Start background Redis subscriber for WS progress messages."""
        redis = await self._get_redis()
        if redis is None:
            return

        self._pubsub = redis.pubsub()
        await self._pubsub.psubscribe("ws:progress:*")
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for Redis Pub/Sub messages and forward to local WS clients."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] != "pmessage":
                    continue
                channel = message["channel"]
                task_id = channel.split(":")[-1]
                payload = message["data"]
                await self._broadcast_local(task_id, payload)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Redis listener error")

    async def stop_listener(self) -> None:
        """Stop the Redis subscriber."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if task_id not in self._connections:
                self._connections[task_id] = set()
            self._connections[task_id].add(websocket)
        logger.info("WS connected for task %s", task_id)

    async def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            conns = self._connections.get(task_id)
            if conns:
                conns.discard(websocket)
                if not conns:
                    del self._connections[task_id]
        logger.info("WS disconnected for task %s", task_id)

    async def send_progress(
        self,
        task_id: str,
        step: str,
        progress: float,
        message: str,
    ) -> None:
        """Publish progress — via Redis if available, otherwise local broadcast."""
        payload = json.dumps({
            "task_id": task_id,
            "step": step,
            "progress": progress,
            "message": message,
        })

        redis = await self._get_redis()
        if redis:
            await redis.publish(f"ws:progress:{task_id}", payload)
        else:
            await self._broadcast_local(task_id, payload)

    async def _broadcast_local(self, task_id: str, payload: str) -> None:
        """Send payload to all locally connected WebSocket clients for a task."""
        async with self._lock:
            conns = self._connections.get(task_id, set()).copy()
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                logger.debug("Failed to send WS message, removing connection")
                await self.disconnect(task_id, ws)


manager = ConnectionManager()

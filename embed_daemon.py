#!/usr/bin/env python3
"""FastEmbed Unix-socket daemon with backward-compatible query/passage modes."""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from fastembed import TextEmbedding


SOCKET_PATH = "/run/kb-embed/embed.sock"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768
MAX_WORKERS = 4
MAX_REQUEST_BYTES = 65_536
WATCHDOG_INTERVAL = 5


class EmbeddingBatch(Protocol):
    def __iter__(self): ...


class EmbeddingModel(Protocol):
    def query_embed(self, texts: list[str]) -> EmbeddingBatch: ...

    def passage_embed(self, texts: list[str]) -> EmbeddingBatch: ...


def parse_request(raw: str) -> tuple[str, str]:
    """Return (mode, text); legacy plain-text requests remain query embeddings."""
    value = raw.strip()
    if not value:
        raise ValueError("empty_request")
    try:
        request = json.loads(value)
    except json.JSONDecodeError:
        return "query", value
    if not isinstance(request, dict) or "mode" not in request:
        return "query", value
    if set(request) != {"mode", "text"}:
        raise ValueError("invalid_request_fields")
    mode = request.get("mode")
    text = request.get("text")
    if mode not in {"query", "passage"}:
        raise ValueError("invalid_embedding_mode")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("invalid_embedding_text")
    return mode, text.strip()


def embed_text(model: EmbeddingModel, text: str, mode: str) -> list[float]:
    generator = model.passage_embed([text]) if mode == "passage" else model.query_embed([text])
    batch = list(generator)
    if len(batch) != 1:
        raise ValueError("invalid_embedding_batch")
    vector = [float(value) for value in batch[0].tolist()]
    if len(vector) != EMBEDDING_DIMENSION or not all(math.isfinite(value) for value in vector):
        raise ValueError("invalid_embedding_vector")
    return vector


def sd_notify(message: str) -> None:
    socket_path = os.environ.get("NOTIFY_SOCKET")
    if not socket_path:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_socket:
            notify_socket.connect(socket_path)
            notify_socket.sendall(message.encode())
    except OSError:
        pass


def watchdog_loop() -> None:
    while True:
        sd_notify("WATCHDOG=1")
        threading.Event().wait(WATCHDOG_INTERVAL)


def _read_request(connection: socket.socket) -> str:
    data = bytearray()
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("request_too_large")
        if b"\n" in chunk:
            break
    line = bytes(data).split(b"\n", 1)[0]
    return line.decode("utf-8")


def handle_client(connection: socket.socket, model: EmbeddingModel) -> None:
    with connection:
        connection.settimeout(30)
        try:
            mode, text = parse_request(_read_request(connection))
            embedding = embed_text(model, text, mode)
            connection.sendall((json.dumps(embedding) + "\n").encode())
        except Exception as exc:
            print(f"Embedding error: {type(exc).__name__}: {exc}", flush=True)
            try:
                connection.sendall(b"null\n")
            except OSError:
                pass


def main() -> None:
    print("Loading FastEmbed model...", flush=True)
    model = TextEmbedding(MODEL_NAME)
    print("Model ready.", flush=True)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o660)
    server.listen(16)
    print(f"kb-embed daemon ready on {SOCKET_PATH}", flush=True)
    sys.stdout.flush()

    threading.Thread(target=watchdog_loop, daemon=True).start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        while True:
            try:
                connection, _ = server.accept()
                pool.submit(handle_client, connection, model)
            except OSError as exc:
                print(f"Accept error: {exc}", flush=True)
                break


if __name__ == "__main__":
    main()

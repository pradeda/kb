#!/usr/bin/env python3
import sys, json
from fastembed import TextEmbedding

text = sys.stdin.read().strip()
if not text:
    sys.exit(1)

model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
embedding = list(model.query_embed([text]))[0].tolist()
print(json.dumps(embedding))

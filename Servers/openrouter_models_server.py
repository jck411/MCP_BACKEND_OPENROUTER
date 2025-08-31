# openrouter_models_server.py
# Minimal OpenRouter Models MCP server with Ruff-friendly typing and style
# Adds explicit popularity sorting (based on API order) and a get_sorting_methods tool.

import asyncio
import datetime as dt
import json
import os
import re
from datetime import datetime
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ----------------------------
# Config
# ----------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = os.getenv("OPENROUTER_CACHE_PATH", "openrouter_models_cache.json")
CACHE_TTL_SECONDS = int(os.getenv("OPENROUTER_CACHE_TTL_SECONDS", "21600"))  # 6 hours
MAX_RETRY = 3

# Heuristic: values above this are likely milliseconds since epoch.
UNIX_MS_THRESHOLD = 1_000_000_000_000

mcp = FastMCP("OpenRouter Models (Lite)")


# ----------------------------
# Cache layer (JSON file + in-memory)
# ----------------------------
class ModelCache:
    def __init__(self, path: str = CACHE_PATH):
        self.path = path
        self.lock = asyncio.Lock()
        self._mem: dict[str, Any] | None = None

    def _now(self) -> datetime:
        return datetime.now(dt.UTC)

    async def load(self) -> dict[str, Any]:
        if self._mem is not None:
            return self._mem
        try:
            with open(self.path, encoding="utf-8"):
                pass
            with open(self.path, encoding="utf-8") as f:
                self._mem = json.load(f)
        except Exception:
            self._mem = {"fetched_at": None, "models": []}
        return self._mem

    async def save(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        # Preserve incoming order exactly as the API returns it (already popularity-sorted).
        data: dict[str, Any] = {"fetched_at": self._now().isoformat(), "models": models}
        self._mem = data
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)
        return data

    async def is_stale(self) -> bool:
        data = await self.load()
        ts = data.get("fetched_at")
        if not ts:
            return True
        try:
            last = datetime.fromisoformat(ts)
            age = (self._now() - last).total_seconds()
            return age > CACHE_TTL_SECONDS
        except Exception:
            return True

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        # App Attribution headers (optional but recommended)
        # https://openrouter.ai/docs/app-attribution
        ref = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if ref:
            h["HTTP-Referer"] = ref
        if title:
            h["X-Title"] = title
        return h

    async def fetch(self, category: str | None = None) -> list[dict[str, Any]]:
        # Note: /api/v1/models supports only `category` as a filter and returns
        # models sorted from most to least used (popularity).
        params = {"category": category} if category else None
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            err: Exception | None = None
            for attempt in range(MAX_RETRY):
                try:
                    r = await client.get(
                        OPENROUTER_API_URL,
                        headers=self._headers(),
                        params=params,
                    )
                    r.raise_for_status()
                    js = r.json()
                    return js.get("data", [])
                except Exception as e:
                    err = e
                    await asyncio.sleep(0.4 * (attempt + 1))
            raise RuntimeError(f"OpenRouter fetch failed: {err}")

    async def get_models(self, force: bool = False, category: str | None = None) -> list[dict[str, Any]]:
        async with self.lock:
            if not force and not await self.is_stale():
                return (await self.load())["models"]
            models = await self.fetch(category=category)
            await self.save(models)
            return models


cache = ModelCache()


# ----------------------------
# Helpers: derived fields + filtering/sorting
# ----------------------------
def _to_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _per_million(v: Any) -> float | None:
    f = _to_float(v)
    return None if f is None else f * 1_000_000


def _as_unix_seconds(v: Any) -> int | None:
    """
    Normalize various timestamp representations into seconds since epoch (UTC).
    Accepts int/float seconds, milliseconds, digit strings, or ISO-8601 strings.
    """
    if v is None:
        return None

    result: int | None = None

    if isinstance(v, int | float):
        n = int(v)
        if n > UNIX_MS_THRESHOLD:
            n //= 1000
        result = n if n > 0 else None

    elif isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            try:
                n = int(float(s))
                if n > UNIX_MS_THRESHOLD:
                    n //= 1000
                result = n if n > 0 else None
            except Exception:
                result = None
        else:
            try:
                s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
                dt_obj = datetime.fromisoformat(s2)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=dt.UTC)
                result = int(dt_obj.timestamp())
            except Exception:
                result = None

    return result


def _to_iso_utc(ts: int | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=dt.UTC).isoformat()
    except Exception:
        return None


def _derive(model: dict[str, Any]) -> dict[str, Any]:
    """Add normalized fields we want to search/sort over, without mutating the original."""
    m = dict(model)

    # Normalize created into a numeric timestamp for reliable sorting
    created_ts = _as_unix_seconds(m.get("created"))
    m["created_ts"] = created_ts
    m["created_iso"] = _to_iso_utc(created_ts)

    pricing: dict[str, Any] = m.get("pricing", {}) or {}
    m["price_prompt_per_1m"] = _per_million(pricing.get("prompt"))
    m["price_completion_per_1m"] = _per_million(pricing.get("completion"))
    m["price_request_per_1m"] = _per_million(pricing.get("request"))
    m["price_image_per_1m"] = _per_million(pricing.get("image"))
    m["price_input_cache_read_per_1m"] = _per_million(pricing.get("input_cache_read"))
    m["price_input_cache_write_per_1m"] = _per_million(pricing.get("input_cache_write"))
    m["price_web_search_per_1m"] = _per_million(pricing.get("web_search"))
    m["price_internal_reasoning_per_1m"] = _per_million(pricing.get("internal_reasoning"))

    arch: dict[str, Any] = m.get("architecture") or {}
    m["tokenizer"] = arch.get("tokenizer")
    m["input_modalities"] = arch.get("input_modalities") or []
    m["output_modalities"] = arch.get("output_modalities") or []

    supported: list[str] = m.get("supported_parameters") or []
    m["supports_tools"] = ("tools" in supported) or ("tool_choice" in supported)
    m["supports_reasoning"] = "reasoning" in supported
    m["supports_structured_outputs"] = "structured_outputs" in supported

    tp: dict[str, Any] = m.get("top_provider") or {}
    m["is_moderated"] = tp.get("is_moderated")
    if not m.get("context_length"):
        m["context_length"] = tp.get("context_length")

    slug = m.get("canonical_slug")
    m["provider"] = slug.split("/", 1)[0] if isinstance(slug, str) and "/" in slug else None

    # popularity_rank will be added later (based on API list order)
    return m


def _text_match(q: str, m: dict[str, Any]) -> bool:
    ql = q.lower()
    return (ql in (m.get("name") or "").lower()) or (ql in (m.get("description") or "").lower())


def _parse_nl(q: str | None) -> dict[str, Any]:
    if not q:
        return {}
    s = q.lower()
    out: dict[str, Any] = {}

    if any(term in s for term in ["free", "no cost", "zero cost", "$0"]):
        out["free_only"] = True
    if any(term in s for term in ["tools", "tool call", "function call"]):
        out["has_tools"] = True
    if "reasoning" in s:
        out["has_reasoning"] = True

    # context hints
    if "128k" in s or "128000" in s:
        out["min_context_length"] = 128000
    elif "100k" in s or "100000" in s:
        out["min_context_length"] = 100000
    elif "32k" in s or "32000" in s:
        out["min_context_length"] = 32000

    # price hints like "under $2"
    m = re.search(r"(?:under|less than|cheaper than)\s*\$?(\d+(?:\.\d+)?)", s)
    if m:
        price = float(m.group(1))
        out["max_completion_price" if "completion" in s else "max_prompt_price"] = price

    # modality hints
    if "image" in s and ("input" in s or "vision" in s):
        out["input_modality"] = "image"
    if "image" in s and ("output" in s or "generate" in s):
        out["output_modality"] = "image"
    if "audio" in s and "input" in s:
        out["input_modality"] = "audio"
    if "audio" in s and ("output" in s or "generate" in s):
        out["output_modality"] = "audio"

    # skip free-form text search if the query looks structural
    out["_skip_text_search"] = any(
        term in s
        for term in [
            "models with",
            "models that",
            "show me",
            "find",
            "list",
            "support",
            "have",
            "can",
            "input",
            "output",
        ]
    )
    return out


def _sort_key(field: str, order: str):
    """Sorts with None-last semantics for numbers; stable for strings."""
    asc = order.lower() == "asc"

    def key(m: dict[str, Any]):
        if field == "prompt_price":
            v = m.get("pricing_per_1m", {}).get("prompt")
            if v is None:
                v = m.get("price_prompt_per_1m")
        elif field == "completion_price":
            v = m.get("pricing_per_1m", {}).get("completion")
            if v is None:
                v = m.get("price_completion_per_1m")
        elif field == "context_length":
            v = m.get("context_length")
        elif field == "name":
            v = m.get("name") or ""
        elif field == "popularity":
            # Lower popularity_rank means more popular; keep None last.
            v = m.get("popularity_rank")
        else:
            # Use normalized numeric timestamp by default
            v = m.get("created_ts")

        # Numbers: (missing -> tail). Invert sign for desc to keep None-last.
        if isinstance(v, int | float):
            return (0, v if asc else -v)
        if v is None:
            return (1, 0)

        # Strings: lexicographic; invert via char-map for desc
        s = str(v)
        return (0, s if asc else "".join(chr(255 - ord(c)) for c in s))

    return key


# ----------------------------
# MCP Tools
# ----------------------------
@mcp.tool()
async def refresh_models(category: str | None = None) -> str:
    """
    Fetch and cache the latest models list from OpenRouter.

    Notes:
    - Endpoint: GET /api/v1/models
    - Query params: category (optional)
    - API returns models already sorted from most to least used (popularity).
    """
    models = await cache.get_models(force=True, category=category)
    return f"✅ Cached {len(models)} models from /api/v1/models"


@mcp.tool()
async def get_sorting_methods() -> dict[str, Any]:
    """
    Return supported sorting methods and their semantics in a machine-readable way
    so the LLM knows what it can do.

    Fields:
      - key: the value to pass to `sort_by`
      - order: allowed order values
      - description: how the sort works
      - default: which sort is used by default in this tool
    """
    return {
        "default": "created",
        "methods": [
            {
                "key": "popularity",
                "order": ["asc", "desc"],
                "description": "API list order rank; lower rank = more popular",
            },
            {
                "key": "created",
                "order": ["asc", "desc"],
                "description": "When the model was added to OpenRouter (normalized timestamp)",
            },
            {"key": "name", "order": ["asc", "desc"], "description": "Alphabetical by display name"},
            {"key": "context_length", "order": ["asc", "desc"], "description": "Max context window"},
            {
                "key": "prompt_price",
                "order": ["asc", "desc"],
                "description": "USD per 1M prompt tokens (client-derived)",
            },
            {
                "key": "completion_price",
                "order": ["asc", "desc"],
                "description": "USD per 1M completion tokens (client-derived)",
            },
        ],
    }


@mcp.tool()
async def get_statistics() -> dict[str, Any]:
    """
    Return key statistics about the cached models.
    """
    data = await cache.load()
    # Attach popularity_rank based on preserved API order
    models = []
    for idx, raw in enumerate(data.get("models", [])):
        dm = _derive(raw)
        dm["popularity_rank"] = idx  # lower = more popular
        models.append(dm)

    total = len(models)
    if total == 0:
        return {"total_models": 0, "message": "Cache empty. Run refresh_models()."}

    def price_stat(field: str) -> dict[str, float | None]:
        vals = [m.get(field) for m in models if isinstance(m.get(field), int | float) and m.get(field) > 0]
        return {
            "available_models": len(vals),
            "min_per_1m_tokens": min(vals) if vals else None,
            "max_per_1m_tokens": max(vals) if vals else None,
            "avg_per_1m_tokens": (sum(vals) / len(vals)) if vals else None,
        }

    ctx_vals = [m.get("context_length") for m in models if isinstance(m.get("context_length"), int)]
    tokenizer_counts: dict[str, int] = {}
    for m in models:
        t = m.get("tokenizer")
        if t:
            tokenizer_counts[t] = tokenizer_counts.get(t, 0) + 1

    return {
        "total_models": total,
        "last_fetched": data.get("fetched_at"),
        "free_models": sum(
            1
            for m in models
            if (m.get("price_prompt_per_1m") in (0.0, None)) and (m.get("price_completion_per_1m") in (0.0, None))
        ),
        "pricing": {
            "prompt": price_stat("price_prompt_per_1m"),
            "completion": price_stat("price_completion_per_1m"),
            "image": price_stat("price_image_per_1m"),
        },
        "capabilities": {
            "tools": sum(1 for m in models if m.get("supports_tools")),
            "reasoning": sum(1 for m in models if m.get("supports_reasoning")),
            "structured_outputs": sum(1 for m in models if m.get("supports_structured_outputs")),
        },
        "context_length": {
            "min": min(ctx_vals) if ctx_vals else None,
            "max": max(ctx_vals) if ctx_vals else None,
            "avg": round(sum(ctx_vals) / len(ctx_vals)) if ctx_vals else None,
        },
        "tokenizers": dict(sorted(tokenizer_counts.items(), key=lambda kv: -kv[1])[:10]),
    }


@mcp.tool()
async def search_models(
    query: str | None = None,
    max_prompt_price: float | None = None,
    max_completion_price: float | None = None,
    min_context_length: int | None = None,
    has_tools: bool | None = None,
    has_reasoning: bool | None = None,
    free_only: bool = False,
    tokenizer: str | None = None,
    input_modality: str | None = None,
    output_modality: str | None = None,
    unmoderated_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "created",  # popularity | created | name | context_length | prompt_price | completion_price
    sort_order: str = "desc",  # asc | desc
) -> dict[str, Any]:
    """
    Search the cached list with simple filters. Call refresh_models first if needed.

    Sorts:
      - popularity: preserves API order (lower rank = more popular)
      - created: normalized timestamp (seconds since epoch)
      - name: lexicographic
      - context_length: numeric
      - prompt_price / completion_price: USD per 1M tokens (derived)

    Returns JSON so clients can render without extra parsing.
    """
    all_raw = (await cache.load()).get("models", [])
    if not all_raw:
        return {"message": "Cache empty. Run refresh_models().", "results": []}

    # Natural-language hinting
    hints = _parse_nl(query)
    if max_prompt_price is None and "max_prompt_price" in hints:
        max_prompt_price = hints["max_prompt_price"]
    if max_completion_price is None and "max_completion_price" in hints:
        max_completion_price = hints["max_completion_price"]
    if min_context_length is None and "min_context_length" in hints:
        min_context_length = hints["min_context_length"]
    if has_tools is None and "has_tools" in hints:
        has_tools = hints["has_tools"]
    if has_reasoning is None and "has_reasoning" in hints:
        has_reasoning = hints["has_reasoning"]
    if "free_only" in hints:
        free_only = hints["free_only"]
    if input_modality is None and "input_modality" in hints:
        input_modality = hints["input_modality"]
    if output_modality is None and "output_modality" in hints:
        output_modality = hints["output_modality"]

    # Attach popularity_rank based on preserved API order
    models = []
    for idx, raw in enumerate(all_raw):
        dm = _derive(raw)
        dm["popularity_rank"] = idx  # lower = more popular
        models.append(dm)

    # Filtering
    results: list[dict[str, Any]] = []
    for m in models:
        # textual query (skip if NL looks structural)
        if query and not hints.get("_skip_text_search") and not _text_match(query, m):
            continue

        if max_prompt_price is not None:
            p = m.get("price_prompt_per_1m")
            if p is not None and p > max_prompt_price:
                continue

        if max_completion_price is not None:
            c = m.get("price_completion_per_1m")
            if c is not None and c > max_completion_price:
                continue

        if (min_context_length is not None) and ((m.get("context_length") or 0) < min_context_length):
            continue

        if has_tools and not m.get("supports_tools"):
            continue

        if has_reasoning and not m.get("supports_reasoning"):
            continue

        if free_only and not (
            (m.get("price_prompt_per_1m") in (0.0, None)) and (m.get("price_completion_per_1m") in (0.0, None))
        ):
            continue

        if tokenizer and m.get("tokenizer") != tokenizer:
            continue

        if input_modality and input_modality not in (m.get("input_modalities") or []):
            continue

        if output_modality and output_modality not in (m.get("output_modalities") or []):
            continue

        if unmoderated_only and m.get("is_moderated") is True:
            continue

        results.append(
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "canonical_slug": m.get("canonical_slug"),
                "provider": m.get("provider"),
                "popularity_rank": m.get("popularity_rank"),
                # Keep original created (opaque) + normalized ts + iso for display
                "created": m.get("created"),
                "created_ts": m.get("created_ts"),
                "created_iso": m.get("created_iso"),
                "context_length": m.get("context_length"),
                "tokenizer": m.get("tokenizer"),
                "input_modalities": m.get("input_modalities"),
                "output_modalities": m.get("output_modalities"),
                "is_moderated": m.get("is_moderated"),
                "supports": {
                    "tools": m.get("supports_tools"),
                    "reasoning": m.get("supports_reasoning"),
                    "structured_outputs": m.get("supports_structured_outputs"),
                },
                "pricing_per_1m": {
                    "prompt": m.get("price_prompt_per_1m"),
                    "completion": m.get("price_completion_per_1m"),
                    "image": m.get("price_image_per_1m"),
                    "request": m.get("price_request_per_1m"),
                },
                "description": m.get("description"),
            }
        )

    # Sorting + pagination
    sort_map = {"popularity", "created", "name", "context_length", "prompt_price", "completion_price"}
    field = sort_by if sort_by in sort_map else "created"
    order = sort_order if sort_order in ("asc", "desc") else "desc"
    results = sorted(results, key=_sort_key(field, order))
    paged = results[offset : offset + limit]

    return {
        "count": len(results),
        "offset": offset,
        "limit": limit,
        "sort_by": field,
        "sort_order": order,
        "results": paged,
    }


if __name__ == "__main__":
    mcp.run()

"""
FTW Predictor V49 — APEX ENSEMBLE (Multi-Signal + Beta Reliability + Calibration)
Python 3.10+
Deps: aiohttp
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import math
import os
import signal
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web

# ============================================================
# CONFIG
# ============================================================

DEFAULT_HISTORY_URL = (
    "https://sky-predictor-1012593186417.asia-southeast1.run.app/"
    "api/wingo-history-1m-1000"
)

@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "9279667822:AAEfV9c7yOhYExvtr8lrnEdqQZGvs8wieWQ")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004368817214")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor1_v49_apex.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (V49 APEX)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token: raise ValueError("TELEGRAM_TOKEN is required")
        if not self.chat: raise ValueError("TELEGRAM_CHAT is required")

CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v49")

# ============================================================
# HELPERS & MATH
# ============================================================

def clip(v: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, float(v)))

def sigmoid(z: float) -> float:
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))

def escape(v: Any) -> str:
    return html.escape(str(v))

def numeric(v: Any) -> int | None:
    if isinstance(v, bool): return None
    text = str(v).strip()
    if not text.isascii() or not text.isdigit(): return None
    try: return int(text)
    except ValueError: return None

def weighted_probability(obs: list[tuple[int, float]]) -> tuple[float, float]:
    if not obs: return 0.5, 0.0
    wsum = sum(w for _, w in obs)
    w2sum = sum(w * w for _, w in obs)
    neff = (wsum * wsum / w2sum) if w2sum > 0 else 0.0
    p = (4.0 + sum(v * w for v, w in obs)) / (8.0 + wsum)
    return clip(p), neff

def recency_weight(index: int, n: int, half_life: float = 330.0) -> float:
    age = max(0, n - 1 - index)
    return 0.5 ** (age / half_life)

def get_regime(seq: list[int]) -> str:
    tail = seq[-24:]
    if len(tail) < 8: return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    flip_rate = flips / max(1, len(tail) - 1)
    run = 1
    for v in reversed(tail[:-1]):
        if v != tail[-1]: break
        run += 1
    if run >= 5: return "RUN"
    if flip_rate >= 0.72: return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20: return "IMBAL"
    return "MIXED"

def entropy_bits(seq: list[int]) -> float:
    if not seq: return 0.0
    c = Counter(seq)
    n = len(seq)
    return -sum((f/n) * math.log2(f/n) for f in c.values())

def cusum_change(seq: list[int], thr: float = 2.5) -> int:
    if len(seq) < 20: return 0
    mean = sum(seq[-20:]) / 20.0
    sp, sn = 0.0, 0.0
    for v in seq[-20:]:
        sp = max(0, sp + (v - mean) - 0.5)
        sn = min(0, sn + (v - mean) + 0.5)
    if sp > thr: return 1
    if sn < -thr: return -1
    return 0

def family_run(seq: list[int]) -> list[tuple[int, int]]:
    if not seq: return []
    out, cur, c = [], seq[0], 1
    for x in seq[1:]:
        if x == cur: c += 1
        else:
            out.append((cur, c)); cur, c = x, 1
    out.append((cur, c))
    return out

def family_signature(seq: list[int]) -> tuple:
    return tuple((int(v), c) for v, c in family_run(seq))

def parse_history(payload: Any) -> list[dict[str, Any]]:
    obj = payload
    for _ in range(8):
        if isinstance(obj, list): break
        if not isinstance(obj, dict): return []
        nxt = None
        for k in ("data", "list", "result", "items", "history", "records"):
            c = obj.get(k)
            if isinstance(c, (list, dict)):
                nxt = c; break
        if nxt is None: return []
        obj = nxt
    if not isinstance(obj, list): return []

    records = {}
    for item in obj:
        if not isinstance(item, dict): continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        num = numeric(item.get("number", item.get("value")))
        if issue is None or num is None or not 0 <= num <= 9: continue
        records[str(issue)] = {"issue": str(issue), "num": num, "size": int(num >= 5)}
    return sorted(records.values(), key=lambda r: int(r["issue"]))

# ============================================================
# BETA RELIABILITY TRACKER
# ============================================================

class BetaTracker:
    """Beta distribution over win probability, with exponential forgetting."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, decay: float = 0.985):
        self.alpha, self.beta, self.decay = alpha, beta, decay

    def update(self, won: bool) -> None:
        self.alpha = self.alpha * self.decay + (1.0 if won else 0.0)
        self.beta  = self.beta  * self.decay + (0.0 if won else 1.0)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return max(0.0, self.alpha + self.beta - 2.0)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    def to_dict(self) -> dict[str, float]:
        return {"a": self.alpha, "b": self.beta}

    @classmethod
    def from_dict(cls, d: dict[str, float] | None, decay: float = 0.985) -> "BetaTracker":
        if not d: return cls(decay=decay)
        return cls(float(d.get("a", 1.0)), float(d.get("b", 1.0)), decay)

# ============================================================
# SIGNAL FRAMEWORK
# ============================================================

@dataclass
class Signal:
    name: str
    side: str              # "BIG" | "SMALL"
    strength: float        # raw confidence 0.5..1.0
    weight: float          # reliability weight
    reason: str = ""
    detail: str = ""

    @property
    def vote(self) -> float:
        """Signed vote: +strength*weight for BIG, -strength*weight for SMALL."""
        s = (self.strength - 0.5) * 2.0 * self.weight
        return s if self.side == "BIG" else -s

# ============================================================
# SIGNAL PRODUCERS
# ============================================================

def signal_family_pattern(seq: list[int], db: "Database", regime: str) -> Signal | None:
    n = len(seq)
    if n < 20: return None
    candidates = []

    for length in range(10, 4, -1):
        if n < length * 2: continue
        suffix = seq[-length:]
        sig = family_signature(suffix)
        obs = []
        for i in range(n - length):
            if family_signature(seq[i:i + length]) == sig:
                obs.append((seq[i + length], recency_weight(i + length, n, 420.0)))
        if len(obs) < CFG.min_support: continue

        p, eff = weighted_probability(obs)
        big_c = sum(1 for v, _ in obs if v == 1)
        small_c = len(obs) - big_c
        pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in sig)
        key = "size_family:" + pattern_str
        adjusted_p, rel = db.adapt_pattern(key, regime, p, is_number=False)
        if rel < 0.15: continue

        side = "BIG" if adjusted_p >= 0.5 else "SMALL"
        score = rel * math.sqrt(max(1.0, eff))
        candidates.append((score, side, pattern_str, big_c, small_c, adjusted_p, rel))

    if not candidates: return None

    big_votes = sum(1 for c in candidates if c[1] == "BIG")
    small_votes = len(candidates) - big_votes
    if big_votes > small_votes: chosen = "BIG"
    elif small_votes > big_votes: chosen = "SMALL"
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]

    best = max((c for c in candidates if c[1] == chosen), key=lambda x: x[0])
    _, side, name, big_c, small_c, p, rel = best
    return Signal(
        name=f"family({name})",
        side=side,
        strength=max(p, 1 - p),
        weight=rel * math.sqrt(max(1.0, big_c + small_c)),
        reason="family pattern",
        detail=f"{name} B[{big_c}] S[{small_c}]",
    )

def signal_skipgram(seq: list[int], db: "Database") -> Signal | None:
    """Length-6 patterns with one wildcard in the middle."""
    n = len(seq)
    if n < 30: return None

    def wildcard_sig(sl: list[int]) -> tuple:
        if len(sl) < 6: return ()
        return (sl[0], sl[1], -1, sl[3], sl[4], sl[5])

    suffix = seq[-6:]
    target_sig = wildcard_sig(suffix)
    obs = []
    for i in range(n - 6):
        if wildcard_sig(seq[i:i + 6]) == target_sig:
            obs.append((seq[i + 6], recency_weight(i + 6, n, 500.0)))
    if len(obs) < CFG.min_support: return None

    p, eff = weighted_probability(obs)
    side = "BIG" if p >= 0.5 else "SMALL"
    key = "skipgram:" + ",".join(map(str, target_sig))
    adjusted_p, rel = db.adapt_pattern(key, "ANY", p, is_number=False)
    if rel < 0.15: return None
    return Signal(
        name="skipgram",
        side=side,
        strength=max(adjusted_p, 1 - adjusted_p),
        weight=rel * math.sqrt(max(1.0, eff)) * 0.85,
        reason="skipgram pattern",
        detail=f"n={len(obs)}",
    )

def signal_trend(seq: list[int]) -> Signal | None:
    if len(seq) < 50: return None
    # Weighted blend of 20/50/100 windows with decay emphasis on newest
    def win_ratio(w: int) -> tuple[float, float]:
        tail = seq[-w:]
        if not tail: return 0.5, 0.0
        weights = [0.5 ** ((len(tail) - 1 - i) / (w / 2)) for i in range(len(tail))]
        p = sum(v * wgt for v, wgt in zip(tail, weights)) / sum(weights)
        return p, math.sqrt(sum(weights) ** 2 / sum(x * x for x in weights)) if sum(x * x for x in weights) > 0 else 0.0

    p20, e20 = win_ratio(20)
    p50, e50 = win_ratio(50)
    p100, e100 = win_ratio(100)
    blended = (p20 * e20 + p50 * e50 + p100 * e100) / max(1e-6, (e20 + e50 + e100))
    dev = abs(blended - 0.5)
    if dev < 0.05: return None
    side = "BIG" if blended > 0.5 else "SMALL"
    strength = clip(0.5 + dev, 0.5, 0.85)
    return Signal(
        name="trend",
        side=side,
        strength=strength,
        weight=0.6,
        reason="trend blend",
        detail=f"20w={p20:.2f} 50w={p50:.2f} 100w={p100:.2f}",
    )

def signal_streak_break(seq: list[int]) -> Signal | None:
    """Survival analysis on current run-length."""
    if len(seq) < 80: return None
    cur = seq[-1]
    run = 1
    for v in reversed(seq[:-1]):
        if v == cur: run += 1
        else: break
    # Empirically count how often a run of length >= run continues vs breaks
    continues, breaks = 0, 0
    streak = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            streak += 1
            if streak - 1 == run:  # we observed a moment where run length == run+1
                pass
        else:
            if streak >= run:
                breaks += 1
            streak = 1
    # Count continuations at length run
    streak = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            streak += 1
            if streak > run:
                continues += 1
        else:
            streak = 1

    total = continues + breaks
    if total < 5: return None
    p_continue = continues / total
    if abs(p_continue - 0.5) < 0.10: return None
    side = "BIG" if (cur == 1 and p_continue >= 0.5) or (cur == 0 and p_continue < 0.5) else "SMALL"
    strength = clip(0.5 + abs(p_continue - 0.5), 0.5, 0.75)
    return Signal(
        name="streak_break",
        side=side,
        strength=strength,
        weight=0.7,
        reason=f"streak={run} continue_p={p_continue:.2f}",
        detail=f"continues={continues} breaks={breaks}",
    )

def signal_cusum(seq: list[int]) -> Signal | None:
    d = cusum_change(seq)
    if d == 0: return None
    side = "BIG" if d == 1 else "SMALL"
    return Signal(
        name="cusum",
        side=side,
        strength=0.58,
        weight=0.8,
        reason="CUSUM break detected",
        detail=f"dir={d}",
    )

def signal_number_skew(seq: list[int], nums: list[int]) -> Signal | None:
    """Uses number distribution to vote on size side."""
    if len(nums) < 100: return None
    recent = nums[-50:]
    c = Counter(recent)
    big = sum(c.get(n, 0) for n in range(5, 10))
    small = sum(c.get(n, 0) for n in range(0, 5))
    total = big + small
    if total == 0: return None
    p_big = big / total
    if abs(p_big - 0.5) < 0.05: return None
    side = "BIG" if p_big > 0.5 else "SMALL"
    return Signal(
        name="num_skew",
        side=side,
        strength=clip(0.5 + abs(p_big - 0.5) * 0.7, 0.5, 0.72),
        weight=0.5,
        reason=f"big%={p_big:.2f}",
        detail=f"big={big} small={small}",
    )

# ============================================================
# DEEP NUMBER TRACKER
# ============================================================

def track_numbers(nums: list[int]) -> tuple[list[float], dict[str, Any]]:
    """Blend of: decayed freq, Markov(1/2/3), post-streak-break stats."""
    if not nums:
        return [0.1] * 10, {"model": "none"}
    n = len(nums)

    # 1) Exponential-decay frequency
    freq = [1.0] * 10
    for i, v in enumerate(nums):
        w = recency_weight(i, n, half_life=250.0)
        freq[v] += w

    # 2) Markov orders 1..3
    markov = [1.0] * 10
    for order in (1, 2, 3):
        if n <= order: continue
        ctx = tuple(nums[-order:])
        obs = Counter()
        for i in range(order, n):
            if tuple(nums[i - order:i]) == ctx:
                obs[nums[i]] += 1
        tot = sum(obs.values())
        if tot >= 2:
            for k in range(10):
                markov[k] += (obs.get(k, 0) / tot) * (3.0 + order)

    # 3) Post-streak-break numbers
    post_break = Counter()
    streak = 1
    for i in range(1, n):
        if nums[i] == nums[i - 1]:
            streak += 1
        else:
            if streak >= 2 and i < n - 0:  # was a break after a streak
                post_break[nums[i]] += 1
            streak = 1

    # 4) Combine
    dist = [0.0] * 10
    for k in range(10):
        dist[k] = freq[k] * 0.4 + markov[k] * 0.5 + (1.0 + post_break.get(k, 0)) * 0.6
    tot = sum(dist)
    dist = [x / tot for x in dist] if tot > 0 else [0.1] * 10

    return dist, {
        "model": "decay+markov+streak",
        "hot3": [k for k, _ in sorted(enumerate(dist), key=lambda x: -x[1])[:3]],
    }

# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, path: str):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA busy_timeout=20000")
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS history(issue TEXT PRIMARY KEY, num INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(
                issue TEXT PRIMARY KEY,
                created REAL NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                actual INTEGER,
                won_number INTEGER NOT NULL DEFAULT 0,
                predicted_p REAL NOT NULL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS pattern_models(name TEXT PRIMARY KEY, n REAL NOT NULL, loss REAL NOT NULL, failures INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS signal_betas(name TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, issue TEXT, text TEXT NOT NULL, created REAL NOT NULL, expires REAL, sent INTEGER NOT NULL DEFAULT 0);
        """)
        # Migration-safe: add predicted_p column if missing on upgrade
        with contextlib.suppress(sqlite3.OperationalError):
            self.c.execute("ALTER TABLE predictions ADD COLUMN predicted_p REAL NOT NULL DEFAULT 0.0")

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try: yield
        except Exception:
            if self.c.in_transaction: self.c.execute("ROLLBACK")
            raise
        else:
            if self.c.in_transaction: self.c.execute("COMMIT")

    def get(self, key: str, default: Any = None) -> Any:
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute("INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for r in records:
                self.c.execute("INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN (SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,)
            )

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute("SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER) ASC").fetchall()
        return [{"issue": r["issue"], "num": int(r["num"]), "size": int(int(r["num"]) >= 5)} for r in rows]

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute("SELECT issue,payload,predicted_p FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER) ASC").fetchall()

    def learn_pattern(self, key: str, probability: float, actual: int) -> None:
        row = self.c.execute("SELECT n, loss, failures FROM pattern_models WHERE name=?", (key,)).fetchone()
        old_n, old_loss, old_fails = (float(row["n"]), float(row["loss"]), int(row["failures"])) if row else (0.0, 0.0, 0)
        n = 0.985 * old_n + 1.0
        loss = 0.985 * old_loss + (clip(probability) - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        fails = old_fails + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO pattern_models(name,n,loss,failures) VALUES (?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET n=excluded.n, loss=excluded.loss, failures=excluded.failures",
            (key, n, loss, fails)
        )

    def adapt_pattern(self, key: str, regime: str, probability: float, is_number: bool = False) -> tuple[float, float]:
        row = self.c.execute("SELECT n, loss, failures FROM pattern_models WHERE name=?", (key,)).fetchone()
        if not row or float(row["n"]) < 2.0: return probability, 1.0
        n = float(row["n"])
        mean_loss = float(row["loss"]) / n
        fails = int(row["failures"])
        reliability = math.exp(-4.0 * max(-0.10, mean_loss - 0.25))
        if fails >= 2: reliability *= 0.2
        if fails >= 4: reliability *= 0.05
        if fails >= 3 and mean_loss > 0.26: reliability *= 0.01
        return clip(probability), max(0.05, min(2.0, reliability))

    # --- Beta reliability per signal ---
    def load_beta(self, name: str) -> BetaTracker:
        row = self.c.execute("SELECT payload FROM signal_betas WHERE name=?", (name,)).fetchone()
        if not row: return BetaTracker()
        return BetaTracker.from_dict(json.loads(row["payload"]))

    def save_beta(self, name: str, beta: BetaTracker) -> None:
        self.c.execute(
            "INSERT INTO signal_betas(name,payload) VALUES (?,?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
            (name, json.dumps(beta.to_dict()))
        )

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
                       (kind, issue, text, time.time(), expires))

    def calibration_summary(self) -> dict[str, Any]:
        """Brier score + reliability buckets over past settled predictions."""
        rows = self.c.execute(
            "SELECT predicted_p, status FROM predictions WHERE status IN ('WIN','LOSS') ORDER BY created DESC LIMIT 500"
        ).fetchall()
        if not rows: return {"n": 0}
        n = len(rows)
        brier = 0.0
        buckets = defaultdict(lambda: [0, 0])  # bucket -> [sum_pred, count]
        for r in rows:
            p = float(r["predicted_p"])
            won = 1.0 if r["status"] == "WIN" else 0.0
            brier += (p - won) ** 2
            b = round(p * 10) / 10
            buckets[b][0] += p; buckets[b][1] += 1
        return {
            "n": n,
            "brier": round(brier / n, 4),
            "buckets": {f"{b:.1f}": round(s / c, 3) for b, (s, c) in sorted(buckets.items())},
        }

    def health_stats(self) -> dict[str, Any]:
        settled = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0])
        wins = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0])
        num_hits = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE won_number=1").fetchone()[0])
        return {
            "history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]),
            "settled": settled,
            "side_wins": wins,
            "side_win_rate": round(100.0 * wins / settled, 2) if settled else None,
            "num_hits": num_hits,
            "num_hit_rate": round(100.0 * num_hits / settled, 2) if settled else None,
            "calibration": self.calibration_summary(),
        }

    def close(self) -> None: self.c.close()

# ============================================================
# ENGINE (Ensemble)
# ============================================================

class Engine:
    def __init__(self, db: Database):
        self.db = db

    def _collect_signals(self, seq: list[int], nums: list[int], regime: str) -> list[Signal]:
        sigs: list[Signal] = []
        for fn in (
            lambda: signal_family_pattern(seq, self.db, regime),
            lambda: signal_skipgram(seq, self.db),
            lambda: signal_trend(seq),
            lambda: signal_streak_break(seq),
            lambda: signal_cusum(seq),
            lambda: signal_number_skew(seq, nums),
        ):
            try:
                s = fn()
                if s: sigs.append(s)
            except Exception as exc:
                LOG.warning("Signal failed: %s", exc)
        return sigs

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(r["size"]) for r in history]
        nums = [int(r["num"]) for r in history]
        regime = get_regime(seq)
        entropy = entropy_bits(seq[-24:]) / 3.32

        signals = self._collect_signals(seq, nums, regime)

        # Load beta reliability per signal name, multiply into weights
        weighted_signals: list[Signal] = []
        for s in signals:
            beta = self.db.load_beta(s.name)
            # beta.mean ~ P(signal correct). Convert to multiplier around 1.0
            multiplier = 0.5 + 1.5 * beta.mean  # 0.5..2.0
            s.weight *= multiplier
            weighted_signals.append(s)

        # Aggregate ensemble vote
        total_vote = sum(s.vote for s in weighted_signals)
        total_weight = sum(s.weight for s in weighted_signals) or 1.0
        raw_p_big = sigmoid(total_vote / max(1.0, total_weight) * 4.0)

        # Entropy damping
        if entropy > 0.85:
            raw_p_big = 0.5 + (raw_p_big - 0.5) * 0.5

        # Calibration shrink (isotonic-style): shrink toward observed hit rate
        calib = self.db.calibration_summary()
        if calib.get("n", 0) >= 30:
            observed = 0.0
            for p_bucket, hit in calib.get("buckets", {}).items():
                if abs(float(p_bucket) - max(raw_p_big, 1 - raw_p_big)) < 0.15:
                    observed = float(hit); break
            if observed > 0:
                # shrink predicted confidence toward observed
                conf = max(raw_p_big, 1 - raw_p_big)
                conf = 0.7 * conf + 0.3 * observed
                raw_p_big = conf if raw_p_big >= 0.5 else 1 - conf

        side = "BIG" if raw_p_big >= 0.5 else "SMALL"
        side_p = raw_p_big if side == "BIG" else 1 - raw_p_big
        side_p = clip(side_p)

        # Number prediction
        num_dist, num_info = track_numbers(nums)
        allowed = range(5, 10) if side == "BIG" else range(0, 5)
        filtered = [0.0] * 10
        for k in range(10):
            filtered[k] = num_dist[k] * (1.6 if k in allowed else 0.08)
        tot = sum(filtered)
        if tot > 0: filtered = [x / tot for x in filtered]
        recent_counts = Counter(nums[-50:])
        num_pred = max(allowed, key=lambda k: (filtered[k], recent_counts.get(k, 0)))
        top3 = sorted(allowed, key=lambda k: filtered[k], reverse=True)[:3]

        # Pick dominant signal reason
        dominant = max(weighted_signals, key=lambda s: abs(s.vote)) if weighted_signals else None

        payload = {
            "issue": issue,
            "p": side_p,
            "side": side,
            "number": num_pred,
            "top3": top3,
            "num_conf": filtered[num_pred],
            "regime": regime,
            "entropy": round(entropy, 3),
            "signals": [
                {"n": s.name, "side": s.side, "str": round(s.strength, 3),
                 "w": round(s.weight, 3), "why": s.reason}
                for s in weighted_signals
            ],
            "vote_sum": round(total_vote, 3),
            "num_tracking": num_info.get("model", "—"),
            "override_reason": dominant.reason if dominant else "ensemble",
            "source": "V49 Apex Ensemble",
            "created": time.time(),
        }
        return payload

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        for row in self.db.pending_predictions():
            actual = lookup.get(row["issue"])
            if not actual: continue

            payload = json.loads(row["payload"])
            actual_size = int(actual["size"])
            actual_num = int(actual["num"])
            predicted_side = payload.get("side", "BIG")
            actual_side = "BIG" if actual_size == 1 else "SMALL"
            won_side = predicted_side == actual_side
            won_num = actual_num == payload.get("number")
            predicted_p = float(row["predicted_p"]) if row["predicted_p"] else float(payload.get("p", 0.5))

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=?, won_number=? WHERE issue=? AND status='PENDING'",
                    ("WIN" if won_side else "LOSS", actual_num, int(won_num), row["issue"])
                )
                # Update per-pattern model
                for s in payload.get("signals", []):
                    if s["n"].startswith("family") or s["n"].startswith("skipgram"):
                        key = s["n"]
                        self.db.learn_pattern(key, s["str"], actual_size)
                    # Update Beta reliability
                    beta = self.db.load_beta(s["n"])
                    beta.update(s["side"] == actual_side)
                    self.db.save_beta(s["n"], beta)

                sticker = (
                    CFG.jack_sticker if (won_side and won_num)
                    else CFG.win_sticker if won_side
                    else CFG.loss_sticker
                )
                self.db.queue("sticker", row["issue"], sticker)

# ============================================================
# FORMATTING
# ============================================================

def format_prediction(payload: dict[str, Any]) -> str:
    prob = float(payload["p"])
    conf = prob * 100
    filled = int(min(10, max(0, conf // 10)))
    bar = "█" * filled + "░" * (10 - filled)
    side = payload["side"]
    marker = "🔴" if side == "BIG" else "🔵"

    sigs = payload.get("signals", [])
    vote_big = sum(1 for s in sigs if s["side"] == "BIG")
    vote_small = len(sigs) - vote_big

    sig_lines = []
    for s in sigs[:5]:
        arrow = "🟢" if s["side"] == "BIG" else "🔵"
        sig_lines.append(
            f"  {arrow} <code>{escape(s['n'])}</code> "
            f"<i>{s['str']:.2f}×{s['w']:.2f}</i>"
        )
    sig_block = "\n".join(sig_lines) if sig_lines else "  <i>no signals</i>"

    n1 = int(payload["number"])
    top3 = payload.get("top3", [n1])
    nconf = payload.get("num_conf", 0) * 100

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"🔢 <b>TOP:</b> <code>{n1}</code> ({nconf:.1f}%) "
        f"· <i>alt:</i> <code>{top3[1] if len(top3)>1 else '—'}</code>, "
        f"<code>{top3[2] if len(top3)>2 else '—'}</code>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONF:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])} · ent={payload.get('entropy','?')}\n"
        f"🗳️ <b>VOTES:</b> BIG[{vote_big}] / SMALL[{vote_small}] · "
        f"Σ=<code>{payload.get('vote_sum', 0):+.2f}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧠 <b>SIGNALS:</b>\n{sig_block}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db: Database):
        self.db, self.engine = db, Engine(db)
        self.stop, self.last_fetch = asyncio.Event(), 0.0

    async def pause(self, sec: float) -> None:
        try: await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, sec))
        except asyncio.TimeoutError: pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def one(url: str):
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception:
                LOG.debug("fetch fail %s", url); return []
        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources: raise RuntimeError("HistoryUnavailable")
        merged = {row["issue"]: row for rows in sources for row in rows}
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        while not self.stop.is_set():
            try:
                self.db.ingest(await self.fetch(session))
                history = self.db.history()
                self.engine.settle(history)
                self.last_fetch = time.time()

                if len(history) >= CFG.min_history:
                    issue = str(int(history[-1]["issue"]) + 1)
                    if not self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone():
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions(issue,created,status,payload,predicted_p) VALUES (?,?,?,?,?)",
                                (issue, time.time(), "PENDING", json.dumps(pred), float(pred["p"]))
                            )
                            self.db.queue("prediction", issue, format_prediction(pred), time.time() + CFG.delivery_ttl)
                failures = 0
            except asyncio.CancelledError: raise
            except Exception:
                failures += 1
                LOG.exception("Worker error")
            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
            if not row:
                await self.pause(0.5); continue
            delay = backoff
            try:
                endpoint = "sendSticker" if row["kind"] == "sticker" else "sendMessage"
                url = f"https://api.telegram.org/bot{CFG.token}/{endpoint}"
                payload = (
                    {"chat_id": CFG.chat, "sticker": row["text"]}
                    if row["kind"] == "sticker"
                    else {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}
                )
                async with session.post(url, json=payload) as res:
                    data = await res.json(content_type=None)
                    if res.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0; continue
                    if data.get("error_code") == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif data.get("error_code") in (400, 401, 403, 404):
                        delay = 60
            except Exception:
                delay = 10
            await self.pause(delay)
            backoff = min(60.0, backoff * 2)

    async def health(self, request: web.Request) -> web.Response:
        status = "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded"
        return web.json_response({"status": status, **self.db.health_stats()})

# ============================================================
# ENTRY POINT
# ============================================================

async def main() -> None:
    CFG.validate()
    db = Database(CFG.db_path)
    runtime = Runtime(db)

    app = web.Application()
    app.router.add_get("/", runtime.health)
    app.router.add_get("/healthz", runtime.health)
    runner = web.AppRunner(app)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop.set)

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session)),
                asyncio.create_task(runtime.sender(session)),
            ]
            await asyncio.gather(*tasks)
    finally:
        await runner.cleanup()
        db.close()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass

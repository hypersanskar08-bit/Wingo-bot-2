# bot.py — FTW VIP PREDICTOR (HYPER V17 — Statistics-First Engine)
# Python 3.10+  |  Deps: aiohttp

import asyncio, contextlib, html, json, logging, math, os, random, signal, socket, sqlite3, time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import aiohttp
from aiohttp import web

# ─── CONFIG ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    token: str = "5837138236:AAFMJPMY8SeXu0CIF-fyywY5M_u3aQm-HiI"
    chat: str = "-1004472318983"
    urls: tuple = ("https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",)
    db_path: str = os.getenv("DB_PATH", "predictor_hyper_v19.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 4000
    min_history: int = int(os.getenv("MIN_HISTORY", "120"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    learning_rate: float = 0.035
    meta_lr: float = 0.03
    decay: float = 0.985
    max_pattern: int = 15
    delivery_ttl: float = 55.0
    name: str = "FTW VIP PREDICTOR (HYPER)"
    # ✅ SKIP LOGIC — this is the single biggest win-rate booster
# 0.0 = predict every round. 0.06 = skip if confidence < 50%. 0.10 = only 50%+
min_edge: float = float(os.getenv("MIN_EDGE", "0.0")) 
# 👆 0.05 ki jagah 0.0 kar do
    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"
    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config: token, chat, or urls")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor")

# ─── UTILITIES ──────────────────────────────────────────────
def clip(x, low=0.02, high=0.98): return max(low, min(high, float(x)))
def logit(p, eps=0.02):
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))
def sigmoid(z):
    z = max(-20.0, min(20.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))
def escape(x): return html.escape(str(x))

def run_length(seq):
    if not seq: return 0
    n = 0
    for x in reversed(seq):
        if x != seq[-1]: break
        n += 1
    return n

def get_regime(seq):
    tail = seq[-24:]
    if len(tail) < 8: return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    rate = flips / (len(tail) - 1)
    if run_length(tail) >= 4: return "RUN"
    if rate >= 0.70: return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20: return "IMBAL"
    return "MIXED"

def numeric(value):
    if isinstance(value, bool): return None
    text = str(value).strip()
    return int(text) if text.isascii() and text.isdigit() else None

def parse_history(payload):
    obj = payload
    for _ in range(6):
        if isinstance(obj, list): break
        if not isinstance(obj, dict): return []
        obj = next((obj[k] for k in ("data", "list", "result", "items", "history")
                    if isinstance(obj.get(k), (list, dict))), None)
    if not isinstance(obj, list): return []
    records = {}
    for item in obj:
        if not isinstance(item, dict): continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))
        if issue is None or number is None or not 0 <= number <= 9: continue
        key = str(issue)
        if key in records and records[key]["num"] != number:
            raise ValueError(f"Conflicting duplicate: {key}")
        records[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda x: int(x["issue"]))

def weighted_probability(observations):
    sw = sum(w for _, w in observations)
    sw2 = sum(w * w for _, w in observations)
    effective_n = sw * sw / sw2 if sw2 else 0.0
    probability = (4 + sum(y * w for y, w in observations)) / (8 + sw)
    return probability, effective_n

def trend_features(seq):
    if not seq: return [0.0] * 32
    def mean(k):
        arr = seq[-k:]; return sum(arr) / len(arr) if arr else 0.5
    def ema(k):
        if not seq: return 0.5
        alpha = 2.0 / (k + 1); val = float(seq[0])
        for x in seq[1:]: val = alpha * x + (1 - alpha) * val
        return val
    def rsi(period=14):
        if len(seq) < period + 1: return 50.0
        gains = losses = 0.0
        for i in range(-period, 0):
            change = seq[i] - seq[i - 1]
            if change > 0: gains += change
            else: losses -= change
        if losses == 0: return 100.0
        return 100.0 - (100.0 / (1.0 + gains / losses))
    def std(k):
        arr = seq[-k:]
        if len(arr) < 2: return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))
    tail = seq[-24:]
    alternation = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    signed_run = (2 * seq[-1] - 1) * min(run_length(seq), 10) / 10.0
    rsi_n = (rsi(14) / 100.0) * 2.0 - 1.0
    macd = (ema(12) - ema(26)) * 2.0
    autocorr = 0.0
    if len(seq) >= 25 and std(24) > 0:
        m24 = mean(24)
        num = sum((seq[i] - m24) * (seq[i - 1] - m24) for i in range(-23, 0))
        den = 24.0 * std(24) ** 2
        autocorr = num / den if den > 0 else 0.0
    return [1.0, 2*mean(3)-1, 2*mean(5)-1, 2*mean(10)-1, 2*mean(24)-1, 2*mean(48)-1, 2*mean(96)-1,
            mean(3)-mean(10), mean(5)-mean(24), mean(10)-mean(48), signed_run, 2*seq[-1]-1,
            (2*alternation-1)*(2*seq[-1]-1), rsi_n, macd, (2*std(10)-1), (2*std(24)-1),
            -1 - sum(x * math.log2(x) for x in [sum(seq[-24:])/24, 1-(sum(seq[-24:])/24)] if x > 0),
            -1 - sum(x * math.log2(x) for x in [sum(seq[-48:])/48, 1-(sum(seq[-48:])/48)] if x > 0),
            autocorr, 1.0 if mean(5)>0.5 else -1.0,
            (2*seq[-1]-1)*(2*mean(24)-1), abs(mean(24)-0.5)*2, (2*seq[-1]-1)*(mean(10)-mean(48)),
            (2*mean(10)-1)/(std(10)+0.1), (2*mean(24)-1)/(std(24)+0.1), (sum(seq[-10:])/10.0)-0.5,
            (sum(seq[-5:])/5.0)-0.5, (2*mean(5)-1)*(2*mean(24)-1), math.tanh((mean(10)-0.5)*6),
            math.tanh((mean(48)-0.5)*6), (2*mean(3)-1)*(2*mean(96)-1)]

EXPERT_NAMES = ["patterns", "markov1", "markov2", "markov3", "markov4",
                "run", "regime", "base", "neighbors", "trend", "momentum", "cyclic"]
def logistic(weights, features): return sigmoid(sum(w * x for w, x in zip(weights, features)))

# ─── DATABASE (critical: adapt() rewritten) ──────────────────
class Database:
    def __init__(self, path):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS history(issue TEXT PRIMARY KEY, num INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(issue TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, actual INTEGER);
            CREATE TABLE IF NOT EXISTS models(key TEXT PRIMARY KEY, n REAL NOT NULL, loss REAL NOT NULL, inverse_loss REAL NOT NULL, failures INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, issue TEXT, text TEXT NOT NULL, created REAL NOT NULL, expires REAL, sent INTEGER NOT NULL DEFAULT 0);
        """)

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield; self.c.execute("COMMIT")
        except BaseException:
            self.c.execute("ROLLBACK"); raise

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.c.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, json.dumps(value)))

    def ingest(self, records):
        with self.transaction():
            for r in records:
                old = self.c.execute("SELECT num FROM history WHERE issue=?", (r["issue"],)).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute("INSERT OR IGNORE INTO history VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute("DELETE FROM history WHERE issue NOT IN "
                            "(SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                            (CFG.max_history,))

    def history(self):
        rows = self.c.execute("SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER)").fetchall()
        return [{"issue": r["issue"], "num": r["num"], "size": int(r["num"] >= 5)} for r in rows]

    def memory(self, key):
        row = self.c.execute("SELECT * FROM models WHERE key=?", (key,)).fetchone()
        return dict(row) if row else {"n": 0.0, "loss": 0.0, "inverse_loss": 0.0, "failures": 0}

    def learn(self, key, probability, actual):
        m = self.memory(key)
        n = CFG.decay * m["n"] + 1
        loss = CFG.decay * m["loss"] + (probability - actual) ** 2
        inv = CFG.decay * m["inverse_loss"] + (1 - probability - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        failures = m["failures"] + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n,loss=excluded.loss,inverse_loss=excluded.inverse_loss,failures=excluded.failures",
            (key, n, loss, inv, failures))

    # ✅ NEW STATISTICAL ADAPT — replaces the aggressive 2-fail flip
    def adapt(self, key, probability):
        m = self.memory(key)
        n = m["n"]
        if n < 1:
            return probability, 0.65, False

        brier = m["loss"] / n
        inv = m["inverse_loss"] / n

        # Flip ONLY if:
        #   1. Inverse is meaningfully better (0.02 Brier margin)
        #   2. Model is actually performing worse than random (brier > 0.24)
        #   3. At least 3 consecutive failures (reduces false positives)
        flipped = (inv + 0.02 < brier) and (brier > 0.24) and (m["failures"] >= 3)
        effective_loss = inv if flipped else brier
        p = (1 - probability) if flipped else probability

        # Reliability: how good is Brier vs random 0.25
        reliability = clip((0.25 - effective_loss) / 0.10, 0.0, 1.0)
        maturity = min(n / 40.0, 1.0)
        # Immature: heavy shrink. Mature: rely on reliability
        shrink = (1 - maturity) * 0.65 + maturity * reliability
        p = 0.5 + (p - 0.5) * shrink

        weight = clip(math.exp(-6 * (effective_loss - 0.22)), 0.20, 2.0)
        if m["failures"] >= 4 and not flipped:
            weight *= 0.5
        return clip(p), weight, flipped

    def queue(self, kind, issue, text, expires=None):
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
                       (kind, issue, text, time.time(), expires))

# ─── BASE ENGINE (mostly unchanged; number prediction improved) ──
class Engine:
    def __init__(self, db): self.db = db

    def get_runs(self, seq):
        if not seq: return []
        runs = []; curr = seq[0]; length = 1
        for x in seq[1:]:
            if x == curr: length += 1
            else:
                runs.append((curr, length)); curr = x; length = 1
        runs.append((curr, length))
        return runs

    def mine_family_patterns(self, seq):
        if not seq: return []
        runs = self.get_runs(seq)
        if len(runs) < 2: return []
        for k in range(min(6, len(runs)), 1, -1):
            target = runs[-k:]; observations = []
            for i in range(len(runs) - k - 1, -1, -1):
                if runs[i:i+k] == target and i + k < len(runs):
                    observations.append((runs[i+k][0], 1.0))
            if observations:
                p, eff_n = weighted_probability(observations)
                bits = "-".join(f"{'B' if v else 'S'}{c}" for v, c in target)
                key = f"family:{bits}"
                rank = abs(p - 0.5) * math.sqrt(eff_n) + 10.0
                return [(rank, p, len(observations), eff_n, key)]
        return []

    def mine_patterns(self, seq):
        n = len(seq)
        fam = self.mine_family_patterns(seq)
        if fam: return fam
        candidates = {}
        def weight(idx): return 0.5 ** ((n - 1 - idx) / 300.0)
        for length in range(5, CFG.max_pattern + 1):
            if n < length * 2: continue
            suffix = tuple(seq[-length:])
            obs, last = [], -length - 1
            for i in range(length, n - length):
                if tuple(seq[i - length:i]) == suffix and i - last >= length + 1:
                    obs.append((seq[i], weight(i))); last = i
            if len(obs) >= CFG.min_support:
                p, eff_n = weighted_probability(obs)
                if eff_n >= CFG.min_support * 0.5:
                    bits = "".join("B" if x else "S" for x in suffix)
                    candidates[f"exact:{bits}"] = (abs(p - 0.5) * math.sqrt(eff_n), p, len(obs), eff_n, f"exact:{bits}")
        return sorted(candidates.values(), key=lambda x: x[0], reverse=True)[:8]

    def mine_experts(self, seq):
        n = len(seq); experts = {}
        def weight(idx): return 0.5 ** ((n - 1 - idx) / 300.0)
        patterns = self.mine_patterns(seq)
        if patterns:
            num = den = 0.0
            for _, p, _, eff_n, key in patterns:
                adjusted, w, _ = self.db.adapt(key, p)
                num += adjusted * (w * math.sqrt(eff_n)); den += (w * math.sqrt(eff_n))
            if den > 0: experts["patterns"] = num / den
        for order in (1, 2, 3, 4):
            suffix = tuple(seq[-order:])
            obs = [(seq[i], weight(i)) for i in range(order, n - order) if tuple(seq[i - order:i]) == suffix]
            if len(obs) >= CFG.min_support: experts[f"markov{order}"] = weighted_probability(obs)[0]
        if n:
            run_at = [1] * n
            for i in range(1, n):
                run_at[i] = run_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
            current_run = min(run_at[-1], 8); last_val = seq[-1]
            run_obs = [(seq[i], weight(i)) for i in range(24, n)
                       if seq[i - 1] == last_val and min(run_at[i - 1], 8) == current_run]
            if len(run_obs) >= CFG.min_support:
                experts["run"] = weighted_probability(run_obs)[0]
        current_regime = get_regime(seq)
        regime_obs = [(seq[i], weight(i)) for i in range(24, n) if get_regime(seq[max(0, i-96):i]) == current_regime]
        if len(regime_obs) >= CFG.min_support: experts["regime"] = weighted_probability(regime_obs)[0]
        experts["base"] = weighted_probability([(y, weight(i)) for i, y in enumerate(seq)])[0]
        nearest = [(seq[i], weight(i) * 0.4 ** sum(a != b for a, b in zip(seq[i-10:i], seq[-10:])))
                   for i in range(10, n - 10)
                   if sum(a != b for a, b in zip(seq[i-10:i], seq[-10:])) <= 2]
        if len(nearest) >= 20: experts["neighbors"] = weighted_probability(nearest)[0]
        features = trend_features(seq)
        tw = self.db.get("trend_weights")
        if not isinstance(tw, list) or len(tw) != len(features): tw = [0.0] * len(features)
        experts["trend"] = logistic(tw, features)
        momentum = sum(seq[-5:]) / 5.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)
        return experts, patterns, features, current_regime

    # ✅ FIXED: Bayesian frequency fallback (no more random.randint)
    def _predict_number(self, seq_nums):
        ngram_db = self.db.get("num_ngrams", {})
        freq = self.db.get("num_freq", {})
        prior = [1.0] * 10
        if freq:
            total = sum(freq.values())
            if total > 0:
                prior = [4.0 + 20.0 * freq.get(str(i), 0) / total for i in range(10)]
        dist = list(prior)
        for length in range(2, min(10, len(seq_nums)) + 1):
            seq_key = ",".join(str(x) for x in seq_nums[-length:])
            if seq_key in ngram_db:
                trans = ngram_db[seq_key]
                total = sum(trans.values())
                if total < 3: continue
                w = min(1.0, total / 20.0) * (0.9 ** (length - 2))
                for i in range(10):
                    dist[i] = dist[i] * (1 - w) + (trans.get(str(i), 0) / total) * w
        if sum(dist) <= 0: return 4, 0.10
        best = max(range(10), key=lambda i: dist[i])
        total = sum(dist)
        return best, dist[best] / total

    def predict(self, history, issue):
        seq = [r["size"] for r in history]; nums = [r["num"] for r in history]
        experts, patterns, features, current_regime = self.mine_experts(seq)
        current_streak = run_length(seq)
        adjusted = []
        for name, p in experts.items():
            corrected, weight, _ = self.db.adapt(f"expert:{name}:{current_regime}", p)
            if name == "patterns": weight *= 2.5
            if current_streak >= 4 and name in ("momentum", "run"): weight *= 3.0
            if name == "trend" and current_regime in ("RUN", "IMBAL"): weight *= 1.4
            adjusted.append((corrected, weight))
        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5
        # ✅ Removed: consensus shrink (was a bug)
        m_feats = [logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES] + [1.0]
        mw = self.db.get("meta_weights")
        if not isinstance(mw, list) or len(mw) != len(m_feats):
            mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
        p_meta = sigmoid(sum(w * f for w, f in zip(mw, m_feats)))
        alpha = min(self.db.get("meta_n", 0) / 80.0, 1.0)
        p_raw = alpha * p_meta + (1 - alpha) * p_avg
        if current_streak >= 4:
            p_raw = 0.90 * float(seq[-1]) + 0.10 * p_raw
        bucket = min(19, int(p_raw * 20))
        n_g, s_g = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((30 * p_raw + s_g) / (30 + n_g))
        # ✅ REMOVED pattern pressure (double-counting bug)
        num_pred, num_conf = self._predict_number(nums)
        size_conf = max(p_cal, 1 - p_cal)
        source = "📊 Size Engine"
        if num_conf > size_conf + 0.05:
            source = "🎯 Number Engine"
            if num_pred >= 5: p_cal = max(p_cal, num_conf)
            else: p_cal = min(p_cal, 1 - num_conf)
            pred_side = "BIG" if num_pred >= 5 else "SMALL"
        else:
            pred_side = "BIG" if p_cal >= 0.5 else "SMALL"
            valid = list(range(5, 10)) if pred_side == "BIG" else list(range(0, 5))
            if num_pred not in valid:
                ngram_db = self.db.get("num_ngrams", {})
                for length in range(min(10, len(nums)), 1, -1):
                    seq_key = ",".join(str(x) for x in nums[-length:])
                    if seq_key in ngram_db:
                        trans = ngram_db[seq_key]
                        valid_trans = {int(k): v for k, v in trans.items() if int(k) in valid}
                        if valid_trans:
                            num_pred = max(valid_trans, key=valid_trans.get); break
                else:
                    num_pred = 4 if pred_side == "SMALL" else 5  # deterministic fallback
        prev_regime = get_regime(seq[:-1]) if len(seq) > 1 else "N/A"
        trend_shift = f"{prev_regime} ➡️ {current_regime}" if prev_regime != current_regime else None
        return {
            "issue": issue, "p": p_cal, "p_raw": p_raw, "side": pred_side,
            "num": num_pred, "regime": current_regime, "source": source,
            "trend_shift": trend_shift,
            "pattern_str": _pattern_str(patterns),
            "experts": experts, "features": features, "meta_feats": m_feats,
            "patterns": [{"key": k, "p": p, "support": s} for r, p, s, e, k in patterns]
        }

    def settle(self, history):
        lookup = {r["issue"]: r for r in history}
        if history:
            sig = f"{history[0]['issue']}-{history[-1]['issue']}-{len(history)}"
            if self.db.get("ngram_sig") != sig:
                num_ngrams, num_freq = {}, {}
                for i in range(1, len(history)):
                    curr = history[i]["num"]
                    num_freq[str(curr)] = num_freq.get(str(curr), 0) + 1
                    for length in range(2, 11):
                        if i >= length:
                            seq_key = ",".join(str(history[j]["num"]) for j in range(i - length, i))
                            num_ngrams.setdefault(seq_key, {})
                            num_ngrams[seq_key][str(curr)] = num_ngrams[seq_key].get(str(curr), 0) + 1
                self.db.set("num_ngrams", num_ngrams); self.db.set("num_freq", num_freq); self.db.set("ngram_sig", sig)
        pending = self.db.c.execute(
            "SELECT issue,payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)").fetchall()
        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None: continue
            p = json.loads(row["payload"]); y = actual["size"]
            won_side = int(p["p"] >= 0.5) == y
            won_num = p.get("num") == actual["num"]
            with self.db.transaction():
                self.db.c.execute("UPDATE predictions SET status=?,actual=? WHERE issue=?",
                                  ("WIN" if won_side else "LOSS", actual["num"], row["issue"]))
                for name, val in p["experts"].items():
                    self.db.learn(f"expert:{name}:{p['regime']}", val, y)
                for pat in p["patterns"]: self.db.learn(pat["key"], pat["p"], y)
                if p.get("features") and p["experts"].get("trend"):
                    err = y - p["experts"]["trend"]
                    tw = self.db.get("trend_weights") or [0.0] * len(p["features"])
                    self.db.set("trend_weights",
                                [clip(w + CFG.learning_rate * (err * x - 0.01 * w), -4, 4)
                                 for w, x in zip(tw, p["features"])])
                target = len(EXPERT_NAMES) + 2
                if p.get("meta_feats"):
                    mw = self.db.get("meta_weights")
                    if not isinstance(mw, list) or len(mw) != target:
                        if isinstance(mw, list) and len(mw) == target - 1:
                            mw = mw[:-1] + [0.0, mw[-1]]
                        else:
                            mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
                    feats = p["meta_feats"]
                    if len(feats) == target - 1: feats = feats[:-1] + [0.0, feats[-1]]
                    if len(feats) == target:
                        mp = sigmoid(sum(w * f for w, f in zip(mw, feats)))
                        grad = 2.0 * (y - mp) * mp * (1.0 - mp)
                        self.db.set("meta_weights",
                                    [clip(w * 0.999 + CFG.meta_lr * grad * f, -4, 4) for w, f in zip(mw, feats)])
                        self.db.set("meta_n", self.db.get("meta_n", 0) + 1)
                bucket = min(19, int(p.get("p_raw", p["p"]) * 20))
                n, s = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
                self.db.set(f"cal20:global:{bucket}", [0.995 * n + 1, 0.995 * s + y])
                if won_side and won_num: sticker = CFG.jack_sticker
                elif won_side: sticker = CFG.win_sticker
                else: sticker = CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)
                self.db.c.execute("UPDATE outbox SET sent=2 WHERE kind='prediction' AND issue=? AND sent=0", (row["issue"],))


def _pattern_str(patterns):
    if not patterns: return "None"
    top = patterns[0]; p_type = top[4].split(":")[0]; bits = top[4].split(":")[-1]
    arrow = "⬆️" if top[1] >= 0.5 else "⬇️"
    return f"👨‍👩‍👧‍👦 <code>{bits}</code> {arrow}" if p_type == "family" else f"<code>{bits}</code> {arrow}"


# ─── UPGRADED ENGINE (V17) ──────────────────────────────────
class UpgradedEngine(Engine):
    @staticmethod
    def normalize(values):
        total = sum(values)
        if total <= 0: return [1.0 / len(values)] * len(values)
        return [v / total for v in values]

    def get_number_distribution(self, nums):
        ngram_db = self.db.get("num_ngrams", {})
        dist = [1.0] * 10
        for length in range(min(10, len(nums)), 1, -1):
            seq_key = ",".join(str(x) for x in nums[-length:])
            if seq_key in ngram_db:
                trans = ngram_db[seq_key]; total = sum(trans.values())
                if total >= 5:
                    for k, v in trans.items(): dist[int(k)] += (v / total) * length
                    break
        return self.normalize(dist)

    def calculate_change_score(self, seq):
        if len(seq) < 20: return 0.0
        fast = sum(seq[-5:]) / 5.0; slow = sum(seq[-20:]) / 20.0
        imbalance = abs(fast - slow) * 2.0
        recent = seq[-10:]
        switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]) / 9.0
        return clip((imbalance * 0.6) + (switches * 0.4), 0.0, 1.0)

    @staticmethod
    def _expected_meta_len(): return len(EXPERT_NAMES) + 2

    def _load_meta_weights(self):
        # ✅ FIXED: sensible init instead of all zeros
        target = self._expected_meta_len()
        mw = self.db.get("meta_weights")
        if isinstance(mw, list) and len(mw) == target: return mw
        if isinstance(mw, list) and len(mw) == target - 1:
            return mw[:-1] + [0.0, mw[-1]]
        return [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [1.0 / len(EXPERT_NAMES), 0.0]

    def _streak_probability(self, seq):
        """Laplace-smoothed historical continuation probability."""
        if not seq: return 0.5
        run_at = [1] * len(seq)
        for i in range(1, len(seq)):
            run_at[i] = run_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
        cont_wins = cont_total = 0
        for i in range(4, len(seq)):
            if run_at[i - 1] >= 4:
                cont_total += 1
                if seq[i] == seq[i - 1]: cont_wins += 1
        # Laplace smoothing
        hist_cont = (cont_wins + 1) / (cont_total + 2)
        # Weight based on sample size
        weight = min(1.0, cont_total / 30.0)
        # Convert to BIG probability given last value
        p_big = hist_cont if seq[-1] == 1 else (1.0 - hist_cont)
        return weight * p_big + (1.0 - weight) * 0.5

    def predict(self, history, issue):
        seq = [r["size"] for r in history]; nums = [r["num"] for r in history]
        current_streak = run_length(seq)

        experts, patterns, features, current_regime = self.mine_experts(seq)
        num_dist = self.get_number_distribution(nums)
        big_prob = sum(num_dist[5:])
        experts["ngram_num"] = big_prob

        adjusted = []
        for name, p in experts.items():
            corrected, weight, _ = self.db.adapt(f"expert:{name}:{current_regime}", p)
            if name == "patterns": weight *= 2.5
            if name == "ngram_num": weight *= 2.0
            if current_streak >= 4 and name in ("momentum", "run"): weight *= 3.0
            if name == "trend" and current_regime in ("RUN", "IMBAL"): weight *= 1.4
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5

        # ✅ REMOVED: consensus shrink — this was killing confidence exactly when it should be highest

        m_feats = ([logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES]
                   + [logit(clip(big_prob, 0.05, 0.95)), 1.0])
        mw = self._load_meta_weights()
        p_meta = sigmoid(sum(w * f for w, f in zip(mw, m_feats)))
        alpha = min(self.db.get("meta_n", 0) / 80.0, 1.0)
        p_raw = alpha * p_meta + (1 - alpha) * p_avg

        # ✅ FIXED: Laplace-smoothed streak continuation
        if current_streak >= 4:
            p_streak = self._streak_probability(seq)
            p_raw = 0.6 * p_streak + 0.4 * p_raw

        bucket = min(19, int(p_raw * 20))
        n_g, s_g = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((30 * p_raw + s_g) / (30 + n_g))

        # ✅ REMOVED pattern pressure — pattern is already an expert

        pred_side = "BIG" if p_cal >= 0.5 else "SMALL"
        valid_nums = range(5, 10) if pred_side == "BIG" else range(0, 5)
        best_num = max(valid_nums, key=lambda x: num_dist[x])
        num_prob = num_dist[best_num]
        change_score = self.calculate_change_score(seq)

        prev_regime = get_regime(seq[:-1]) if len(seq) > 1 else "N/A"
        trend_shift = f"{prev_regime} ➡️ {current_regime}" if prev_regime != current_regime else None

        # ✅ Edge calculation for skip/confidence display
        edge = abs(p_cal - 0.5)
        confidence = max(p_cal, 1 - p_cal)

        # ✅ Optional skip — the single biggest win-rate booster
        if CFG.min_edge > 0 and edge < CFG.min_edge:
            return None  # Skip prediction

        return {
            "issue": issue, "p": p_cal, "p_raw": p_raw, "side": pred_side,
            "num": best_num, "num_probability": num_prob, "change_score": change_score,
            "regime": current_regime, "source": "🎯 Meta+Evidence",
            "trend_shift": trend_shift, "pattern_str": _pattern_str(patterns),
            "edge": edge, "confidence": confidence,
            "experts": experts, "features": features, "meta_feats": m_feats,
            "patterns": [{"key": k, "p": p, "support": s} for r, p, s, e, k in patterns]
        }


# ─── FORMATTER ──────────────────────────────────────────────
def format_prediction(p):
    conf = max(p["p"], 1 - p["p"]) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if p["side"] == "BIG" else "SMALL"
    color = "🔴" if side == "BIG" else "🔵"
    quality = "🔥 HIGH VIP" if conf >= 65 else "⭐ VIP" if conf >= 60 else "✅ NORMAL"
    stake = "1.5x" if conf >= 65 else "1x" if conf >= 60 else "0.5x"
    shift_text = f"\n🔄 <b>TREND SHIFT:</b> {escape(p['trend_shift'])}" if p.get("trend_shift") else ""
    pat_text = f"\n🧩 <b>PATTERN:</b> {p.get('pattern_str', 'None')}" if p.get("pattern_str") and p.get("pattern_str") != "None" else ""
    source_text = f"\n🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}"
    # ✅ Show edge/confidence
    edge_text = f"\n📐 <b>EDGE:</b> {p.get('edge', 0)*100:.1f}% | <b>CI:</b> {conf:.1f}%" if p.get("edge") else ""
    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {color} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake}){edge_text}\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime', 'N/A'))}{source_text}{shift_text}{pat_text}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ─── RUNTIME (with skip handling) ──────────────────────────
class Runtime:
    def __init__(self, db):
        self.db = db
        self.engine = UpgradedEngine(db)
        self.stop = asyncio.Event()
        self.last_fetch = 0.0
        self.error = None

    async def pause(self, seconds):
        try: await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, seconds))
        except asyncio.TimeoutError: pass

    async def fetch(self, session):
        async def one(url):
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception: return []
        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources: raise RuntimeError("HistoryUnavailable")
        merged = {}
        for rows in sources:
            for row in rows:
                if row["issue"] in merged and merged[row["issue"]]["num"] != row["num"]:
                    raise ValueError("SourceConflict")
                merged[row["issue"]] = row
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session):
        failures = 0
        while not self.stop.is_set():
            try:
                records = await self.fetch(session)
                self.db.ingest(records)
                history = self.db.history()
                self.last_fetch = time.time()
                self.error = None; failures = 0
                self.engine.settle(history)
                latest = int(history[-1]["issue"])
                self.db.set("latest_seen", latest)
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    exists = self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone()
                    if not exists:
                        # ✅ Handle skip
                        p = self.engine.predict(history, issue)
                        with self.db.transaction():
                            if p is None:
                                LOG.info(f"Skipping issue {issue} (edge too low)")
                                self.db.c.execute(
                                    "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                    (issue, time.time(), "SKIP", json.dumps({"skipped": True})))
                            else:
                                self.db.c.execute("INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                                  (issue, time.time(), "PENDING", json.dumps(p)))
                                self.db.queue("prediction", issue, format_prediction(p),
                                              time.time() + CFG.delivery_ttl)
            except asyncio.CancelledError: raise
            except Exception as exc:
                self.error = type(exc).__name__; failures += 1
                LOG.error("Worker failure: %s", self.error)
            await self.pause(min(60, CFG.fetch_every * 2 ** min(failures, 4)))

    async def sender(self, session):
        endpoint_msg = f"https://api.telegram.org/bot{CFG.token}/sendMessage"
        endpoint_sticker = f"https://api.telegram.org/bot{CFG.token}/sendSticker"
        backoff = 1.0
        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
            if not row:
                await self.pause(0.5); continue
            delay = backoff
            try:
                if row["kind"] == "sticker":
                    url = endpoint_sticker; payload = {"chat_id": CFG.chat, "sticker": row["text"]}
                else:
                    url = endpoint_msg; payload = {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}
                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)
                    if response.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0; continue
                    code = data.get("error_code", response.status)
                    if code == 429: delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404):
                        delay = 60
                        if code == 400 and row["kind"] == "sticker":
                            LOG.error(f"Invalid Sticker ID: {row['text']}")
                            self.db.c.execute("UPDATE outbox SET sent=2 WHERE id=?", (row["id"],)); continue
            except asyncio.CancelledError: raise
            except Exception as e: LOG.warning(f"Telegram fail: {e}")
            await self.pause(delay); backoff = min(60.0, backoff * 2)

    async def health(self, request):
        total = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        wins = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0]
        skips = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='SKIP'").fetchone()[0]
        acc = (wins / total * 100) if total else 0.0
        return web.json_response({
            "status": "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded",
            "history": self.db.c.execute("SELECT COUNT(*) FROM history").fetchone()[0],
            "settled": total, "wins": wins, "win_rate": f"{acc:.1f}%", "skipped": skips,
        })


# ─── MAIN ──────────────────────────────────────────────────
async def main():
    CFG.validate()
    db = Database(CFG.db_path)
    runtime = Runtime(db)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop.set)
    app = web.Application(); app.router.add_get("/", runtime.health)
    runner = web.AppRunner(app); tasks = []; stop_task = None
    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300, ssl=False, family=socket.AF_INET)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [asyncio.create_task(runtime.worker(session), name="worker"),
                     asyncio.create_task(runtime.sender(session), name="sender")]
            stop_task = asyncio.create_task(runtime.stop.wait())
            LOG.info(f"{CFG.name} V17 Started (Statistics-First Engine, min_edge={CFG.min_edge})")
            await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
            runtime.stop.set()
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc: raise exc
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        runtime.stop.set()
        for task in tasks: task.cancel()
        if stop_task: stop_task.cancel()
        await asyncio.gather(*tasks, *([stop_task] if stop_task else []), return_exceptions=True)
        await runner.cleanup(); db.c.close()

if __name__ == "__main__":
    asyncio.run(main())

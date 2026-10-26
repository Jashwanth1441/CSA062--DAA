"""
CO1 Assignment 1 - Student Record Retrieval
Three search strategies for a Registration Number:
  A. Linear (sequential) search      - iterative, brute force, no preprocessing
  B. Binary search (recursive)       - decrease-and-conquer, needs sorted data
  C. Hash table with chaining        - space-time trade-off (hashing), needs a built index
Run:  python3 student_search.py
"""
import random, time, json, math, sys

# ----------------------------------------------------------------------------
# A. Linear search (iterative).  Basic operation: key comparison  a[i] == key
# ----------------------------------------------------------------------------
def linear_search(a, key):
    comps = 0
    for i in range(len(a)):
        comps += 1
        if a[i] == key:
            return i, comps
    return -1, comps

# ----------------------------------------------------------------------------
# B. Binary search (recursive). Basic operation: key comparison with a[mid]
#    (one three-way comparison per probe)
# ----------------------------------------------------------------------------
def binary_search_rec(a, key, lo, hi):
    """Search sorted list a[lo..hi] (inclusive). Returns (index, comparisons)."""
    if lo > hi:                       # base case: empty sub-array
        return -1, 0
    mid = (lo + hi) // 2
    if a[mid] == key:                 # 1 comparison per call
        return mid, 1
    if key < a[mid]:
        idx, c = binary_search_rec(a, key, lo, mid - 1)
    else:
        idx, c = binary_search_rec(a, key, mid + 1, hi)
    return idx, c + 1

def binary_search(a, key):
    return binary_search_rec(a, key, 0, len(a) - 1)

# ----------------------------------------------------------------------------
# C. Hash table with separate chaining.
#    Basic operation: key comparison inside the bucket chain
# ----------------------------------------------------------------------------
class HashIndex:
    def __init__(self, keys, load=0.75):
        self.m = int(len(keys) / load) | 1          # table size (odd)
        self.t = [None] * self.m
        for pos, k in enumerate(keys):              # build: n insertions, O(1) each
            h = k % self.m
            if self.t[h] is None:
                self.t[h] = [(k, pos)]
            else:
                self.t[h].append((k, pos))

    def insert(self, k, pos):
        h = k % self.m
        if self.t[h] is None:
            self.t[h] = [(k, pos)]
        else:
            self.t[h].append((k, pos))

    def search(self, key):
        comps = 0
        chain = self.t[key % self.m]
        if chain is not None:
            for k, pos in chain:
                comps += 1
                if k == key:
                    return pos, comps
        return -1, comps

# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
def correctness_tests():
    print("=" * 70)
    print("CORRECTNESS TESTS (all three algorithms vs. Python reference)")
    print("=" * 70)
    cases = {
        "empty database":          [],
        "single record (present)": [4021],
        "two records":             [7, 3],
        "small database":          [45, 12, 99, 7, 63, 21, 88],
        "1000 random records":     random.Random(1).sample(range(10**7), 1000),
    }
    total = 0
    for name, data in cases.items():
        srt = sorted(data)
        hx = HashIndex(data)
        probes = list(data)[:3] + list(data)[-3:] + [-5, 0, 10**9, 5]
        if data:
            probes += [min(data) - 1, max(data) + 1]
        ok = True
        for q in probes:
            ref = q in set(data)
            ia, _ = linear_search(data, q)
            ib, _ = binary_search(srt, q)
            ic, _ = hx.search(q)
            r = ((ia != -1) == ref) and ((ib != -1) == ref) and ((ic != -1) == ref)
            if ref:
                r = r and data[ia] == q and srt[ib] == q and data[ic] == q
            ok &= r
            total += 1
        print(f"  {name:28s} probes={len(probes):2d}  ->  {'PASS' if ok else 'FAIL'}")
        assert ok
    print(f"  Total probes checked: {total}  -> ALL PASSED\n")

def make_dataset(n, seed=42):
    rng = random.Random(seed)
    keys = rng.sample(range(10**9, 10**9 + 20 * n + 10), n)   # unique Reg. numbers
    return keys

def make_queries(keys, q_each, seed=7):
    rng = random.Random(seed)
    hits = [rng.choice(keys) for _ in range(q_each)]
    kset = set(keys)
    miss = []
    lo, hi = min(keys), max(keys)
    while len(miss) < q_each:
        x = rng.randint(lo, hi)
        if x not in kset:
            miss.append(x)
    return hits, miss

def run_queries(fn, queries):
    t0 = time.perf_counter()
    tot_c = 0
    for q in queries:
        _, c = fn(q)
        tot_c += c
    dt = time.perf_counter() - t0
    return dt / len(queries), tot_c / len(queries)

def best_of(fn, queries, reps):
    best_t, c = float("inf"), 0
    for _ in range(reps):
        t, c = run_queries(fn, queries)
        best_t = min(best_t, t)
    return best_t, c

def experiments():
    sizes = [1000, 10000, 100000, 1000000]
    Q = 100                                    # queries of each kind (hit / miss)
    results = {}
    print("=" * 70)
    print(f"EXPERIMENT: {Q} successful + {Q} unsuccessful searches per size, same data & queries for all")
    print("=" * 70)
    for n in sizes:
        keys = make_dataset(n)
        hits, miss = make_queries(keys, Q)

        t0 = time.perf_counter(); srt = sorted(keys); t_sort = time.perf_counter() - t0
        t0 = time.perf_counter(); hx = HashIndex(keys); t_hash = time.perf_counter() - t0

        # cross-check correctness on this dataset
        for q in hits[:20] + miss[:20]:
            a = linear_search(keys, q)[0] != -1
            b = binary_search(srt, q)[0] != -1
            c = hx.search(q)[0] != -1
            assert a == b == c

        reps = 3 if n <= 100000 else 1
        r = {"n": n, "t_sort": t_sort, "t_hash_build": t_hash, "hash_load": n / hx.m}
        L = lambda q: linear_search(keys, q)
        B = lambda q: binary_search(srt, q)
        H = hx.search
        for name, fn in (("linear", L), ("binary", B), ("hash", H)):
            th, ch = best_of(fn, hits, reps)
            tm, cm = best_of(fn, miss, reps)
            r[name] = {"t_hit": th, "c_hit": ch, "t_miss": tm, "c_miss": cm,
                       "t_avg": (th + tm) / 2, "c_avg": (ch + cm) / 2}
        # explicit best / worst cases
        r["linear"]["best"] = linear_search(keys, keys[0])[1]              # first record
        r["linear"]["worst"] = linear_search(keys, keys[-1])[1]            # last record
        r["binary"]["best"] = binary_search(srt, srt[(n - 1) // 2])[1]     # middle record
        r["binary"]["worst"] = max(binary_search(srt, q)[1] for q in hits + miss)
        chain_lens = [len(c) for c in hx.t if c]
        r["hash"]["best"] = 1
        r["hash"]["worst"] = max(chain_lens)                               # longest chain
        results[n] = r
        print(f"n={n:>9,d}  sort={t_sort*1e3:9.2f} ms  hash-build={t_hash*1e3:9.2f} ms  "
              f"(load factor {n/hx.m:.2f}, longest chain {max(chain_lens)})")
        for name in ("linear", "binary", "hash"):
            x = r[name]
            print(f"   {name:7s} hit: {x['t_hit']*1e6:12.3f} us {x['c_hit']:12.1f} cmp | "
                  f"miss: {x['t_miss']*1e6:12.3f} us {x['c_miss']:12.1f} cmp | "
                  f"best={x['best']} worst={x['worst']}")
    return results

def day4_experiment(results, n=1_000_000):
    print("\n" + "=" * 70)
    print("DAY-4 CHALLENGE: n = 1,000,000 ; 50,000 searches/hour ; updates once per day")
    print("=" * 70)
    keys = make_dataset(n)
    srt = sorted(keys)
    rng = random.Random(99)
    new_keys = rng.sample(range(2 * 10**9, 2 * 10**9 + 10**6), 2000)   # one day's new admissions
    out = {}
    # (a) naive nightly rebuild: re-sort everything
    allk = keys + new_keys
    t0 = time.perf_counter(); s2 = sorted(allk); out["resort_full"] = time.perf_counter() - t0
    # (b) nightly merge: sort k new keys, merge into existing sorted array (O(n + k))
    import heapq
    t0 = time.perf_counter()
    nk = sorted(new_keys)
    merged = list(heapq.merge(srt, nk))
    out["merge_update"] = time.perf_counter() - t0
    assert merged == s2
    # (b2) Timsort on the already-sorted array with the new sorted run appended (exploits existing runs)
    t0 = time.perf_counter()
    inplace = srt + nk
    inplace.sort()
    out["timsort_presorted_plus_run"] = time.perf_counter() - t0
    assert inplace == s2
    # memory of the extra index structure (tracemalloc, keys themselves excluded)
    import tracemalloc
    tracemalloc.start(); _s = sorted(keys); out["mem_sorted_MB"] = tracemalloc.get_traced_memory()[0] / 1e6; tracemalloc.stop(); del _s
    tracemalloc.start(); _h = HashIndex(keys); out["mem_hash_MB"] = tracemalloc.get_traced_memory()[0] / 1e6; tracemalloc.stop(); del _h
    # (c) hash: incremental insert of k keys (O(k)) vs full rebuild
    hx = HashIndex(keys)
    t0 = time.perf_counter()
    for i, k in enumerate(new_keys):
        hx.insert(k, n + i)
    out["hash_insert_k"] = time.perf_counter() - t0
    t0 = time.perf_counter(); HashIndex(allk); out["hash_rebuild"] = time.perf_counter() - t0
    for k, v in out.items():
        print(f"  {k:28s}: " + (f"{v:10.2f} MB" if k.startswith("mem") else f"{v*1e3:10.2f} ms"))
    # workload analysis using measured per-search times at n = 1,000,000
    r = results[n]
    per_day = 50_000 * 24
    print(f"\n  Searches per day = {per_day:,}  ({50_000/3600:.2f} per second)")
    work = {}
    for name, pre in (("linear", 0.0), ("binary", r["t_sort"]), ("hash", r["t_hash_build"])):
        per = r[name]["t_avg"]
        hour = pre + 50_000 * per
        day = pre + per_day * per
        work[name] = {"pre": pre, "per_search": per, "hour_total": hour, "day_total": day,
                      "cpu_pct_hour": 100 * (50_000 * per) / 3600}
        print(f"  {name:7s} per-search={per*1e6:11.3f} us | 50,000 searches={50_000*per:9.4f} s | "
              f"one-off preprocessing={pre:7.3f} s | day total={day:10.3f} s | "
              f"CPU busy (search only) = {100*(50_000*per)/3600:8.4f} %")
    # break-even searches vs linear
    for name in ("binary", "hash"):
        saved = r["linear"]["t_avg"] - r[name]["t_avg"]
        be = work[name]["pre"] / saved
        work[name]["breakeven_vs_linear"] = be
        print(f"  break-even vs linear for {name}: {be:,.1f} searches")
    out["work"] = work
    return out

if __name__ == "__main__":
    random.seed(0)
    print(f"Python {sys.version.split()[0]}")
    correctness_tests()
    res = experiments()
    d4 = day4_experiment(res)
    with open("results.json", "w") as f:
        json.dump({"res": {str(k): v for k, v in res.items()}, "d4": d4}, f, indent=1)
    print("\nResults saved to results.json")

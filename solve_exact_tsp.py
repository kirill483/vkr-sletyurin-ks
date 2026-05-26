import pickle
import time
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


DATASET_PATH = "data/300_10.pkl"
OUT_PATH = "results/exact_300_10.pkl"

SCALE = 1_000_000
INF = np.int64(9_000_000_000_000_000_000 // 4)
PARENT_NONE = np.uint16(65535)


# Ограничитель, чтобы случайно не запустить exact DP на слишком большом числе кластеров.
# Для m=20, k=5 память примерно: dp ~ 838 MB, parent ~ 209 MB.
MAX_EXACT_FIELDS = 22


def compute_real_cost(depot, templates, pi):
    """
    depot: [2]
    templates: [N, K, 5]
    pi: [N], значения 1..N*K

    Считает реальную float-стоимость маршрута:
        depot -> input первого шаблона
        + coverage первого шаблона
        + output_i -> input_{i+1}
        + coverage следующих шаблонов
        + output последнего шаблона -> depot
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    selected = np.asarray(pi, dtype=np.int64) - 1

    field_ids = selected // n_templates
    template_ids = selected % n_templates

    assert np.array_equal(
        np.sort(field_ids), np.arange(n_fields)
    ), "Invalid tour: each field must be selected exactly once"

    chosen = templates[field_ids, template_ids]

    chosen_in = chosen[:, 0:2]
    chosen_out = chosen[:, 2:4]
    coverage_lengths = chosen[:, 4]

    travel_cost = (
        np.linalg.norm(chosen_in[0] - depot)
        + np.linalg.norm(chosen_in[1:] - chosen_out[:-1], axis=1).sum()
        + np.linalg.norm(chosen_out[-1] - depot)
    )

    coverage_cost = coverage_lengths.sum()

    return float(travel_cost + coverage_cost)


def build_costs(depot, templates):
    """
    Возвращает int64-стоимости:
        start_cost[f, t] = depot -> template(f,t), включая coverage(f,t)
        trans_cost[f1,t1,f2,t2] = template(f1,t1) -> template(f2,t2), включая coverage(f2,t2)
        end_cost[f,t] = template(f,t) -> depot
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    template_in = templates[..., 0:2]
    template_out = templates[..., 2:4]
    template_len = templates[..., 4]

    start_cost = (
        np.linalg.norm(template_in - depot[None, None, :], axis=-1)
        + template_len
    )

    end_cost = np.linalg.norm(template_out - depot[None, None, :], axis=-1)

    trans_cost = (
        np.linalg.norm(
            template_out[:, :, None, None, :] - template_in[None, None, :, :, :],
            axis=-1,
        )
        + template_len[None, None, :, :]
    )

    start_int = np.rint(start_cost * SCALE).astype(np.int64)
    trans_int = np.rint(trans_cost * SCALE).astype(np.int64)
    end_int = np.rint(end_cost * SCALE).astype(np.int64)

    return start_int, trans_int, end_int


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _idx(mask, f, t, m, k):
        return (mask * m + f) * k + t


    @njit(cache=True)
    def _held_karp_gtsp_numba(start_cost, trans_cost, end_cost):
        m, k = start_cost.shape
        full = 1 << m
        total_states = full * m * k

        dp = np.empty(total_states, dtype=np.int64)
        parent = np.empty(total_states, dtype=np.uint16)

        for i in range(total_states):
            dp[i] = INF
            parent[i] = PARENT_NONE

        # Инициализация: depot -> выбранный шаблон стартового кластера.
        for f in range(m):
            mask = 1 << f
            for t in range(k):
                pos = _idx(mask, f, t, m, k)
                dp[pos] = start_cost[f, t]
                parent[pos] = PARENT_NONE

        # Основная динамика по подмножествам кластеров.
        for mask in range(full):
            for last_f in range(m):
                if (mask & (1 << last_f)) == 0:
                    continue

                for last_t in range(k):
                    cur_pos = _idx(mask, last_f, last_t, m, k)
                    cur_cost = dp[cur_pos]
                    if cur_cost >= INF:
                        continue

                    for next_f in range(m):
                        if (mask & (1 << next_f)) != 0:
                            continue

                        next_mask = mask | (1 << next_f)

                        for next_t in range(k):
                            new_cost = cur_cost + trans_cost[last_f, last_t, next_f, next_t]
                            new_pos = _idx(next_mask, next_f, next_t, m, k)

                            if new_cost < dp[new_pos]:
                                dp[new_pos] = new_cost
                                parent[new_pos] = np.uint16(last_f * k + last_t)

        full_mask = full - 1
        best_cost = INF
        best_f = -1
        best_t = -1

        for f in range(m):
            for t in range(k):
                pos = _idx(full_mask, f, t, m, k)
                total_cost = dp[pos] + end_cost[f, t]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_f = f
                    best_t = t

        return best_cost, best_f, best_t, parent

else:
    _held_karp_gtsp_numba = None


def _held_karp_gtsp_python(start_cost, trans_cost, end_cost):
    """
    Медленный fallback без numba. Подходит только для маленьких m.
    Для GTSP20 настоятельно лучше установить numba.
    """

    m, k = start_cost.shape
    full = 1 << m

    dp = {}
    parent = {}

    for f in range(m):
        mask = 1 << f
        for t in range(k):
            key = (mask, f, t)
            dp[key] = int(start_cost[f, t])
            parent[key] = None

    for mask in range(full):
        items = [
            (last_f, last_t, cur_cost)
            for (msk, last_f, last_t), cur_cost in dp.items()
            if msk == mask
        ]

        for last_f, last_t, cur_cost in items:
            for next_f in range(m):
                if mask & (1 << next_f):
                    continue

                next_mask = mask | (1 << next_f)

                for next_t in range(k):
                    new_key = (next_mask, next_f, next_t)
                    new_cost = cur_cost + int(trans_cost[last_f, last_t, next_f, next_t])

                    if new_key not in dp or new_cost < dp[new_key]:
                        dp[new_key] = new_cost
                        parent[new_key] = (last_f, last_t)

    full_mask = full - 1
    best_cost = None
    best_f = None
    best_t = None

    for f in range(m):
        for t in range(k):
            key = (full_mask, f, t)
            total_cost = dp[key] + int(end_cost[f, t])
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_f = f
                best_t = t

    route = []
    mask = full_mask
    f = best_f
    t = best_t

    while True:
        route.append((f, t))
        prev = parent[(mask, f, t)]
        if prev is None:
            break
        mask ^= 1 << f
        f, t = prev

    route.reverse()
    return int(best_cost), route


def solve_gtsp_exact_dp(depot, templates):
    """
    Точный Held-Karp-style DP для GTSP.

    Возвращает:
        cost_float, pi, status

    status:
        OPTIMAL — точный оптимум найден
        ERROR_* — что-то пошло не так
    """

    n_fields, n_templates, template_dim = templates.shape
    assert template_dim >= 5, "Expected templates with coverage length: [N, K, 5]"

    if n_fields > MAX_EXACT_FIELDS:
        return None, None, f"ERROR_TOO_MANY_FIELDS_FOR_EXACT_DP_{n_fields}"

    start_cost, trans_cost, end_cost = build_costs(depot, templates)

    if NUMBA_AVAILABLE:
        best_cost_int, best_f, best_t, parent = _held_karp_gtsp_numba(
            start_cost, trans_cost, end_cost
        )

        if best_f < 0:
            return None, None, "ERROR_NO_SOLUTION"

        # Восстановление маршрута по parent.
        m = n_fields
        k = n_templates
        mask = (1 << m) - 1
        f = int(best_f)
        t = int(best_t)
        route = []

        while True:
            route.append((f, t))
            pos = (mask * m + f) * k + t
            p = int(parent[pos])
            if p == int(PARENT_NONE):
                break

            mask ^= 1 << f
            f = p // k
            t = p % k

        route.reverse()
    else:
        print(
            "WARNING: numba is not installed. Using very slow pure-Python fallback. "
            "For GTSP20 install numba: pip install numba"
        )
        best_cost_int, route = _held_karp_gtsp_python(start_cost, trans_cost, end_cost)

    # pi в формате 1..N*K, как в твоём OR-Tools коде.
    pi = [1 + f * n_templates + t for f, t in route]

    # Пересчитываем float-стоимость без округления SCALE, чтобы формат совпадал с эвристическим кодом.
    cost_float = compute_real_cost(depot, templates, pi)

    return cost_float, pi, "OPTIMAL"


def unpack_sample(sample):
    """
    Поддерживает оба формата:

    1) tuple:
        (depot, templates)

    2) dict:
        {
            "depot": depot,
            "templates": templates
        }
    """

    if isinstance(sample, dict):
        depot = sample["depot"]
        templates = sample["templates"]
    else:
        depot, templates = sample

    depot = np.asarray(depot, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)

    return depot, templates


if __name__ == "__main__":
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    results = []

    for i, sample in enumerate(dataset):
        depot, templates = unpack_sample(sample)

        start = time.time()
        cost, pi, status = solve_gtsp_exact_dp(depot, templates)
        duration = time.time() - start

        results.append((cost, pi, duration, status))

        if cost is None:
            print(
                f"{i + 1}/{len(dataset)} no solution, "
                f"status={status}, time={duration:.3f}s"
            )
        else:
            print(
                f"{i + 1}/{len(dataset)} "
                f"cost={cost:.6f}, status={status}, time={duration:.3f}s"
            )

    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved exact DP GTSP results to: {OUT_PATH}")

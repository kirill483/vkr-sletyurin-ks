import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_path = "data/300_10.pkl"
log_path = "outputs/tsp_20/=10_1ATTENTION20runtestdeportGTSP20_20260522T135527/train_log.jsonl"


def unpack_sample(sample):
    """
    Поддерживает:
        (depot, templates)

    или:
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

    return np.asarray(depot, dtype=np.float64), np.asarray(templates, dtype=np.float64)


def compute_instance_constant(templates):
    """
    templates: [N, K, 4] или [N, K, 5]

    Если есть length:
        constant = sum_i min_m templates[i, m, 4]

    Если length нет:
        constant = 0
    """
    if templates.shape[-1] < 5:
        return 0.0

    lengths = templates[:, :, 4]          # [N, K]
    min_per_field = lengths.min(axis=1)   # [N]

    return float(min_per_field.sum())


with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

constants = []

for sample in dataset:
    _, templates = unpack_sample(sample)
    constants.append(compute_instance_constant(templates))

constants = np.asarray(constants, dtype=np.float64)

mean_constant = float(constants.mean())
median_constant = float(np.median(constants))
std_constant = float(constants.std())

print(f"Dataset size: {len(dataset)}")
print(f"Mean coverage constant:   {mean_constant:.6f}")
print(f"Median coverage constant: {median_constant:.6f}")
print(f"Std coverage constant:    {std_constant:.6f}")


rows = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

# Берём именно тот датасет, на котором проверяется,
# обновлять rollout baseline или нет
rb = df[df["type"] == "rollout_baseline"].copy()
rb = rb.sort_values("epoch")

# Выкидываем среднюю константу покрытия из средних cost-ов
rb["candidate_mean_no_const"] = rb["candidate_mean"] - mean_constant
rb["baseline_mean_before_no_const"] = rb["baseline_mean_before"] - mean_constant

plt.figure(figsize=(10, 5))

plt.plot(
    rb["epoch"],
    rb["candidate_mean_no_const"],
    marker="o",
    label="Кандидат без константы",
)

plt.plot(
    rb["epoch"],
    rb["baseline_mean_before_no_const"],
    linestyle="--",
    label="Текущий baseline без константы",
)

# Отмечаем эпохи, где baseline обновился
updates = rb[rb["baseline_updated"] == True]

if len(updates) > 0:
    plt.scatter(
        updates["epoch"],
        updates["candidate_mean_no_const"],
        s=80,
        marker="*",
        label="Baseline обновлён",
    )

plt.xlabel("Эпоха")
plt.ylabel("Длина маршрута без константы")
plt.title("График обучения на валидационном датасете без константы покрытия")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("grafik_10_no_const.png", dpi=200)
plt.show()
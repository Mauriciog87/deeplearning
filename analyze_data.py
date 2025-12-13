import sqlite3
from collections import Counter
import numpy as np

conn = sqlite3.connect('data/roulette.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tablas:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT id, name, source FROM sessions")
sessions = cur.fetchall()
print("\nSesiones:")
for s in sessions:
    print(f"  ID {s['id']}: {s['name']} ({s['source']})")

cur.execute("SELECT session_id, COUNT(*) as cnt FROM spins GROUP BY session_id")
print("\nSpins por sesión:")
for r in cur.fetchall():
    print(f"  Session {r['session_id']}: {r['cnt']} spins")

cur.execute("SELECT number FROM spins ORDER BY timestamp")
numbers = [r['number'] for r in cur.fetchall()]
print(f"\nTotal spins: {len(numbers)}")

RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

counter = Counter(numbers)
print("\n=== DISTRIBUCIÓN DE NÚMEROS ===")
print("Top 10 más frecuentes:")
for num, cnt in counter.most_common(10):
    pct = cnt / len(numbers) * 100
    expected = 100/37
    color = "🔴" if num in RED else ("🟢" if num == 0 else "⚫")
    deviation = pct - expected
    print(f"  {color} {num:2d}: {cnt:3d} veces ({pct:.1f}%) [desv: {deviation:+.1f}%]")

print("\n10 menos frecuentes:")
for num, cnt in counter.most_common()[-10:]:
    pct = cnt / len(numbers) * 100
    expected = 100/37
    color = "🔴" if num in RED else ("🟢" if num == 0 else "⚫")
    deviation = pct - expected
    print(f"  {color} {num:2d}: {cnt:3d} veces ({pct:.1f}%) [desv: {deviation:+.1f}%]")

colors = ["red" if n in RED else ("green" if n == 0 else "black") for n in numbers]
color_counter = Counter(colors)
print("\n=== COLORES ===")
for c, cnt in color_counter.most_common():
    pct = cnt / len(numbers) * 100
    if c == "red":
        expected = 18/37 * 100
    elif c == "black":
        expected = 18/37 * 100
    else:
        expected = 1/37 * 100
    print(f"  {c}: {cnt} ({pct:.1f}%) [esperado: {expected:.1f}%]")

parities = ["zero" if n == 0 else ("odd" if n % 2 == 1 else "even") for n in numbers]
parity_counter = Counter(parities)
print("\n=== PAR/IMPAR ===")
for p, cnt in parity_counter.most_common():
    pct = cnt / len(numbers) * 100
    print(f"  {p}: {cnt} ({pct:.1f}%)")

high_low = ["zero" if n == 0 else ("high" if n >= 19 else "low") for n in numbers]
hl_counter = Counter(high_low)
print("\n=== ALTO/BAJO ===")
for h, cnt in hl_counter.most_common():
    pct = cnt / len(numbers) * 100
    print(f"  {h}: {cnt} ({pct:.1f}%)")

dozens = [0 if n == 0 else (n-1)//12 + 1 for n in numbers]
dozen_counter = Counter(dozens)
print("\n=== DOCENAS ===")
for d, cnt in sorted(dozen_counter.items()):
    pct = cnt / len(numbers) * 100
    name = f"Docena {d}" if d > 0 else "Zero"
    print(f"  {name}: {cnt} ({pct:.1f}%)")

columns = [0 if n == 0 else ((n-1) % 3) + 1 for n in numbers]
col_counter = Counter(columns)
print("\n=== COLUMNAS ===")
for c, cnt in sorted(col_counter.items()):
    pct = cnt / len(numbers) * 100
    name = f"Columna {c}" if c > 0 else "Zero"
    print(f"  {name}: {cnt} ({pct:.1f}%)")

print("\n=== ANÁLISIS DE SECUENCIAS ===")

same_color_streaks = []
current_streak = 1
for i in range(1, len(colors)):
    if colors[i] == colors[i-1] and colors[i] != "green":
        current_streak += 1
    else:
        if current_streak > 1:
            same_color_streaks.append(current_streak)
        current_streak = 1
print(f"Rachas de mismo color (>1): {len(same_color_streaks)}")
if same_color_streaks:
    print(f"  Máxima: {max(same_color_streaks)}, Promedio: {np.mean(same_color_streaks):.1f}")

same_dozen_streaks = []
current_streak = 1
for i in range(1, len(dozens)):
    if dozens[i] == dozens[i-1] and dozens[i] != 0:
        current_streak += 1
    else:
        if current_streak > 1:
            same_dozen_streaks.append(current_streak)
        current_streak = 1
print(f"Rachas de misma docena (>1): {len(same_dozen_streaks)}")
if same_dozen_streaks:
    print(f"  Máxima: {max(same_dozen_streaks)}, Promedio: {np.mean(same_dozen_streaks):.1f}")

print("\n=== PATRONES DE TRANSICIÓN ===")
transitions = {}
for i in range(1, len(numbers)):
    prev_dozen = dozens[i-1]
    curr_dozen = dozens[i]
    key = (prev_dozen, curr_dozen)
    transitions[key] = transitions.get(key, 0) + 1

print("Transiciones entre docenas:")
for d1 in [1, 2, 3]:
    row = []
    for d2 in [1, 2, 3]:
        cnt = transitions.get((d1, d2), 0)
        total_from_d1 = sum(transitions.get((d1, x), 0) for x in [0,1,2,3])
        pct = cnt / total_from_d1 * 100 if total_from_d1 > 0 else 0
        row.append(f"{pct:.0f}%")
    print(f"  D{d1} -> D1:{row[0]} D2:{row[1]} D3:{row[2]}")

print("\n=== NÚMEROS CALIENTES/FRÍOS ===")
expected_per_number = len(numbers) / 37
hot_threshold = expected_per_number * 1.5
cold_threshold = expected_per_number * 0.5

hot = [(n, c) for n, c in counter.items() if c >= hot_threshold]
cold = [(n, c) for n, c in counter.items() if c <= cold_threshold]
print(f"Números calientes (>150% esperado): {[n for n,c in hot]}")
print(f"Números fríos (<50% esperado): {[n for n,c in cold]}")

missing = [n for n in range(37) if n not in counter]
print(f"Números que nunca salieron: {missing}")

print("\n=== TEST CHI-CUADRADO ===")
expected = len(numbers) / 37
chi2 = sum((counter.get(n, 0) - expected)**2 / expected for n in range(37))
print(f"Chi² = {chi2:.2f}")
print(f"Grados de libertad = 36")
print(f"Valor crítico (α=0.05) = 50.99")
if chi2 < 50.99:
    print("✓ La distribución es estadísticamente uniforme (no hay sesgo significativo)")
else:
    print("⚠ La distribución muestra desviación significativa de la uniformidad")

conn.close()

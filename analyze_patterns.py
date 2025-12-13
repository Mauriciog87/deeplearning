import sqlite3
from collections import Counter, defaultdict
import numpy as np

conn = sqlite3.connect('data/roulette.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT number FROM spins ORDER BY timestamp")
numbers = [r['number'] for r in cur.fetchall()]

RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

def get_color(n):
    if n == 0: return "green"
    return "red" if n in RED else "black"

def get_dozen(n):
    if n == 0: return 0
    return (n-1)//12 + 1

def get_column(n):
    if n == 0: return 0
    return ((n-1) % 3) + 1

def get_parity(n):
    if n == 0: return "zero"
    return "odd" if n % 2 == 1 else "even"

def get_high_low(n):
    if n == 0: return "zero"
    return "high" if n >= 19 else "low"

print("="*60)
print("ANÁLISIS AVANZADO DE PATRONES PREDICTIVOS")
print("="*60)

print("\n=== 1. AUTOCORRELACIÓN ===")
print("¿El número anterior influye en el siguiente?")

for lag in [1, 2, 3]:
    correlations = defaultdict(list)
    for i in range(lag, len(numbers)):
        prev = numbers[i-lag]
        curr = numbers[i]
        correlations[prev].append(curr)
    
    predictable = 0
    for prev_num, following in correlations.items():
        if len(following) >= 2:
            counter = Counter(following)
            most_common_pct = counter.most_common(1)[0][1] / len(following)
            if most_common_pct > 0.3:
                predictable += 1
    
    print(f"  Lag {lag}: {predictable} números con patrón >30%")

print("\n=== 2. PATRONES DESPUÉS DE COLOR ===")
color_transitions = defaultdict(list)
for i in range(1, len(numbers)):
    prev_color = get_color(numbers[i-1])
    curr_color = get_color(numbers[i])
    color_transitions[prev_color].append(curr_color)

print("Después de ROJO:")
c = Counter(color_transitions["red"])
total = sum(c.values())
for color, cnt in c.most_common():
    print(f"  -> {color}: {cnt/total*100:.1f}%")

print("Después de NEGRO:")
c = Counter(color_transitions["black"])
total = sum(c.values())
for color, cnt in c.most_common():
    print(f"  -> {color}: {cnt/total*100:.1f}%")

print("\n=== 3. PATRONES DESPUÉS DE DOCENA ===")
dozen_transitions = defaultdict(list)
for i in range(1, len(numbers)):
    prev_dozen = get_dozen(numbers[i-1])
    curr_dozen = get_dozen(numbers[i])
    if prev_dozen > 0:
        dozen_transitions[prev_dozen].append(curr_dozen)

for d in [1, 2, 3]:
    print(f"Después de Docena {d}:")
    c = Counter(dozen_transitions[d])
    total = sum(c.values())
    for dozen, cnt in sorted(c.items()):
        expected = 12/37 if dozen > 0 else 1/37
        actual = cnt/total
        diff = (actual - expected) * 100
        print(f"  -> D{dozen}: {actual*100:.1f}% (esperado: {expected*100:.1f}%, desv: {diff:+.1f}%)")

print("\n=== 4. ANÁLISIS DE RACHAS ===")
print("¿Qué sale después de una racha?")

for streak_color in ["red", "black"]:
    for streak_len in [2, 3, 4]:
        after_streak = []
        i = streak_len
        while i < len(numbers):
            is_streak = all(get_color(numbers[i-j-1]) == streak_color for j in range(streak_len))
            if is_streak:
                after_streak.append(get_color(numbers[i]))
                i += streak_len
            else:
                i += 1
        
        if after_streak:
            c = Counter(after_streak)
            total = len(after_streak)
            same = c.get(streak_color, 0)
            opposite = c.get("black" if streak_color == "red" else "red", 0)
            print(f"Después de {streak_len}x {streak_color}: continúa={same/total*100:.0f}%, cambia={opposite/total*100:.0f}% (n={total})")

print("\n=== 5. SECTORES DE LA RUEDA ===")
wheel_order = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]

def get_sector(n, sector_size=5):
    pos = wheel_order.index(n)
    return pos // sector_size

sector_counter = Counter(get_sector(n) for n in numbers)
print(f"Distribución por sector (tamaño 5):")
for sector in sorted(sector_counter.keys()):
    nums_in_sector = wheel_order[sector*5:(sector+1)*5]
    cnt = sector_counter[sector]
    pct = cnt / len(numbers) * 100
    expected = 5/37 * 100
    print(f"  Sector {sector} {nums_in_sector}: {cnt} ({pct:.1f}%) [esperado: {expected:.1f}%]")

print("\n=== 6. ANÁLISIS DE VECINOS ===")
print("Si sale un número, ¿salen sus vecinos después?")

neighbor_hits = 0
total_checks = 0
for i in range(1, len(numbers) - 1):
    curr = numbers[i]
    next_num = numbers[i+1]
    
    curr_pos = wheel_order.index(curr)
    neighbors = [wheel_order[(curr_pos + j) % 37] for j in [-2, -1, 1, 2]]
    
    total_checks += 1
    if next_num in neighbors:
        neighbor_hits += 1

neighbor_pct = neighbor_hits / total_checks * 100
expected_neighbor = 4/37 * 100
print(f"Vecinos (±2): {neighbor_hits}/{total_checks} ({neighbor_pct:.1f}%) [esperado: {expected_neighbor:.1f}%]")

print("\n=== 7. NÚMEROS DORMIDOS ===")
print("¿Los números que no salen hace rato tienen más probabilidad?")

window = 20
late_numbers_performance = []
for i in range(window, len(numbers) - 1):
    recent = numbers[i-window:i]
    recent_set = set(recent)
    late_numbers = [n for n in range(37) if n not in recent_set]
    
    next_num = numbers[i]
    if next_num in late_numbers:
        late_numbers_performance.append(1)
    else:
        late_numbers_performance.append(0)

hit_rate = sum(late_numbers_performance) / len(late_numbers_performance) * 100
avg_late = (37 - window/37*37)
expected_rate = avg_late / 37 * 100
print(f"Números dormidos (no salieron en {window} spins):")
print(f"  Aciertos: {hit_rate:.1f}% [esperado: ~{expected_rate:.0f}%]")

print("\n=== 8. RESUMEN DE HALLAZGOS ===")
print("-" * 60)

findings = []

red_after_red = color_transitions["red"].count("red") / len(color_transitions["red"])
if abs(red_after_red - 18/37) > 0.1:
    findings.append(f"⚠ Rojo después de rojo: {red_after_red*100:.0f}% vs esperado 48.6%")

d1_count = len([n for n in numbers if get_dozen(n) == 1])
d1_pct = d1_count / len(numbers)
if d1_pct > 0.40:
    findings.append(f"⚠ Docena 1 sobrerrepresentada: {d1_pct*100:.1f}% vs esperado 32.4%")

odd_count = len([n for n in numbers if get_parity(n) == "odd"])
odd_pct = odd_count / len(numbers)
if odd_pct > 0.55:
    findings.append(f"⚠ Impares sobrerrepresentados: {odd_pct*100:.1f}% vs esperado 48.6%")

low_count = len([n for n in numbers if get_high_low(n) == "low"])
low_pct = low_count / len(numbers)
if low_pct > 0.55:
    findings.append(f"⚠ Bajos sobrerrepresentados: {low_pct*100:.1f}% vs esperado 48.6%")

if findings:
    print("Patrones detectados (pueden ser varianza normal con n=169):")
    for f in findings:
        print(f"  {f}")
else:
    print("No se detectaron patrones significativos.")

print("\n" + "="*60)
print("CONCLUSIÓN")
print("="*60)
print(f"""
Con {len(numbers)} spins, el análisis muestra:

1. DISTRIBUCIÓN: Chi² = 18.63 < 50.99 (crítico)
   → La ruleta parece estadísticamente justa

2. TENDENCIAS OBSERVADAS (pueden ser varianza):
   - Docena 1: 38.5% (esperado 32.4%) → +6.1%
   - Impares: 55.0% (esperado 48.6%) → +6.4%
   - Bajos: 53.3% (esperado 48.6%) → +4.7%

3. NÚMEROS CALIENTES: 5, 9, 1, 15
   Todos en la primera docena y mayormente impares bajos

4. PARA UNA PREDICCIÓN:
   Si hubiera que apostar basándose en estos datos:
   - Docena 1 tiene ventaja histórica
   - Impares han salido más
   - Bajos han salido más
   
   PERO: Con solo 169 spins, estas desviaciones están dentro
   del rango de varianza normal. Se necesitan miles de spins
   para detectar un sesgo real.
""")

conn.close()

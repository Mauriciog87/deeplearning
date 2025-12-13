#!/usr/bin/env python
import argparse
import sys
import os
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_color_emoji(color: str) -> str:
    return {"red": "🔴", "black": "⚫", "green": "🟢"}.get(color, "")


def format_spin_display(number: int) -> str:
    from src.database.models import Spin
    spin = Spin(number=number)
    emoji = get_color_emoji(spin.color)
    if number == 0:
        return f"{emoji} {number} (Verde)"
    return f"{emoji} {number} ({spin.color.capitalize()}, {spin.parity.capitalize()}, {spin.high_low.capitalize()}, D{spin.dozen}, C{spin.column})"


def session_create(args):
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    
    name = args.name or input("Nombre de la sesión: ").strip()
    if not name:
        print("❌ El nombre no puede estar vacío")
        return 1
    
    casino = args.casino or ""
    notes = args.notes or ""
    
    session = repo.create_empty_session(name, source="manual", casino=casino, notes=notes)
    
    print(f"\n✅ Sesión creada exitosamente")
    print(f"   ID: {session.id}")
    print(f"   Nombre: {session.name}")
    if casino:
        print(f"   Casino: {casino}")
    print(f"\n💡 Para agregar números usa:")
    print(f"   python roulette_cli.py live {session.id}")
    
    return 0


def session_live(args):
    from src.database import RouletteRepository
    from src.database.models import Spin
    
    repo = RouletteRepository()
    
    session_id = args.session_id
    session = repo.db.get_session(session_id)
    
    if not session:
        print(f"❌ Sesión {session_id} no encontrada")
        return 1
    
    print(f"\n🎰 Modo Live - Sesión: {session.name} (ID: {session.id})")
    print("=" * 60)
    print("Ingresá números de ruleta (0-36)")
    print("Comandos: 'q' salir | 's' estadísticas | 'l' últimos 10 | 'h' ayuda")
    print("=" * 60)
    
    numbers_added = 0
    
    while True:
        try:
            user_input = input("\n🎲 Número: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'h':
                print("\n📖 Comandos disponibles:")
                print("   q - Salir")
                print("   s - Ver estadísticas")
                print("   l - Ver últimos 10 números")
                print("   c - Ver conteo por color")
                print("   d - Ver conteo por docena")
                continue
            elif user_input == 's':
                all_numbers = repo.get_numbers_by_session(session_id)
                if all_numbers:
                    from src.utils.analysis import quick_stats
                    stats = quick_stats(all_numbers)
                    print(f"\n📊 Estadísticas ({len(all_numbers)} spins):")
                    print(f"   🔴 Rojo: {stats['red']} ({stats['red_pct']:.1f}%)")
                    print(f"   ⚫ Negro: {stats['black']} ({stats['black_pct']:.1f}%)")
                    print(f"   🟢 Verde: {stats['green']} ({stats['green_pct']:.1f}%)")
                else:
                    print("   No hay números aún")
                continue
            elif user_input == 'l':
                all_numbers = repo.get_numbers_by_session(session_id)
                if all_numbers:
                    last_10 = all_numbers[-10:]
                    print(f"\n📜 Últimos {len(last_10)} números:")
                    for i, n in enumerate(last_10, 1):
                        print(f"   {i}. {format_spin_display(n)}")
                else:
                    print("   No hay números aún")
                continue
            elif user_input == 'c':
                all_numbers = repo.get_numbers_by_session(session_id)
                if all_numbers:
                    colors = {"red": 0, "black": 0, "green": 0}
                    for n in all_numbers:
                        colors[Spin(number=n).color] += 1
                    print(f"\n🎨 Conteo por color:")
                    print(f"   🔴 Rojo: {colors['red']}")
                    print(f"   ⚫ Negro: {colors['black']}")
                    print(f"   🟢 Verde: {colors['green']}")
                continue
            elif user_input == 'd':
                all_numbers = repo.get_numbers_by_session(session_id)
                if all_numbers:
                    dozens = {0: 0, 1: 0, 2: 0, 3: 0}
                    for n in all_numbers:
                        dozens[Spin(number=n).dozen] += 1
                    print(f"\n📈 Conteo por docena:")
                    print(f"   D1 (1-12): {dozens[1]}")
                    print(f"   D2 (13-24): {dozens[2]}")
                    print(f"   D3 (25-36): {dozens[3]}")
                    print(f"   Zero: {dozens[0]}")
                continue
            
            try:
                num = int(user_input)
                if 0 <= num <= 36:
                    spin = repo.add_number_to_session(session_id, num)
                    numbers_added += 1
                    total = len(repo.get_numbers_by_session(session_id))
                    print(f"   ✅ {format_spin_display(num)} (Total: {total})")
                else:
                    print("   ❌ Número inválido (debe ser 0-36)")
            except ValueError:
                print("   ❌ Entrada inválida")
        
        except KeyboardInterrupt:
            print("\n")
            break
    
    print(f"\n✅ Sesión finalizada: {numbers_added} números agregados")
    return 0


def session_list(args):
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    sessions = repo.get_sessions()
    
    if not sessions:
        print("\n📋 No hay sesiones")
        return 0
    
    print("\n" + "=" * 70)
    print("📋 Sesiones")
    print("=" * 70)
    print(f"\n{'ID':<6} {'Nombre':<30} {'Spins':>8} {'Creada':<20}")
    print("-" * 70)
    
    for s in sessions:
        created = s['created_at'][:16] if s['created_at'] else 'N/A'
        name = s['name'][:28] + ".." if len(s['name']) > 30 else s['name']
        print(f"{s['id']:<6} {name:<30} {s['spin_count']:>8} {created:<20}")
    
    total_spins = sum(s['spin_count'] for s in sessions)
    print("-" * 70)
    print(f"Total: {len(sessions)} sesiones, {total_spins} spins")
    
    return 0


def predict_mode(args):
    from src.database import RouletteRepository, Prediction
    from src.utils.predictor import RoulettePredictor
    
    repo = RouletteRepository()
    
    session_id = args.session_id
    if session_id:
        session = repo.db.get_session(session_id)
        if not session:
            print(f"❌ Sesión {session_id} no encontrada")
            return 1
        history = repo.get_numbers_by_session(session_id)
    else:
        history = repo.get_all_numbers()
    
    if len(history) < 5:
        print("❌ Se necesitan al menos 5 números para predecir")
        return 1
    
    model_path = args.model or "models/roulette_agent.pt"
    predictor = RoulettePredictor(model_path)
    
    print("\n🔮 Modo Predicción")
    print("=" * 60)
    print(f"Historial: {len(history)} números")
    print("Comandos: 'q' salir | 'p' nueva predicción | 'a' accuracy")
    print("=" * 60)
    
    while True:
        try:
            predictions = predictor.predict_from_history(history)
            
            print(f"\n{'─' * 40}")
            print("📊 PREDICCIÓN:")
            print(predictor.format_prediction(predictions))
            print(f"{'─' * 40}")
            
            user_input = input("\n🎲 Ingresá el número que salió (o comando): ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'p':
                continue
            elif user_input == 'a':
                accuracy = repo.get_prediction_accuracy(session_id)
                if accuracy.get("total", 0) == 0:
                    print("\n📈 No hay predicciones registradas aún")
                else:
                    print(f"\n📈 Precisión de predicciones ({accuracy['total']} total):")
                    print(f"   Número exacto: {accuracy['number']['correct']}/{accuracy['total']} ({accuracy['number']['pct']:.1f}%)")
                    print(f"   Color: {accuracy['color']['correct']}/{accuracy['total']} ({accuracy['color']['pct']:.1f}%)")
                    print(f"   Paridad: {accuracy['parity']['correct']}/{accuracy['total']} ({accuracy['parity']['pct']:.1f}%)")
                    print(f"   Alto/Bajo: {accuracy['high_low']['correct']}/{accuracy['total']} ({accuracy['high_low']['pct']:.1f}%)")
                    print(f"   Docena: {accuracy['dozen']['correct']}/{accuracy['total']} ({accuracy['dozen']['pct']:.1f}%)")
                    print(f"   Columna: {accuracy['column']['correct']}/{accuracy['total']} ({accuracy['column']['pct']:.1f}%)")
                continue
            
            try:
                actual = int(user_input)
                if 0 <= actual <= 36:
                    pred_obj = predictor.create_prediction_object(predictions, session_id or 0)
                    pred_obj.actual_number = actual
                    pred_obj.compute_correctness()
                    
                    if session_id:
                        repo.add_number_to_session(session_id, actual)
                        history.append(actual)
                    
                    repo.save_prediction(pred_obj)
                    
                    print(f"\n✅ Resultado: {format_spin_display(actual)}")
                    print(f"\n📊 Aciertos:")
                    print(f"   {'✅' if pred_obj.color_correct else '❌'} Color")
                    print(f"   {'✅' if pred_obj.parity_correct else '❌'} Paridad")
                    print(f"   {'✅' if pred_obj.high_low_correct else '❌'} Alto/Bajo")
                    print(f"   {'✅' if pred_obj.dozen_correct else '❌'} Docena")
                    print(f"   {'✅' if pred_obj.column_correct else '❌'} Columna")
                    print(f"   {'✅' if predictions['number'][0] == actual else '❌'} Número exacto")
                else:
                    print("   ❌ Número inválido (debe ser 0-36)")
            except ValueError:
                print("   ❌ Entrada inválida")
        
        except KeyboardInterrupt:
            print("\n")
            break
    
    return 0


def show_accuracy(args):
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    session_id = args.session_id
    
    accuracy = repo.get_prediction_accuracy(session_id)
    
    if accuracy.get("total", 0) == 0:
        print("\n📈 No hay predicciones registradas")
        return 0
    
    title = f"Sesión {session_id}" if session_id else "Todas las sesiones"
    
    print(f"\n{'=' * 50}")
    print(f"📈 Precisión de Predicciones - {title}")
    print(f"{'=' * 50}")
    print(f"\nTotal de predicciones: {accuracy['total']}")
    print(f"\n{'Categoría':<20} {'Aciertos':>10} {'Porcentaje':>12}")
    print("-" * 45)
    print(f"{'Número exacto':<20} {accuracy['number']['correct']:>10} {accuracy['number']['pct']:>11.1f}%")
    print(f"{'Color':<20} {accuracy['color']['correct']:>10} {accuracy['color']['pct']:>11.1f}%")
    print(f"{'Paridad':<20} {accuracy['parity']['correct']:>10} {accuracy['parity']['pct']:>11.1f}%")
    print(f"{'Alto/Bajo':<20} {accuracy['high_low']['correct']:>10} {accuracy['high_low']['pct']:>11.1f}%")
    print(f"{'Docena':<20} {accuracy['dozen']['correct']:>10} {accuracy['dozen']['pct']:>11.1f}%")
    print(f"{'Columna':<20} {accuracy['column']['correct']:>10} {accuracy['column']['pct']:>11.1f}%")
    print("-" * 45)
    
    print("\n💡 Valores esperados por azar:")
    print("   Número: 2.7% | Color: 48.6% | Paridad: 48.6%")
    print("   Alto/Bajo: 48.6% | Docena: 32.4% | Columna: 32.4%")
    
    return 0


def add_numbers(args):
    from src.database import RouletteRepository
    
    numbers = []
    for n in args.numbers:
        try:
            num = int(n)
            if not 0 <= num <= 36:
                print(f"❌ Número inválido: {num} (debe ser 0-36)")
                return 1
            numbers.append(num)
        except ValueError:
            print(f"❌ Número inválido: {n}")
            return 1
    
    if not numbers:
        print("❌ No se proporcionaron números")
        return 1
    
    repo = RouletteRepository()
    
    session_name = args.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    sessions = repo.get_sessions()
    session_id = None
    
    for s in sessions:
        if s['name'] == session_name:
            session_id = s['id']
            break
    
    if session_id is None:
        session = repo.create_session_with_numbers(numbers, name=session_name)
        session_id = session.id
        print(f"✅ Sesión '{session_name}' creada con {len(numbers)} números")
    else:
        for num in numbers:
            repo.db.add_spin(session_id, num)
        print(f"✅ {len(numbers)} números agregados a '{session_name}'")
    
    print(f"   Números: {numbers}")
    return 0


def import_csv_cmd(args):
    from src.database import RouletteRepository
    
    if not os.path.exists(args.file):
        print(f"❌ Archivo no encontrado: {args.file}")
        return 1
    
    numbers = []
    
    try:
        with open(args.file, 'r', newline='', encoding='utf-8') as f:
            first_line = f.readline().strip()
            f.seek(0)
            
            try:
                int(first_line.split(',')[0])
                has_header = False
            except ValueError:
                has_header = True
            
            reader = csv.reader(f)
            if has_header:
                next(reader)
            
            for row in reader:
                for val in row:
                    val = val.strip()
                    if val:
                        try:
                            num = int(val)
                            if 0 <= num <= 36:
                                numbers.append(num)
                        except ValueError:
                            pass
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return 1
    
    if not numbers:
        print("❌ No se encontraron números válidos en el archivo")
        return 1
    
    repo = RouletteRepository()
    session_name = args.session or f"import_{os.path.basename(args.file)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = repo.create_session_with_numbers(numbers, name=session_name, source="csv_import")
    
    print(f"✅ {len(numbers)} números importados de {args.file}")
    print(f"   Sesión: {session_name} (ID: {session.id})")
    
    return 0


def show_stats(args):
    from src.database import RouletteRepository
    from src.utils.analysis import full_analysis
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
        title = f"Sesión {args.session}"
    else:
        numbers = repo.get_all_numbers()
        title = "Todos los datos"
    
    if not numbers:
        print("No hay datos")
        return 0
    
    analysis = full_analysis(numbers)
    
    print(f"\n{'=' * 60}")
    print(f"📊 Estadísticas: {title}")
    print(f"{'=' * 60}")
    
    print(f"\nTotal spins: {analysis['total_spins']}")
    print(f"Números únicos vistos: {analysis['unique_numbers_seen']}")
    
    c = analysis['colors']
    print(f"\n🎨 Colores:")
    print(f"   🔴 Rojo:   {c['red']:>5} ({c['red_pct']:.1f}%)")
    print(f"   ⚫ Negro:  {c['black']:>5} ({c['black_pct']:.1f}%)")
    print(f"   🟢 Verde:  {c['green']:>5} ({c['green_pct']:.1f}%)")
    
    h = analysis['halves']
    print(f"\n📊 Mitades:")
    print(f"   Bajo (1-18):   {h['low_1_18']:>5} ({h['low_pct']:.1f}%)")
    print(f"   Alto (19-36):  {h['high_19_36']:>5} ({h['high_pct']:.1f}%)")
    
    p = analysis['parity']
    print(f"\n🔢 Paridad:")
    print(f"   Impar: {p['odd']:>5} ({p['odd_pct']:.1f}%)")
    print(f"   Par:   {p['even']:>5} ({p['even_pct']:.1f}%)")
    
    d = analysis['dozens']
    print(f"\n📈 Docenas:")
    print(f"   D1 (1-12):   {d['first']:>5} ({d['first_pct']:.1f}%)")
    print(f"   D2 (13-24):  {d['second']:>5} ({d['second_pct']:.1f}%)")
    print(f"   D3 (25-36):  {d['third']:>5} ({d['third_pct']:.1f}%)")
    
    print(f"\n🔥 Números calientes (últimos 50):")
    for num, count in analysis['hot_cold']['hot'][:5]:
        print(f"   {num}: {count} veces")
    
    print(f"\n❄️ Números fríos (últimos 50):")
    for num, count in analysis['hot_cold']['cold'][:5]:
        print(f"   {num}: {count} veces")
    
    return 0


def export_data(args):
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
    else:
        numbers = repo.get_all_numbers()
    
    if not numbers:
        print("No hay datos para exportar")
        return 0
    
    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['number'])
            for num in numbers:
                writer.writerow([num])
        
        print(f"✅ {len(numbers)} números exportados a {args.output}")
        return 0
    except Exception as e:
        print(f"❌ Error exportando: {e}")
        return 1


def clear_database(args):
    if not args.confirm:
        print("❌ Usá --confirm para confirmar el borrado")
        return 1
    
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    repo.clear_all_data(confirm=True)
    
    print("✅ Base de datos limpiada")
    return 0


def interactive_mode(args):
    from src.database import RouletteRepository
    
    repo = RouletteRepository()
    
    session_name = args.session or f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = repo.create_empty_session(session_name)
    
    print(f"\n🎰 Modo Interactivo - Sesión: {session_name}")
    print("=" * 50)
    print("Ingresá números (0-36), o:")
    print("  'q' salir | 's' estadísticas | 'l' últimos 10")
    print("=" * 50 + "\n")
    
    numbers = []
    
    while True:
        try:
            user_input = input("Número: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 's':
                if numbers:
                    from src.utils.analysis import quick_stats
                    stats = quick_stats(numbers)
                    print(f"\n  Spins: {len(numbers)}")
                    print(f"  Rojo: {stats['red_pct']:.1f}%")
                    print(f"  Negro: {stats['black_pct']:.1f}%\n")
                else:
                    print("  No hay números aún\n")
                continue
            elif user_input == 'l':
                if numbers:
                    print(f"  Últimos 10: {numbers[-10:]}\n")
                else:
                    print("  No hay números aún\n")
                continue
            
            try:
                num = int(user_input)
                if 0 <= num <= 36:
                    numbers.append(num)
                    repo.db.add_spin(session.id, num)
                    print(f"  ✅ {format_spin_display(num)} (total: {len(numbers)})")
                else:
                    print("  ❌ Inválido (debe ser 0-36)")
            except ValueError:
                print("  ❌ Entrada inválida")
        
        except KeyboardInterrupt:
            print("\n")
            break
    
    print(f"\n✅ Sesión completa: {len(numbers)} números agregados")
    return 0


def bias_test_cmd(args):
    from src.database import RouletteRepository
    from src.utils.bias_detection import format_formal_bias_report
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
        title = f"Sesión {args.session}"
    else:
        numbers = repo.get_all_numbers()
        title = "Todos los datos"
    
    if len(numbers) < 100:
        print(f"❌ Se necesitan al menos 100 spins para análisis de sesgo (tiene {len(numbers)})")
        return 1
    
    print(f"\n📊 Análisis de Sesgo - {title}")
    print(format_formal_bias_report(numbers))
    
    return 0


def backtest_cmd(args):
    from src.database import RouletteRepository
    from src.utils.backtesting import (
        flat_bet_backtest,
        walk_forward_optimization,
        create_bias_strategy,
        create_hot_numbers_strategy,
        format_backtest_report
    )
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
    else:
        numbers = repo.get_all_numbers()
    
    if len(numbers) < 200:
        print(f"❌ Se necesitan al menos 200 spins para backtesting (tiene {len(numbers)})")
        return 1
    
    strategy = args.strategy or "bias"
    initial_balance = args.balance or 1000.0
    bet_amount = args.bet or 1.0
    
    print(f"\n🎰 Backtesting - Estrategia: {strategy}")
    print(f"   Balance inicial: ${initial_balance:.2f}")
    print(f"   Apuesta por spin: ${bet_amount:.2f}")
    print(f"   Total spins: {len(numbers)}")
    
    if strategy == "bias":
        strategy_fn = create_bias_strategy(0.03)
    elif strategy == "hot":
        strategy_fn = create_hot_numbers_strategy(5)
    else:
        strategy_fn = create_bias_strategy(0.03)
    
    if args.walk_forward:
        print("\n⏳ Ejecutando Walk-Forward Optimization...")
        result = walk_forward_optimization(
            numbers,
            strategy_fn,
            training_window=args.train_window or 500,
            testing_window=args.test_window or 100,
            bet_amount=bet_amount,
            initial_balance=initial_balance
        )
        
        print(f"\n📈 Resultados Walk-Forward:")
        print(f"   Períodos de entrenamiento: {result.training_periods}")
        print(f"   Retorno in-sample: ${result.in_sample_return:.2f}")
        print(f"   Retorno out-of-sample: ${result.out_of_sample_return:.2f}")
        print(f"   Ratio de robustez: {result.robustness_ratio:.3f}")
        print("\n" + format_backtest_report(result.aggregate_backtest))
    else:
        train_data = numbers[:len(numbers)//2]
        test_data = numbers[len(numbers)//2:]
        
        bet_numbers = strategy_fn(train_data)
        
        if not bet_numbers:
            print("❌ La estrategia no encontró números para apostar")
            return 1
        
        print(f"\n🎯 Números seleccionados: {bet_numbers}")
        
        result = flat_bet_backtest(
            test_data,
            bet_numbers,
            bet_amount=bet_amount,
            initial_balance=initial_balance
        )
        
        print("\n" + format_backtest_report(result))
    
    return 0


def metrics_cmd(args):
    from src.database import RouletteRepository
    from src.utils.metrics import (
        compare_to_fair,
        find_anomalous_numbers,
        format_distribution_report,
        format_anomaly_report
    )
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
        title = f"Sesión {args.session}"
    else:
        numbers = repo.get_all_numbers()
        title = "Todos los datos"
    
    if len(numbers) < 50:
        print(f"❌ Se necesitan al menos 50 spins para métricas (tiene {len(numbers)})")
        return 1
    
    print(f"\n📊 Métricas de Distribución - {title}")
    
    comparison = compare_to_fair(numbers)
    print(format_distribution_report(comparison))
    
    anomalies = find_anomalous_numbers(numbers, sigma_threshold=args.sigma or 2.0)
    print(format_anomaly_report(anomalies))
    
    return 0


def kelly_cmd(args):
    from src.utils.backtesting import kelly_criterion, PAYOUTS, BetType
    
    win_prob = args.probability
    payout = args.payout or 35  # Default straight up
    fraction = args.fraction or 0.25
    bankroll = args.bankroll or 1000
    
    kelly_full = kelly_criterion(win_prob, payout, fraction=1.0)
    kelly_frac = kelly_criterion(win_prob, payout, fraction=fraction)
    
    fair_prob = 1 / 37
    expected_value = win_prob * payout - (1 - win_prob)
    house_edge = -expected_value * 100
    
    print(f"\n🎰 Kelly Criterion Calculator")
    print("=" * 50)
    print(f"Probabilidad de ganar: {win_prob*100:.2f}%")
    print(f"Probabilidad justa:    {fair_prob*100:.2f}%")
    print(f"Payout:                {payout}:1")
    print(f"Bankroll:              ${bankroll:.2f}")
    print()
    print(f"📊 Análisis:")
    print(f"   Valor esperado por $1: ${expected_value:.4f}")
    print(f"   House edge:            {house_edge:.2f}%")
    print()
    print(f"📈 Kelly Criterion:")
    print(f"   Full Kelly:            {kelly_full*100:.2f}% del bankroll")
    print(f"   Full Kelly apuesta:    ${kelly_full * bankroll:.2f}")
    print()
    print(f"   Fractional ({fraction*100:.0f}%):      {kelly_frac*100:.2f}% del bankroll")
    print(f"   Fractional apuesta:    ${kelly_frac * bankroll:.2f}")
    print()
    
    if expected_value > 0:
        print("✅ Apuesta con valor esperado positivo")
    else:
        print("❌ Apuesta con valor esperado negativo (la casa tiene ventaja)")
    
    print("\n⚠️  Nota: Salirrosas (2016) encontró que flat betting")
    print("   supera a Kelly en la práctica debido a menor varianza.")
    print("=" * 50)
    
    return 0


def statistics_cmd(args):
    from src.database import RouletteRepository
    from src.utils.statistics import summary_statistics, filter_profitable_numbers
    
    repo = RouletteRepository()
    
    if args.session:
        numbers = repo.get_numbers_by_session(args.session)
        title = f"Sesión {args.session}"
    else:
        numbers = repo.get_all_numbers()
        title = "Todos los datos"
    
    if len(numbers) < 100:
        print(f"❌ Se necesitan al menos 100 spins para estadísticas avanzadas (tiene {len(numbers)})")
        return 1
    
    stats = summary_statistics(numbers)
    
    print(f"\n📊 Estadísticas Avanzadas - {title}")
    print("=" * 60)
    print(f"Total spins: {stats['total_spins']}")
    print(f"Números únicos vistos: {stats['unique_numbers']}")
    print()
    
    print("─" * 60)
    print("CHI-SQUARE TEST")
    print("─" * 60)
    chi = stats['chi_square']
    print(f"Estadístico: {chi['statistic']:.3f}")
    print(f"P-value: {chi['p_value']:.6f}")
    print(f"Significancia: {chi['significance']}")
    print(f"¿Sesgada?: {'Sí' if chi['is_biased'] else 'No'}")
    print()
    
    print("─" * 60)
    print("ENTROPÍA")
    print("─" * 60)
    ent = stats['entropy']
    print(f"Entropía observada: {ent['observed_entropy']:.4f} bits")
    print(f"Entropía máxima: {ent['max_entropy']:.4f} bits")
    print(f"Uniformidad: {ent['uniformity']*100:.1f}%")
    print()
    
    profitable = filter_profitable_numbers(numbers, require_statistical_significance=False)
    
    print("─" * 60)
    print(f"NÚMEROS CON P >= 3% ({len(profitable)} encontrados)")
    print("─" * 60)
    for p in profitable[:10]:
        sig = "✅" if p.passes_threshold else "⚠️"
        print(f"  {sig} #{p.number:2d}: {p.observed_probability*100:.2f}% (ventaja: {p.advantage:+.2f}%)")
    
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RL Roulette - CLI de Gestión de Datos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python roulette_cli.py session create "Casino Córdoba"
  python roulette_cli.py live 1
  python roulette_cli.py predict --session 1
  python roulette_cli.py accuracy
  python roulette_cli.py stats --session 1
  python roulette_cli.py bias-test --session 1
  python roulette_cli.py backtest --strategy bias --walk-forward
  python roulette_cli.py metrics --session 1
  python roulette_cli.py kelly --probability 0.035 --payout 35
  python roulette_cli.py statistics --session 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos')
    
    create_parser = subparsers.add_parser('session', help='Gestión de sesiones')
    create_subparsers = create_parser.add_subparsers(dest='session_command')
    
    create_session = create_subparsers.add_parser('create', help='Crear nueva sesión')
    create_session.add_argument('name', nargs='?', help='Nombre de la sesión')
    create_session.add_argument('--casino', '-c', help='Nombre del casino')
    create_session.add_argument('--notes', '-n', help='Notas adicionales')
    
    create_subparsers.add_parser('list', help='Listar sesiones')
    
    live_parser = subparsers.add_parser('live', help='Modo live para ingresar números')
    live_parser.add_argument('session_id', type=int, help='ID de la sesión')
    
    predict_parser = subparsers.add_parser('predict', help='Modo predicción')
    predict_parser.add_argument('--session', '-s', dest='session_id', type=int, help='ID de sesión')
    predict_parser.add_argument('--model', '-m', help='Ruta al modelo')
    
    accuracy_parser = subparsers.add_parser('accuracy', help='Ver precisión de predicciones')
    accuracy_parser.add_argument('--session', '-s', dest='session_id', type=int, help='ID de sesión')
    
    add_parser = subparsers.add_parser('add', help='Agregar números manualmente')
    add_parser.add_argument('numbers', nargs='+', help='Números de ruleta (0-36)')
    add_parser.add_argument('--session', '-s', help='Nombre de sesión')
    
    import_parser = subparsers.add_parser('import', help='Importar desde CSV')
    import_parser.add_argument('file', help='Ruta del archivo CSV')
    import_parser.add_argument('--session', '-s', help='Nombre de sesión')
    
    subparsers.add_parser('sessions', help='Listar sesiones')
    subparsers.add_parser('list-sessions', help='Listar sesiones')
    
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas')
    stats_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analizar datos')
    analyze_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    
    export_parser = subparsers.add_parser('export', help='Exportar a CSV')
    export_parser.add_argument('output', help='Ruta del archivo de salida')
    export_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    
    clear_parser = subparsers.add_parser('clear', help='Limpiar base de datos')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirmar borrado')
    
    interactive_parser = subparsers.add_parser('interactive', help='Modo interactivo')
    interactive_parser.add_argument('--session', '-s', help='Nombre de sesión')
    
    bias_parser = subparsers.add_parser('bias-test', help='Análisis formal de sesgo (Salirrosas 2016)')
    bias_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting de estrategias')
    backtest_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    backtest_parser.add_argument('--strategy', choices=['bias', 'hot', 'cold'], default='bias', help='Estrategia')
    backtest_parser.add_argument('--balance', type=float, default=1000.0, help='Balance inicial')
    backtest_parser.add_argument('--bet', type=float, default=1.0, help='Monto por apuesta')
    backtest_parser.add_argument('--walk-forward', action='store_true', help='Usar walk-forward optimization')
    backtest_parser.add_argument('--train-window', type=int, default=500, help='Ventana de entrenamiento')
    backtest_parser.add_argument('--test-window', type=int, default=100, help='Ventana de testing')
    
    metrics_parser = subparsers.add_parser('metrics', help='Métricas de distribución (JSD, Wasserstein)')
    metrics_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    metrics_parser.add_argument('--sigma', type=float, default=2.0, help='Umbral sigma para anomalías')
    
    kelly_parser = subparsers.add_parser('kelly', help='Calculadora Kelly Criterion')
    kelly_parser.add_argument('--probability', '-p', type=float, required=True, help='Probabilidad de ganar (0-1)')
    kelly_parser.add_argument('--payout', type=int, default=35, help='Payout (default: 35 para straight up)')
    kelly_parser.add_argument('--fraction', type=float, default=0.25, help='Fracción de Kelly (default: 0.25)')
    kelly_parser.add_argument('--bankroll', type=float, default=1000, help='Bankroll total')
    
    statistics_parser = subparsers.add_parser('statistics', help='Estadísticas avanzadas (papers)')
    statistics_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == 'session':
        if args.session_command == 'create':
            return session_create(args)
        elif args.session_command == 'list':
            return session_list(args)
        else:
            create_parser.print_help()
            return 0
    elif args.command == 'live':
        return session_live(args)
    elif args.command == 'predict':
        return predict_mode(args)
    elif args.command == 'accuracy':
        return show_accuracy(args)
    elif args.command == 'add':
        return add_numbers(args)
    elif args.command == 'import':
        return import_csv_cmd(args)
    elif args.command in ['sessions', 'list-sessions']:
        return session_list(args)
    elif args.command in ['stats', 'analyze']:
        return show_stats(args)
    elif args.command == 'export':
        return export_data(args)
    elif args.command == 'clear':
        return clear_database(args)
    elif args.command == 'interactive':
        return interactive_mode(args)
    elif args.command == 'bias-test':
        return bias_test_cmd(args)
    elif args.command == 'backtest':
        return backtest_cmd(args)
    elif args.command == 'metrics':
        return metrics_cmd(args)
    elif args.command == 'kelly':
        return kelly_cmd(args)
    elif args.command == 'statistics':
        return statistics_cmd(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

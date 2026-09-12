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
    from src.database import RouletteRepository
    from src.engine.prediction_engine import PredictorType, PredictionEngine
    from src.utils.evaluation_harness import (
        LiveModelPrediction,
        make_fair_prediction,
        make_last_n_prediction,
        make_rolling_frequency_prediction,
    )

    repo = RouletteRepository()

    session_id = args.session_id
    if session_id:
        session = repo.db.get_session(session_id)
        if not session:
            print(f"Sesion {session_id} no encontrada")
            return 1
        history = repo.get_numbers_by_session(session_id)
    else:
        print('Selecciona una sesion con --session para conservar el orden temporal.')
        return 1

    if len(history) < 1:
        print("Se necesita al menos 1 numero para predecir")
        return 1

    engine = PredictionEngine(model_path=args.model)
    engine.load_history(history, session_id=session_id)

    bet_top_n = args.bet_top_n or 5
    last_n = args.last_n or 18

    print("\nModo Prediccion")
    print("=" * 60)
    print(f"Historial: {len(history)} numeros")
    print("Comandos: 'q' salir | 'p' nueva prediccion | 'a' accuracy")
    print("=" * 60)

    while True:
        try:
            engine_predictions = engine.predict_all()
            live_predictions = _build_live_predictions(
                engine,
                engine_predictions,
                history,
                bet_top_n,
                last_n,
                LiveModelPrediction,
                PredictorType,
                make_fair_prediction,
                make_rolling_frequency_prediction,
                make_last_n_prediction,
            )

            emitted = dict(engine_predictions)
            consensus = engine.get_consensus_prediction(engine_predictions)
            emitted[consensus.predictor] = consensus
            for name, prediction in live_predictions.items():
                if prediction is not None and name not in ('fair', 'consensus', 'bias'):
                    emitted[name] = engine.prediction_from_probabilities(name, prediction.number_probs)
            repo.db.emit_predictions(session_id, emitted, expected_count=len(history))
            decision = engine.get_policy_decision()
            print(f'DQN policy: {decision.action if decision.action is not None else decision.status.reason}')
            print(f"\n{'-' * 60}")
            print("PREDICCIONES:")
            print(_format_live_predictions(live_predictions, bet_top_n))
            if args.show_probs:
                print(_format_live_probabilities(live_predictions))
            print(f"{'-' * 60}")

            user_input = input("\nIngresa el numero que salio (o comando): ").strip().lower()

            if user_input == 'q':
                break
            if user_input == 'p':
                continue
            if user_input == 'a':
                accuracy = repo.get_prediction_accuracy(session_id)
                if accuracy.get("total", 0) == 0:
                    print("\nNo hay predicciones registradas aun")
                else:
                    print(f"\nPrecision registrada ({accuracy['total']} total):")
                    print(f"   Numero exacto: {accuracy['number']['correct']}/{accuracy['total']} ({accuracy['number']['pct']:.1f}%)")
                    print(f"   Color: {accuracy['color']['correct']}/{accuracy['total']} ({accuracy['color']['pct']:.1f}%)")
                    print(f"   Paridad: {accuracy['parity']['correct']}/{accuracy['total']} ({accuracy['parity']['pct']:.1f}%)")
                    print(f"   Alto/Bajo: {accuracy['high_low']['correct']}/{accuracy['total']} ({accuracy['high_low']['pct']:.1f}%)")
                    print(f"   Docena: {accuracy['dozen']['correct']}/{accuracy['total']} ({accuracy['dozen']['pct']:.1f}%)")
                    print(f"   Columna: {accuracy['column']['correct']}/{accuracy['total']} ({accuracy['column']['pct']:.1f}%)")
                continue

            try:
                actual = int(user_input)
            except ValueError:
                print("Entrada invalida")
                continue

            if not 0 <= actual <= 36:
                print("Numero invalido (debe ser 0-36)")
                continue

            if session_id:
                repo.add_number_to_session(session_id, actual)

            history.append(actual)
            engine.add_number(actual)

            print(f"\nResultado: {format_spin_display(actual)}")
            print(_format_live_hits(live_predictions, actual, bet_top_n))

        except KeyboardInterrupt:
            print("\n")
            break

    return 0


def _build_live_predictions(
    engine,
    engine_predictions,
    history,
    bet_top_n,
    last_n,
    live_prediction_cls,
    predictor_type_cls,
    make_fair_prediction,
    make_rolling_frequency_prediction,
    make_last_n_prediction,
):
    predictions = {}

    consensus = engine.get_consensus_prediction()
    consensus_name = consensus.predictor.value if consensus is not None else "consensus"
    predictions[consensus_name] = _live_prediction_from_full_prediction(
        consensus_name,
        consensus,
        bet_top_n,
        live_prediction_cls,
    )
    predictions["bias"] = _live_prediction_from_full_prediction(
        "bias",
        engine_predictions.get(predictor_type_cls.BIAS),
        bet_top_n,
        live_prediction_cls,
    )
    predictions["fair"] = make_fair_prediction(history, bet_top_n)
    predictions["rolling_frequency"] = make_rolling_frequency_prediction(history, bet_top_n)
    predictions["last_n"] = make_last_n_prediction(history, last_n, bet_top_n)

    return predictions


def _live_prediction_from_full_prediction(
    model,
    prediction,
    bet_top_n,
    live_prediction_cls,
):
    if prediction is None:
        return None

    number_probs = prediction.number.all_probabilities
    top_numbers = prediction.top_numbers or sorted(
        number_probs.items(),
        key=lambda item: (-item[1], item[0]),
    )[:bet_top_n]
    predicted = int(prediction.number.value)
    confidence = max(number_probs.values()) if number_probs else 0.0

    return live_prediction_cls(
        model=model,
        predicted=predicted,
        confidence=confidence,
        top_numbers=top_numbers[:bet_top_n],
        number_probs=number_probs,
    )


def _format_live_predictions(predictions, bet_top_n):
    lines = [
        f"{'model':<20} {'predicted':>9} {'conf':>8} top_numbers",
        "-" * 60,
    ]

    for model, prediction in predictions.items():
        if prediction is None:
            lines.append(f"{model:<20} {'inactive':>9} {'-':>8} -")
            continue

        top_numbers = ", ".join(
            f"{number}:{probability * 100:.1f}%"
            for number, probability in prediction.top_numbers[:bet_top_n]
        )
        lines.append(
            f"{model:<20} {prediction.predicted:>9} "
            f"{prediction.confidence * 100:>7.2f}% {top_numbers}"
        )

    return "\n".join(lines)


def _format_live_probabilities(predictions):
    lines = ["", "Full probability distributions:"]

    for model, prediction in predictions.items():
        if prediction is None:
            continue
        probs = ", ".join(
            f"{number}:{prediction.number_probs.get(number, 0.0) * 100:.2f}%"
            for number in range(37)
        )
        lines.append(f"{model}: {probs}")

    return "\n".join(lines)


def _format_live_hits(predictions, actual, bet_top_n):
    lines = ["", "Aciertos por modelo:"]

    for model, prediction in predictions.items():
        if prediction is None:
            lines.append(f"   {model:<20} inactive")
            continue

        top_numbers = [number for number, _ in prediction.top_numbers[:bet_top_n]]
        exact = "OK" if prediction.predicted == actual else "--"
        top_hit = "OK" if actual in top_numbers else "--"
        lines.append(
            f"   {model:<20} exact={exact} top{bet_top_n}={top_hit}"
        )

    return "\n".join(lines)


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
        create_last_n_bunching_strategy,
        format_backtest_report
    )
    
    repo = RouletteRepository()
    
    if args.session is None:
        from copy import copy
        outcomes = []
        for session in repo.get_sessions():
            selected = copy(args)
            selected.session = session['id']
            print(f"Sesion {selected.session}")
            outcomes.append(backtest_cmd(selected))
        return 0 if outcomes and any(code == 0 for code in outcomes) else 1
    numbers = repo.get_numbers_by_session(args.session)
    
    if len(numbers) < 200:
        print(f"❌ Se necesitan al menos 200 spins para backtesting (tiene {len(numbers)})")
        return 1
    
    strategy = args.strategy or "bias"
    initial_balance = args.balance
    bet_amount = args.bet
    
    print(f"\n🎰 Backtesting - Estrategia: {strategy}")
    print(f"   Balance inicial: ${initial_balance:.2f}")
    print(f"   Apuesta por spin: ${bet_amount:.2f}")
    print(f"   Total spins: {len(numbers)}")
    
    if strategy == "bias":
        strategy_fn = create_bias_strategy(0.03)
    elif strategy == "hot":
        strategy_fn = create_hot_numbers_strategy(5)
    elif strategy == "last-n":
        strategy_fn = create_last_n_bunching_strategy()
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


def heatmaps_cmd(args):
    from src.database import RouletteRepository
    from src.utils.heatmaps import HeatmapConfig, generate_heatmap_report

    repo = RouletteRepository()

    if args.session is None:
        from copy import copy
        from pathlib import Path
        outcomes = []
        for session in repo.get_sessions():
            selected = copy(args)
            selected.session = session['id']
            selected.output_dir = str(Path(args.output_dir) / f"session_{selected.session}")
            outcomes.append(heatmaps_cmd(selected))
        return 0 if outcomes and any(code == 0 for code in outcomes) else 1
    numbers = repo.get_numbers_by_session(args.session)
    source_label = f"session {args.session}"

    config = HeatmapConfig(
        window_size=args.window_size,
        step_size=args.step,
        sector_count=args.sector_count,
        output_dir=args.output_dir,
        show=args.show,
        z_threshold=args.z_threshold,
        fdr_alpha=args.fdr_alpha,
        include_anomalies=not args.no_anomalies,
    )
    session_numbers = None
    if args.include_drift and not args.session:
        session_numbers = {
            f"{session['id']}:{session['name']}": repo.get_numbers_by_session(session["id"])
            for session in repo.get_sessions()
        }

    try:
        result = generate_heatmap_report(
            numbers,
            config,
            source_label=source_label,
            session_numbers=session_numbers,
        )
    except ValueError as error:
        print(f"Heatmaps error: {error}")
        return 1

    print(f"\nHeatmaps - {result.source_label}")
    print("=" * 60)
    print(f"Total spins: {result.total_spins}")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")
    if result.anomalies:
        print("\nTop anomalies:")
        for anomaly in result.anomalies[:10]:
            print(
                f"- {anomaly.severity} {anomaly.window_label} "
                f"{anomaly.feature}: z={anomaly.z_score:+.2f}, "
                f"p={anomaly.p_value:.6f}, q={anomaly.q_value:.6f}, "
                f"fdr={'yes' if anomaly.fdr_significant else 'no'}, "
                f"observed={anomaly.observed_rate:.3f}, expected={anomaly.expected_rate:.3f}"
            )
    if result.drift_results:
        print("\nLargest session drift:")
        for drift in sorted(result.drift_results, key=lambda item: item.distance, reverse=True)[:5]:
            print(f"- {drift.session_a} vs {drift.session_b}: {drift.distance:.4f}")
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


def randomness_cmd(args):
    from src.database import RouletteRepository
    from src.utils.randomness import analyze_randomness, format_randomness_report

    sessions = RouletteRepository().get_evaluation_sessions()
    if args.session is not None:
        sessions = [session for session in sessions if session.session_id == str(args.session)]
    evaluated = 0
    for session in sessions:
        print(f'Sesion {session.session_id}: {len(session.numbers)} resultados')
        if len(session.numbers) < args.min_spins:
            print(f'Se necesitan al menos {args.min_spins} resultados por sesion.')
            continue
        report = analyze_randomness(
            session.numbers, window_size=args.window_size, step_size=args.step,
            alpha=args.alpha, markov_max_lag=args.markov_max_lag,
            resamples=args.resamples, seed=args.seed, fdr_method=args.fdr_method,
        )
        print(format_randomness_report(report))
        evaluated += 1
    return 0 if evaluated else 1


def monitor_cmd(args):
    import json
    from pathlib import Path
    from src.checkpoints import atomic_save
    from src.database import RouletteRepository
    from src.datasets import dataset_manifest
    from src.utils.bias_detection import WheelBiasAnalyzer

    try:
        family = tuple(str(value) for value in sorted(args.sessions))
        if len(set(family)) != len(family):
            raise ValueError('Las sesiones de la familia no pueden repetirse')
        repo = RouletteRepository()
        available = {session.session_id: session for session in repo.get_evaluation_sessions()}
        if any(session_id not in available for session_id in family):
            raise ValueError('Alguna sesion solicitada no existe')
        sessions = [available[session_id] for session_id in family]
        database_path = str(Path(repo.db.db_path).resolve())
        state_path = Path(args.state)
        if state_path.resolve() == Path(database_path):
            raise ValueError('El estado debe usar un archivo distinto de la base de datos')
        if args.output and Path(args.output).resolve() in (state_path.resolve(), Path(database_path)):
            raise ValueError('El informe debe usar un archivo distinto del estado y de la base de datos')
        saved = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        if saved is not None and (saved.get('schema_version') != 1 or saved.get('method') != 'roulette_monitor_family_v1'
                                  or saved.get('database_path') != database_path or saved.get('sessions') != list(family)
                                  or saved.get('alpha') != args.alpha):
            raise ValueError('El estado no coincide con la base, la familia o el presupuesto de error solicitado')
        analyzers = {}
        for session in sessions:
            if saved is None:
                analyzer = WheelBiasAnalyzer(stream_id=session.session_id, alpha=args.alpha,
                                             streams=family, descriptive_reports=False)
            else:
                record = saved['analyzers'][session.session_id]
                if record['source'] != session.source:
                    raise ValueError('Cambio el origen de una sesion monitoreada')
                analyzer = WheelBiasAnalyzer.from_state(record['state'])
                if (analyzer.stream_id != session.session_id or analyzer.budget.streams != family
                        or analyzer.budget.alpha != args.alpha or analyzer.descriptive_reports):
                    raise ValueError('El monitor no coincide con su familia de inferencia')
                previous = [tuple(item) for item in record['state']['seen_events']]
                prefix = list(zip(session.spin_ids, session.numbers))[:len(previous)]
                if previous != prefix:
                    raise ValueError('Los resultados previos cambiaron, se eliminaron o se reordenaron; el monitor requiere un historial que solo crezca')
            analyzers[session.session_id] = analyzer
        for session in sessions:
            analyzer = analyzers[session.session_id]
            if args.reset:
                analyzer.reset()
            for event_id, number in zip(session.spin_ids, session.numbers):
                analyzer.add_spin(number, event_id=event_id)
        state = {'schema_version': 1, 'method': 'roulette_monitor_family_v1', 'database_path': database_path,
                 'sessions': list(family), 'alpha': args.alpha,
                 'analyzers': {session.session_id: {'source': session.source, 'state': analyzers[session.session_id].to_state()}
                               for session in sessions}}
        report = {'schema_version': 1, 'dataset': dataset_manifest(sessions), 'alpha': args.alpha, 'sessions': {}}
        for session in sessions:
            analyzer = analyzers[session.session_id]
            report['sessions'][session.session_id] = dict(
                analyzer.sequential_evidence(),
                probability_bounds={str(number): analyzer.monitor.probability_bounds([number]) for number in range(37)},
                candidates_under_fixed_probability_model=analyzer.get_recommended_bets(),
            )
        atomic_save(state_path, state, neural=False)
        if args.output:
            atomic_save(args.output, report, neural=False)
        for session_id, result in report['sessions'].items():
            print(f"Sesion {session_id}: observaciones={result['observations']}; p secuencial={result['anytime_p_value']:.6g}; alpha asignado={result['allocated_alpha']:.6g}; rechazo={result['rejected']}")
        print('Las cotas requieren probabilidades condicionales constantes. Rechazar uniformidad no demuestra una ventaja rentable ni persistente.')
        print('El presupuesto cubre esta familia y sus reinicios. La configuracion y la seleccion de datos deben fijarse antes de observar los resultados.')
        print(f'Estado del monitor: {state_path}')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'No se pudo actualizar el monitor: {error}')
        return 1


def evaluate_cmd(args):
    from dataclasses import asdict
    import json
    from pathlib import Path
    from src.database import RouletteRepository
    from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward, format_evaluation_report

    sessions = RouletteRepository().get_evaluation_sessions()
    if args.session is not None:
        sessions = [session for session in sessions if session.session_id == str(args.session)]
    config = EvaluationConfig(
        training_window=args.train_window, testing_window=args.test_window,
        step_size=args.step, bet_top_n=args.bet_top_n, seed=args.seed, model_path=args.model,
        ece_bins=args.ece_bins, bootstrap_resamples=args.bootstrap_resamples,
        confidence_level=args.confidence_level, comparison_baseline=args.comparison_baseline,
        compute_intervals=not args.no_intervals, runs=args.runs, lstm_epochs=args.epochs,
        device=args.device, initial_bankroll=args.bankroll, unit_stake=args.unit_stake,
        models=tuple(args.models) + (('dqn',) if args.model and 'dqn' not in args.models else ()),
        block_length=args.block_length,
    )
    result = evaluate_walk_forward(sessions, config)
    print(format_evaluation_report(result))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(result), indent=2, allow_nan=False), encoding='utf-8')
        print(f'Resultados y manifiesto: {output}')
    return 0 if result.folds else 1


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
    from src.console import configure_console
    configure_console()
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
  python roulette_cli.py randomness --session 1
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
    predict_parser.add_argument('--bet-top-n', type=int, default=5, help='Cantidad de números top a mostrar')
    predict_parser.add_argument('--last-n', type=int, default=18, help='Ventana para baseline last_n')
    predict_parser.add_argument('--show-probs', action='store_true', help='Mostrar distribuciones completas')
    
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
    backtest_parser.add_argument('--strategy', choices=['bias', 'hot', 'last-n'], default='bias', help='Estrategia')
    backtest_parser.add_argument('--balance', type=float, default=1000.0, help='Balance inicial')
    backtest_parser.add_argument('--bet', type=float, default=1.0, help='Monto por apuesta')
    backtest_parser.add_argument('--walk-forward', action='store_true', help='Usar walk-forward optimization')
    backtest_parser.add_argument('--train-window', type=int, default=500, help='Ventana de entrenamiento')
    backtest_parser.add_argument('--test-window', type=int, default=100, help='Ventana de testing')
    
    metrics_parser = subparsers.add_parser('metrics', help='Métricas de distribución (JSD, Wasserstein)')
    metrics_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    metrics_parser.add_argument('--sigma', type=float, default=2.0, help='Umbral sigma para anomalías')

    heatmaps_parser = subparsers.add_parser('heatmaps', help='Generar heatmaps PNG')
    heatmaps_parser.add_argument('--session', '-s', type=int, help='ID de sesion')
    heatmaps_parser.add_argument('--output-dir', default='reports/heatmaps', help='Directorio de salida')
    heatmaps_parser.add_argument('--window-size', type=int, default=100, help='Tamano de ventana rolling')
    heatmaps_parser.add_argument('--step', type=int, default=25, help='Paso entre ventanas rolling')
    heatmaps_parser.add_argument('--sector-count', type=int, default=12, help='Cantidad de sectores de rueda')
    heatmaps_parser.add_argument('--z-threshold', type=float, default=2.5, help='Umbral z-score para alertas')
    heatmaps_parser.add_argument('--fdr-alpha', type=float, default=0.05, help='Alpha FDR para q-values de alertas')
    heatmaps_parser.add_argument('--include-drift', action='store_true', help='Generar drift entre sesiones en modo global')
    heatmaps_parser.add_argument('--no-anomalies', action='store_true', help='No generar resumen de anomalías')
    heatmaps_parser.add_argument('--show', action='store_true', help='Mostrar figuras al generarlas')

    randomness_parser = subparsers.add_parser('randomness', help='Tests de aleatoriedad e independencia')
    randomness_parser.add_argument('--session', '-s', type=int, help='ID de sesion')
    randomness_parser.add_argument('--window-size', type=int, default=100, help='Tamano de ventana para entropy drift')
    randomness_parser.add_argument('--step', type=int, default=50, help='Paso entre ventanas')
    randomness_parser.add_argument('--alpha', type=float, default=0.01, help='Nivel de significancia')
    randomness_parser.add_argument('--min-spins', type=int, default=50, help='Minimo de spins requerido')
    randomness_parser.add_argument('--resamples', type=int, default=9999)
    randomness_parser.add_argument('--seed', type=int, default=42)
    randomness_parser.add_argument('--fdr-method', choices=['by', 'bh'], default='by')
    randomness_parser.add_argument('--markov-max-lag', type=int, default=3, help='Lag maximo para scan Markov')

    monitor_parser = subparsers.add_parser('monitor', help='Monitoreo secuencial con estado persistente y presupuesto de error')
    monitor_parser.add_argument('--sessions', nargs='+', type=int, required=True, help='Familia de sesiones fijada antes del monitoreo')
    monitor_parser.add_argument('--state', required=True, help='Archivo JSON de estado; se reutiliza al continuar')
    monitor_parser.add_argument('--alpha', type=float, default=.05, help='Presupuesto total de error para la familia y sus reinicios')
    monitor_parser.add_argument('--reset', action='store_true', help='Iniciar otro segmento con menor presupuesto, conservando los eventos ya procesados')
    monitor_parser.add_argument('--output', help='Archivo JSON de evidencia, cotas y datos de referencia')

    evaluate_parser = subparsers.add_parser('evaluate', help='Walk-forward evaluation del motor completo')
    evaluate_parser.add_argument('--session', '-s', type=int, help='ID de sesión')
    evaluate_parser.add_argument('--train-window', type=int, default=500, help='Ventana de entrenamiento')
    evaluate_parser.add_argument('--test-window', type=int, default=100, help='Ventana de testing')
    evaluate_parser.add_argument('--step', type=int, default=100, help='Paso entre folds')
    evaluate_parser.add_argument('--bet-top-n', type=int, default=5, help='Cantidad de números apostados por predicción')
    evaluate_parser.add_argument('--seed', type=int, default=42, help='Seed para baseline random')
    evaluate_parser.add_argument('--model', help='Path opcional a modelo DQN')
    evaluate_parser.add_argument('--ece-bins', type=int, default=10, help='Bins para calibracion')
    evaluate_parser.add_argument('--bootstrap-resamples', type=int, default=2000, help='Remuestreos bootstrap')
    evaluate_parser.add_argument('--confidence-level', type=float, default=0.95, help='Nivel de confianza para intervalos')
    evaluate_parser.add_argument('--comparison-baseline', default='fair', help='Baseline para comparacion pareada')
    evaluate_parser.add_argument('--runs', type=int, default=5)
    evaluate_parser.add_argument('--epochs', type=int, default=30)
    evaluate_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluate_parser.add_argument('--bankroll', type=float, default=1000.0)
    evaluate_parser.add_argument('--unit-stake', type=float, default=1.0)
    evaluate_parser.add_argument('--models', nargs='*', choices=['lstm', 'extra_trees', 'bias', 'dqn'], default=['lstm', 'extra_trees', 'bias'])
    evaluate_parser.add_argument('--block-length', type=int)
    evaluate_parser.add_argument('--output', help='Archivo JSON con resultados, filas y manifiesto')
    evaluate_parser.add_argument('--no-intervals', action='store_true', help='Desactivar intervalos bootstrap')
    
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
    elif args.command == 'heatmaps':
        return heatmaps_cmd(args)
    elif args.command == 'randomness':
        return randomness_cmd(args)
    elif args.command == 'evaluate':
        return evaluate_cmd(args)
    elif args.command == 'monitor':
        return monitor_cmd(args)
    elif args.command == 'kelly':
        return kelly_cmd(args)
    elif args.command == 'statistics':
        return statistics_cmd(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

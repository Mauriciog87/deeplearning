# arXiv Paper Memory

Fecha de lectura: 2026-06-16

Revisión de implementación: 2026-09-12. Las decisiones siguientes distinguen inspiración, reproducción de fórmulas y adaptaciones. El harness usa ECE descriptivo por ejecución y cotas separadas para residuos condicionales en bins fijos; no implementa los intervalos poblacionales l2 de Sun et al. Los contratos, las verificaciones y los resultados están en [el plan de implementación](RESEARCH_IMPLEMENTATION_PLAN.md).

Objetivo: usar papers como memoria tecnica para mejorar el motor de ruleta sin vender prediccion magica. La prioridad actual es diagnostico estadistico, deteccion de drift, calibracion y evaluacion robusta. Los modelos pesados quedan fuera hasta que exista volumen de datos y un baseline clasico fuerte.

Fuentes usadas: PDFs de `https://arxiv.org/pdf/{id}`, metadata de arXiv API, y busquedas puntuales en arXiv para titulos cuando el PDF no extrajo metadata limpia.

## Decision Summary

Implementado:
- Walk-forward por sesión, liquidación de las 47 acciones, PASS, probabilidades simultáneas y separación entre sesgo y rentabilidad.
- ECE por ejecución, bootstrap temporal pareado, cotas de residuos condicionales y diagnóstico conjunto. No se reproducen el KDE de Posocco ni los intervalos l2 de Sun; `positive_bootstrap_fraction` tampoco es la probabilidad de mejora de Agarwal.
- Evidencia multinomial persistente y comparación KT categórica. Las alertas batch usan BY por defecto; no son un monitor secuencial ni una implementación de e-GAI.
- LSTM categórico por defecto y representación ordinal como ablación explícita.

Experimentos implementados, sin promoción automática:
- E-SR/e-CUSUM, CTM con apuestas lineales corregidas y PITMonitor.
- Temperature scaling, MCLLO regularizado e isotonic normalizado con particiones cronológicas.
- Recalibración online con base finita, comparación Brier/2 y residuos medidos del optimizador.
- Benchmark sintético reproducible con incertidumbre y resultados desfavorables conservados.

Exclusiones vigentes:
- IPS/DR no hacen falta para replay pasivo con todos los resultados observados y liquidación conocida. Se reconsideran si las acciones afectan los resultados o su observación.
- TimesFM, Chronos y MOMENT requieren evidencia de señal útil y una comparación que justifique su costo; no se incorporan por ahora.
- La predicción física requiere medidas de la rueda y la bola que este dataset no contiene. SMECE requiere etiquetas probabilísticas; aquí se observan números 0–36.

## `1609.09601` - Biased Roulette Wheel: A Quantitative Trading Strategy Approach

Decision: implement_now.

Lectura tecnica: el paper trata una rueda sesgada como una estrategia cuantitativa: recolecta 10,980 spins reales, estima desviaciones de probabilidad, usa backtesting y walk-forward optimization, y compara staking plano contra Kelly. La idea central no es "predecir" un spin aislado, sino explotar desviaciones persistentes bajo una disciplina de evaluacion temporal.

Revision de implementacion: la referencia a Ornstein-Uhlenbeck no demuestra que exista memoria predictiva. Una frecuencia movil de ventana W sobre indicadores IID tiene correlacion de orden uno (W-1)/W y coeficiente de regresion de deriva 1/W por el solapamiento. El helper ahora muestra ese nulo y aclara que extrapola la frecuencia de la proxima ventana, no la probabilidad del proximo giro. Las sesiones conservan su identidad.

Aplicacion ahora: mantener walk-forward, heatmaps rolling, drift entre sesiones, ROI y drawdown. No usar Kelly como default porque el propio paper muestra que flat betting puede ser mas robusto en corto plazo.

Descartar por ahora: apuestas agresivas basadas en maxima probabilidad puntual sin intervalos.

Decision implementada: se comparan las 47 acciones con liquidacion comun y PASS. Una apuesta a pleno necesita p>1/36 para tener ganancia esperada positiva; p>1/37 solo supera la referencia uniforme. Los filtros heredados usan intervalos binomiales exactos con ajuste Bonferroni para 37 numeros en una muestra fija IID. El monitor secuencial usa proyecciones de una region conjunta bajo probabilidades condicionales constantes. Ninguna de estas condiciones prueba que una ventaja persista.

## `1204.6412` - Predicting the outcome of roulette

Decision: reference_only.

Lectura tecnica: el paper modela la fisica de la bola y la rueda. La senal predictiva viene de posicion inicial, velocidad y aceleracion; con conteo mecanico o camara, los autores reportan ventaja esperada positiva. El metodo es causal/fisico, no estadistico sobre una lista final de numeros.

Supuestos: acceso al proceso fisico antes de que cierre la apuesta, medicion precisa y misma rueda. En este repo solo hay resultados 0-36 despues del spin, por lo que el paper no justifica entrenar un modelo secuencial sobre numeros crudos.

Aplicacion ahora: usarlo como advertencia. Si alguna vez hay captura de video/velocidad, crear otro pipeline. Para el motor actual, solo sirve para separar "sesgo fisico medible" de "patrones ilusorios en resultados".

## `2108.13264` - Deep Reinforcement Learning at the Edge of the Statistical Precipice

Decision: implement_now.

Lectura tecnica: el paper muestra que comparar medias puntuales en RL con pocas corridas puede cambiar conclusiones. Recomienda intervalos, comparaciones pareadas, distribuciones de rendimiento y probabilidad de mejora. La leccion principal para ruleta es que un modelo con ROI medio mayor no debe aceptarse si el intervalo cruza cero o si la comparacion no esta pareada por los mismos folds/spins.

Supuestos: resultados ruidosos, muestra finita y muchos metodos comparados. En ruleta esto es mas severo porque la ventaja esperada justa es negativa y los eventos raros dominan el ROI.

Aplicación actual: bootstrap de bloques temporales dentro de cada sesión y remuestreo de ejecuciones, con los mismos índices para todos los métodos y semillas. Se comparan diferencias pareadas, drawdown y exposición. Los intervalos dependen de la estabilidad temporal requerida por ese remuestreo; no son una garantía universal del paper. `positive_bootstrap_fraction` es la fracción de diferencias bootstrap positivas, no una probabilidad posterior ni la métrica de probabilidad de mejora de Agarwal.

## `2011.04102` - Reliable Off-policy Evaluation for Reinforcement Learning

Decision: future_experiment.

Lectura tecnica: propone estimaciones robustas/optimistas de recompensa para OPE con incertidumbre distribucional. Es relevante cuando una politica objetivo se evalua con datos generados por otra politica, sin desplegarla.

Supuestos: el problema general tiene recompensas parcialmente observadas y políticas de comportamiento. En el replay pasivo de este repo se observa el resultado y se conoce la liquidación de cualquier acción; se puede calcular cada recompensa contrafactual sin propensiones, mientras apostar no altere el proceso ni la observación.

Aplicación actual: evaluación cronológica de información completa, con bankroll y PASS. No implementar IPS/DR para este caso. Una aplicación donde las acciones afecten lo observado necesitaría otro contrato de identificación y logging.

## `2212.06355` - A Review of Off-Policy Evaluation in Reinforcement Learning

Decision: future_experiment.

Lectura técnica: revisa direct method, importance sampling, doubly robust, eficiencia y propiedades estadísticas. Las necesidades de logging dependen de qué recompensas son observables y de cómo las acciones afectan el proceso.

Aplicación actual: el reporte separa replay pasivo de información completa y aplicaciones con observación dependiente de la acción. La ausencia de propensiones no impide el primero. Ningún backtest establece que una ventaja histórica persista al desplegar la política.

## `2011.14359` - Optimal Mixture Weights for Off-Policy Evaluation with Multiple Behavior Policies

Decision: future_experiment.

Lectura tecnica: estudia como mezclar estimadores generados por multiples politicas de comportamiento para reducir varianza. La parte util es conceptual: sesiones o casinos distintos no deben mezclarse ingenuamente si su proceso generador cambia.

Supuestos: cada fuente tiene estimador aproximadamente insesgado y se conoce su procedencia. En este repo, las sesiones si existen, pero no siempre codifican mesa/casino/condiciones suficientes.

Aplicacion ahora: drift heatmap entre sesiones y advertencias al mezclar datos. Futuro: pesos por sesion solo despues de medir estabilidad y tamano efectivo.

## `2109.03480` - Estimating Expected Calibration Errors

Decision: implement_now.

Lectura tecnica: compara estimadores de ECE y muestra que la evaluacion de calibracion depende de binning, reliability diagrams y sesgos del estimador. Para modelos 0-36, la calibracion importa mas que el top-1: una distribucion puede acertar poco pero estar bien calibrada si sus probabilidades son honestas.

Supuestos: clasificacion probabilistica con scores comparables. Encaja directamente con predictores que devuelven `all_probabilities`.

Aplicacion actual: mantener ECE descriptivo por bins, junto a log loss y Brier. El estimador KDE del paper no esta implementado. La calibracion por clase no equivale a calibracion conjunta del vector completo.

## `2408.08998` - A Confidence Interval for the l2 Expected Calibration Error

Decision: implement_now.

Lectura tecnica: desarrolla intervalos para ECE l2 y cubre top-1-to-k calibration, incluyendo confidence calibration y full calibration. La idea que importa aqui es que ECE tambien necesita incertidumbre, no solo un numero.

Supuestos: el estimador corregido y sus intervalos requieren el diseno y las condiciones del paper. Tener suficientes muestras por bin no justifica reemplazarlos por un bootstrap percentil de ECE L1. Esa sustitucion fallo incluso para un predictor uniforme perfectamente calibrado.

Aplicacion actual: retirar esos intervalos bootstrap y ofrecer cotas conservadoras para un objetivo diferente y explicito: residuos condicionales acumulados en bins fijos. El ECE adaptativo sigue siendo descriptivo. La masa del conjunto top-k tampoco equivale a la calibracion vectorial top-1-to-k del paper.

## `2603.14092` - Soft Mean Expected Calibration Error (SMECE)

Decision: reference_only.

Lectura tecnica: SMECE corrige el caso donde los labels son probabilisticos, no binarios. El paper advierte que forzar labels suaves a outcomes duros puede medir otra cosa.

Supuestos: etiquetas como distribuciones o probabilidades subjetivas. El resultado de ruleta es un numero observado, por lo que los labels son duros.

Aplicacion ahora: no implementar SMECE. Futuro: si se incorporan labels probabilisticos de OCR/confianza o consenso humano, revisar.

## `1706.03415` - Inductive Conformal Martingales for Change-Point Detection

Decision: implement_now.

Lectura tecnica: detecta cambios en streams sin conocer el modelo de cambio, asumiendo i.i.d. bajo el regimen nulo. El martingale conformal es potente pero mas complejo que lo necesario para el primer paso del motor.

Supuestos: stream y referencia de intercambioabilidad. En ruleta, la hipotesis nula natural es frecuencia justa e independencia; cambios de rueda/sesion rompen esto.

Aplicación actual: conservar las ventanas como diagnóstico descriptivo y usar contratos separados para los monitores secuenciales. `ReferenceConditionalMonitor` y `PITMonitor` son experimentos opcionales basados en las referencias posteriores descritas abajo; no se atribuye su implementación exacta a este paper.

## `2602.13848` - Testing For Distribution Shifts with Conditional Conformal Test Martingales

Decision: implement_now.

Lectura tecnica: propone tests secuenciales de shift contra una referencia fija y corrige error de estimacion de referencia finita. Es relevante porque el motor compara streams nuevos contra fair/reference.

Supuestos: referencia fija y muestra entrante secuencial. En el repo, la referencia fair 1/37 es conocida, pero las sesiones reales pueden ser referencia empirica.

Aplicación actual: `ReferenceConditionalMonitor` usa una referencia congelada, una banda DKW y la apuesta lineal corregida de la ecuación 7. Mezcla siete apuestas fijas; no reproduce la actualización smoothed ONS del paper. El presupuesto total incluye el fallo de la banda de referencia. Requiere referencia y observaciones IID de la misma distribución bajo el nulo, y tiene potencia limitada frente a cambios que no desplazan suficientemente la media del PIT empírico. Las pruebas verifican fórmula, átomos, replay y falsas alarmas.

## `2407.07290` - Causal Discovery-Driven Change Point Detection in Time Series

Decision: implement_now.

Lectura tecnica: usa estructura causal y divergencia condicional para detectar cambios en componentes de series multivariadas. La leccion practica es no mirar solo un agregado global: color, paridad, docenas, columnas y sectores pueden cambiar distinto.

Supuestos: variables multivariadas y relaciones causales. En ruleta estas features son derivadas deterministas del numero, no causas independientes.

Aplicacion ahora: alertas por feature derivada, no solo heatmap por numero. Descartar causal discovery para este dominio.

## `2212.00173` - SPADE: Semi-supervised Anomaly Detection under Distribution Mismatch

Decision: implement_now.

Lectura tecnica: SPADE combina pseudo-labeling y ensembles para anomalias con mismatch entre datos etiquetados y no etiquetados. La parte transferible es la cautela ante mismatch: una anomalia en una sesion no debe entrenar automaticamente el detector global.

Supuestos: samples tabulares/imagen, etiquetas parciales y one-class classifiers. El motor tiene secuencias categoricas pequenas, por lo que SPADE completo es excesivo.

Aplicacion ahora: detector simple, interpretable y por ventana. Futuro: ensembles solo si hay muchas sesiones y labels de mesa defectuosa.

## `2403.18716` - Statistical testing of random number generators and their improvement using randomness extraction

Decision: implement_now.

Lectura tecnica: enfatiza que los tests estadisticos no prueban aleatoriedad definitiva, pero son una herramienta practica y parametrizable. Usa baterias de tests y compara RNGs/post-procesamiento.

Supuestos: secuencias largas y tests multiples. Para ruleta, no hay bits crudos de RNG, pero si se pueden testear uniformidad, independencia, runs, transiciones y entropia rolling.

Aplicacion ahora: CLI `randomness` con tests reproducibles y p-values. No afirmar que un p-value alto prueba que la rueda es justa.

## `2001.11838` - The time-adaptive statistical testing for random number generators

Decision: implement_now.

Lectura tecnica: propone usar baterias adaptativas: correr tests baratos primero y dedicar mas tiempo a los mas prometedores. La idea reduce el costo de baterias grandes sin perder potencia practica.

Supuestos: muchas pruebas posibles y secuencias de longitud variable. En el repo, conviene empezar con tests baratos y claros antes de NIST/TestU01.

Aplicacion ahora: incluir tests secuenciales ligeros y mostrar advertencias por muestra corta. Futuro: bateria adaptativa real si se exportan secuencias largas.

## `2310.10688` - A Decoder-Only Foundation Model for Time-Series Forecasting

Decision: future_experiment.

Lectura tecnica: TimesFM preentrena un decoder-only grande para forecasting zero-shot. Su senal esperada es continuidad temporal en series reales, no eventos discretos i.i.d. 0-36.

Supuestos: series numericas con patrones, estacionalidad o dependencia. La ruleta justa no debe tener estructura predictiva.

Aplicacion ahora: no usar sobre numeros crudos. Futuro: probar solo sobre features agregadas por ventana, como entropia, residual maximo o drift, y comparar contra ARIMA/rolling mean.

## `2403.07815` - Chronos: Learning the Language of Time Series

Decision: future_experiment.

Lectura tecnica: Chronos tokeniza valores de series y entrena arquitecturas tipo T5 para forecasting probabilistico. Es fuerte cuando hay series con dinamica real o transfer learning util.

Supuestos: continuidad temporal y datos preentrenados relevantes. La secuencia de ruleta no comparte esos supuestos si la rueda es justa.

Aplicacion ahora: no usar. Futuro: si se modela una serie de metricas rolling, Chronos puede compararse como baseline externo, no como predictor de proximo numero.

## `2402.03885` - MOMENT: A Family of Open Time-series Foundation Models

Decision: future_experiment.

Lectura tecnica: MOMENT aborda pretraining de modelos abiertos para multiples tareas de series temporales y destaca el costo/dificultad de benchmarks. Requiere infraestructura y evaluacion cuidadosa.

Supuestos: datasets grandes y tareas de forecasting/anomaly/classification. El repo no tiene volumen ni benchmark TSFM.

Aplicacion ahora: no integrar. Futuro: usar embeddings sobre features rolling si hay evidencia de drift real.

## `2412.19286` - Time Series Foundational Models: Their Role in Anomaly Detection and Prediction

Decision: future_experiment.

Lectura tecnica: evalua TSFM para anomalias/prediccion y advierte que modelos estadisticos o deep learning simples igualan o superan TSFM en varios escenarios, especialmente sin patrones claros. Esto refuerza una regla conservadora: primero baseline simple.

Supuestos: datasets de anomalias y forecasting con recursos computacionales altos. En ruleta, el riesgo de sobreajuste es mayor que el beneficio esperado.

Aplicacion ahora: descartar TSFM en el camino critico. Reabrir solo con un protocolo de benchmark, datos suficientes y baseline clasico fuerte.

## Implementation Decisions From Reading

Implementado: diagnósticos batch de aleatoriedad y heatmaps con supuestos explícitos; inferencia secuencial separada; evaluación temporal pareada; calibración marginal y diagnóstico conjunto; políticas de información completa; comparadores de recalibración y cambio opcionales. Las secciones siguientes y el plan identifican las adaptaciones y sus pruebas.

Se mantienen las exclusiones del resumen: predicción física sin medidas, TSFM sin señal demostrada y métodos IPS/DR innecesarios para el contrato pasivo actual. Ningún detector activa por sí solo una estrategia agresiva.

## Second-Wave arXiv Search - 2026-06-16

Objetivo: cubrir huecos que quedaron fuera pero son importantes para el proyecto: OPE/bandits, control de falsos positivos, cambio online de distribucion, calibracion multiclase y riesgo de staking.

### `2311.14110` - When is Off-Policy Evaluation Useful in Contextual Bandits?

Decision: read_next.

Lectura: DataCOPE estudia si un dataset de bandits permite evaluar una política. El problema de recompensas parcialmente observadas no coincide con la liquidación pasiva de ruleta.

Aplicación actual: el reporte OPE readiness conserva los requisitos de IPS/DR/SWITCH para aplicaciones dependientes de la acción y aclara por qué no son necesarios para el replay de información completa actual. No se implementa DataCOPE.

### `1612.01205` - Optimal and Adaptive Off-policy Evaluation in Contextual Bandits

Decision: read_next.

Lectura: compara IPS, DR y SWITCH para contextual bandits, con foco en tradeoff bias-varianza y dificultad cuando no hay reward model consistente.

Implementacion ahora: no implementar IPS/DR sin propensiones. Si se agregan logs de apuestas, el siguiente paso es un modulo OPE separado con chequeo de overlap.

### `1511.03722` - Doubly Robust Off-policy Value Evaluation for Reinforcement Learning

Decision: read_next.

Lectura: extiende DR a decision secuencial y lo ubica como pieza de safe policy improvement. El valor para el proyecto es metodologico: no evaluar politicas de apuesta nuevas solo con datos generados por otra politica sin modelar esa diferencia.

Aplicación actual: separar replay pasivo y evaluación de trayectorias afectadas por acciones. No se implementa el estimador DR secuencial; no aporta identificación adicional cuando se conocen todas las recompensas contrafactuales.

### `1801.04756` - A Binning Approach to Quickest Change Detection with Unknown Post-Change Distribution

Decision: implement_now.

Lectura: detecta cambios secuenciales cuando el pre-change es conocido y el post-change desconocido, usando bins y conteos. Ruleta 0-36 ya es un sistema de bins naturales.

Implementacion ahora: agregar test de cambio categorial por ventanas izquierda/derecha al comando `randomness`.

### `2506.01452` - e-GAI for Online FDR Control

Decision: implement_now.

Lectura: e-GAI controla FDR online bajo los requisitos de validez condicional de sus e-values. Un ajuste batch de p-values no reproduce ese algoritmo ni permite mirar repetidamente sin costo.

Aplicación actual: BY por defecto para familias batch dependientes; BH queda como opción con sus supuestos. Para seguimiento persistente se usa evidencia multinomial y un presupuesto explícito por stream/reinicio. Es control de probabilidad de falsas alarmas para esa familia, no una implementación de e-GAI ni una promesa de FDR online adaptativo.

### `1706.05378` - Multi-A/B Testing with Online FDR Control

Decision: future_experiment.

Lectura: combina multi-armed bandits y online FDR para tests monitoreados continuamente. Relevante si el sistema prueba politicas de apuesta en vivo.

Implementacion ahora: no. Requiere eventos online y definicion clara de hipotesis por politica.

### `2602.18573` - Multiclass Calibration Assessment and Recalibration via Linear Log Odds

Decision: implemented_experiment.

Lectura: propone evaluar y recalibrar modelos multiclase sin acceso interno. Encaja con predictores black-box 0-36.

Aplicación actual: `ProbabilityCalibrator('mcllo')` ajusta pendientes e interceptos de log odds respecto de una clase de referencia, con parámetros acotados y regularización hacia identidad. Es una adaptación de la parametrización del paper. El harness reserva una partición cronológica de calibración y exporta estados y resultados del optimizador. ECE por clase y masa top-k no se presentan como full calibration.

### `2512.09054` - Normalization-Aware Isotonic Multiclass Calibration

Decision: implemented_experiment.

Lectura: corrige problemas de isotonic one-vs-rest al respetar normalizacion multiclase. Es prometedor para recalibracion de distribuciones 0-36.

Aplicación actual: `ProbabilityCalibrator('normalized_isotonic')` usa la pérdida multiclase normalizada de la ecuación 4, con bloques PAVA fijos y valores positivos. En coordenadas logarítmicas es un problema convexo con restricciones lineales. Se fija una escala y un dominio finito, y se reporta una cota de suboptimalidad numérica contrastada con un programa lineal independiente. No se reproduce el algoritmo MCMC ni se certifica el óptimo sobre todas las funciones isotónicas. La derivación está en el plan; la búsqueda no estableció que sea un resultado nuevo.

### `1511.02339` - Markov Chain Order Estimation with Conditional Mutual Information

Decision: implement_now.

Lectura: estima orden Markov en secuencias simbolicas con tests de informacion condicional. Para ruleta, esto pregunta si hay dependencia temporal real.

Implementacion ahora: agregar scan simple de dependencia por lag en `randomness`, con p-values ajustados.

### `1710.01787` - On Kelly Betting: Some Limitations

Decision: reference_only.

Lectura: muestra limitaciones del criterio Kelly cuando importan aproximaciones y drawdown. Relevante para comunicar riesgo, no para predecir numeros.

Implementacion ahora: mantener Kelly como calculadora y no como estrategia default automatica.

## Investigación y auditoría adicional - 2026-09-12

### Evidencia multinomial y secuencias de confianza

Fuentes: [Lindon y Malek, `2011.03567`](https://arxiv.org/abs/2011.03567), [Ryu y Wornell, `2402.03683`](https://arxiv.org/abs/2402.03683).

`MultinomialMonitor` mezcla tres marginales Dirichlet contra un nulo categórico declarado. Las actualizaciones coinciden con la razón de verosimilitudes expresada mediante funciones gamma. Se invierte la región conjunta para proyectar probabilidades de números y subconjuntos; su cobertura requiere un vector de probabilidades condicionales constante. El comparador KT usa solamente Dirichlet(1/2), el caso categórico de la sección 3.1 de Ryu y Wornell. No reproduce su construcción general para vectores acotados.

El monitor conserva IDs de eventos, valida el replay y distribuye el presupuesto entre streams y reinicios. Se integra con el analyzer, el CLI `monitor`, la política de valor esperado conservador y los checkpoints HH. `test_sequential_inference.py` y `test_sequential_analyzer.py` verifican fórmulas, proyecciones, cobertura bajo el nulo, persistencia y rechazo de historia alterada. La ausencia de alarma no certifica una rueda justa.

### Calibración conjunta y cotas secuenciales

Fuentes: [Vaicenavicius et al.](https://proceedings.mlr.press/v89/vaicenavicius19a.html), [Pinelis, teorema 3.5](https://arxiv.org/abs/1208.2200v2).

El contraejemplo conjunto usa seis permutaciones de un vector con tres clases activas: los errores de confianza y por clase pueden ser cero aunque la distribución condicional del vector sea incorrecta. `JointCalibrationMonitor` agrega un diagnóstico de variación total por celdas del vector completo y una razón de verosimilitudes con alternativa predecible. Promedia procesos entre ejecuciones, sin multiplicar evidencia por semillas que comparten resultados. Su nulo es corrección predictiva condicional dado el pasado y el pronóstico, más fuerte que calibración condicionada solo al pronóstico. Es una alternativa explícita, no una reproducción del test omnibus de Vaicenavicius.

Las cotas de `calibration.py` usan la desigualdad maximal para martingalas en espacio de Hilbert de Pinelis y un presupuesto sumable sobre horizontes diádicos. Cubren residuos condicionales acumulados en bins fijos, no ECE poblacional. `test_joint_calibration.py` y `test_calibration_contract.py` cubren el contraejemplo, réplicas de semillas, fórmulas, dependencia y vectores perfectamente calibrados.

### Detectores de cambio opcionales

Fuentes: [Shin, Ramdas y Rinaldo, `2203.03532`](https://arxiv.org/abs/2203.03532), [CTM, `2602.13848`](https://arxiv.org/abs/2602.13848), [PITMonitor, `2603.13156`](https://arxiv.org/abs/2603.13156).

`CategoricalEDetector` implementa e-SR y e-CUSUM con alternativas que aumentan la probabilidad de un número. El umbral A controla longitud media hasta una falsa alarma; no controla la probabilidad de alarmar alguna vez. El benchmark fija A=H/alpha para usar la consecuencia finita P(T≤H)≤alpha.

El CTM usa la adaptación con referencia finita descrita arriba. PITMonitor usa rangos secuenciales aleatorizados, apuestas de histograma y una mezcla sobre tiempos de inicio que conserva la masa aún no iniciada. Requiere PITs IID bajo el nulo; marginales estacionarias con dependencia no bastan. La localización posterior a la alarma usa el comparador Dirichlet(1/2) del paper y no es un intervalo de confianza. Un cambio hacia mejor calibración también puede disparar el monitor. `test_change_detection.py` verifica fórmulas, ties, persistencia, desplazamientos y simulaciones nulas.

### Recalibración online

Fuente: [Marx, Kuleshov y Ermon, `2409.19157`](https://arxiv.org/abs/2409.19157).

`OnlineRecalibrator` adapta el enfoque Blackwell/ORCA a una base RBF finita y a regret de Brier/2 contra un pronóstico base. El optimizador evalúa el peor payoff de los 37 resultados antes de observar la etiqueta. Las violaciones positivas de la condición de semiespacio se conservan y entran en la cota residual; una respuesta del solver no se interpreta como certificado por sí sola. La cota controla la base elegida y no implica calibración completa o ausencia de regret si persisten errores del oráculo.

El estado se mantiene entre folds y se separa por sesión y ejecución. `test_online_recalibration.py` verifica gradientes, secuencia forecast/update, cotas con optimización deliberadamente limitada y replay. El método se habilita explícitamente en `evaluate` o `research`.

### Decisiones de representación y validación

La revisión de [Double DQN, `1509.06461`](https://arxiv.org/abs/1509.06461) confirma que `DQNAgent.train` selecciona la siguiente acción con la red online y la evalúa con la red target. Aplica la máscara de acciones, suprime el bootstrap al terminar el episodio y lo conserva al truncar. `test_agent_state.py` verifica estos contratos y la continuación tras guardar/cargar. Esa corrección del algoritmo no implica que haya señal aprendible en una rueda justa.

Los números de bolsillo son categorías, por lo que el LSTM usa one-hot como representación predeterminada. La codificación ordinal previa permanece disponible como ablación y en sus checkpoints antiguos. `test_lstm_representation.py` comprueba entrenamiento y continuación exacta de ambas; la elección de representación no demuestra una mejora de predicción.

El benchmark registra configuración, semillas, hashes de datos/código y mediciones por ensayo para ocho escenarios. Las diferencias entre métodos son pareadas sobre los mismos resultados. Los intervalos Monte Carlo binomiales son exactos; los de medias entre ensayos usan una aproximación Student-t y no son simultáneos entre todas las comparaciones. Las campañas y sus limitaciones se registran en el plan. Los datos sintéticos y las políticas oráculo no establecen una ventaja sobre ruedas reales.

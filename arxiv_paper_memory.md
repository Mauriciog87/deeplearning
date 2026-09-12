# arXiv Paper Memory

Fecha de lectura: 2026-06-16

Revision de implementacion: 2026-09-12. Las decisiones de lectura distinguen inspiracion de reproduccion del metodo. El harness actual usa ECE descriptivo por ejecucion y cotas separadas para residuos condicionales en bins fijos; no implementa los intervalos poblacionales l2 de Sun et al. El contrato y los avances estan en `RESEARCH_IMPLEMENTATION_PLAN.md`.

Objetivo: usar papers como memoria tecnica para mejorar el motor de ruleta sin vender prediccion magica. La prioridad actual es diagnostico estadistico, deteccion de drift, calibracion y evaluacion robusta. Los modelos pesados quedan fuera hasta que exista volumen de datos y un baseline clasico fuerte.

Fuentes usadas: PDFs de `https://arxiv.org/pdf/{id}`, metadata de arXiv API, y busquedas puntuales en arXiv para titulos cuando el PDF no extrajo metadata limpia.

## Decision Summary

implement_now:
- `1609.09601`: walk-forward, backtesting y sesgo como trading cuantitativo.
- `2108.13264`: bootstrap temporal adaptado y comparaciones pareadas. La fraccion de remuestreos positivos no es la metrica de probabilidad de mejora del paper.
- `2109.03480`: ECE con bins/reliability y cuidado con estimadores.
- `2408.08998`: referencia sobre sesgo y limites de la inferencia de calibracion; sus intervalos l2 no estan reproducidos.
- `1706.03415`, `2602.13848`, `2407.07290`: usar inspiracion de cambio de distribucion, pero con ventanas y tests simples.
- `2212.00173`: usar idea de robustez ante mismatch, no SPADE completo.
- `2403.18716`, `2001.11838`: suite de tests de aleatoriedad reproducible.

future_experiment:
- `2011.04102`, `2212.06355`, `2011.14359`: OPE formal si el motor empieza a comparar politicas de apuesta offline.
- `2310.10688`, `2403.07815`, `2402.03885`, `2412.19286`: TSFM solo sobre features agregadas y con mucha mas data.

reference_only:
- `1204.6412`: ruleta fisica predictiva requiere sensores/velocidad/posicion, no aplica al dataset de numeros crudos.
- `2603.14092`: SMECE es util conceptualmente para labels probabilisticos, pero aqui los outcomes son etiquetas duras 0-36.

## `1609.09601` - Biased Roulette Wheel: A Quantitative Trading Strategy Approach

Decision: implement_now.

Lectura tecnica: el paper trata una rueda sesgada como una estrategia cuantitativa: recolecta 10,980 spins reales, estima desviaciones de probabilidad, usa backtesting y walk-forward optimization, y compara staking plano contra Kelly. La idea central no es "predecir" un spin aislado, sino explotar desviaciones persistentes bajo una disciplina de evaluacion temporal.

Supuestos: las probabilidades no son exactamente estacionarias; el paper menciona comportamiento tipo Ornstein-Uhlenbeck, por lo que el sesgo puede revertir. Esto encaja con ruleta 0-36 solo si los datos vienen de la misma mesa/condicion y se evita mezclar sesiones incompatibles.

Aplicacion ahora: mantener walk-forward, heatmaps rolling, drift entre sesiones, ROI y drawdown. No usar Kelly como default porque el propio paper muestra que flat betting puede ser mas robusto en corto plazo.

Descartar por ahora: apuestas agresivas basadas en maxima probabilidad puntual sin intervalos.

## `1204.6412` - Predicting the outcome of roulette

Decision: reference_only.

Lectura tecnica: el paper modela la fisica de la bola y la rueda. La senal predictiva viene de posicion inicial, velocidad y aceleracion; con conteo mecanico o camara, los autores reportan ventaja esperada positiva. El metodo es causal/fisico, no estadistico sobre una lista final de numeros.

Supuestos: acceso al proceso fisico antes de que cierre la apuesta, medicion precisa y misma rueda. En este repo solo hay resultados 0-36 despues del spin, por lo que el paper no justifica entrenar un modelo secuencial sobre numeros crudos.

Aplicacion ahora: usarlo como advertencia. Si alguna vez hay captura de video/velocidad, crear otro pipeline. Para el motor actual, solo sirve para separar "sesgo fisico medible" de "patrones ilusorios en resultados".

## `2108.13264` - Deep Reinforcement Learning at the Edge of the Statistical Precipice

Decision: implement_now.

Lectura tecnica: el paper muestra que comparar medias puntuales en RL con pocas corridas puede cambiar conclusiones. Recomienda intervalos, comparaciones pareadas, distribuciones de rendimiento y probabilidad de mejora. La leccion principal para ruleta es que un modelo con ROI medio mayor no debe aceptarse si el intervalo cruza cero o si la comparacion no esta pareada por los mismos folds/spins.

Supuestos: resultados ruidosos, muestra finita y muchos metodos comparados. En ruleta esto es mas severo porque la ventaja esperada justa es negativa y los eventos raros dominan el ROI.

Aplicacion ahora: bootstrap por filas/folds, baseline fair, comparacion pareada por indice, P(mejora), drawdown y reglas anti-leakage en walk-forward.

## `2011.04102` - Reliable Off-policy Evaluation for Reinforcement Learning

Decision: future_experiment.

Lectura tecnica: propone estimaciones robustas/optimistas de recompensa para OPE con incertidumbre distribucional. Es relevante cuando una politica objetivo se evalua con datos generados por otra politica, sin desplegarla.

Supuestos: hay trayectorias, politicas de comportamiento, recompensas y estructura secuencial. El motor actual puede simular apuestas sobre historial, pero no tiene logs ricos de politica de comportamiento ni propensities.

Aplicacion ahora: no implementar OPE formal. La accion practica es dejar el evaluate walk-forward como metodo principal y no aceptar modelos si no superan baselines en datos retenidos.

## `2212.06355` - A Review of Off-Policy Evaluation in Reinforcement Learning

Decision: future_experiment.

Lectura tecnica: revisa OPE, direct method, importance sampling, doubly robust, eficiencia y propiedades estadisticas. Sirve como mapa para no inventar evaluacion offline sin propensities.

Supuestos: MDP/contextual bandit con politicas observables. En ruleta, las acciones de apuesta del usuario y la politica que genero los datos no estan suficientemente registradas.

Aplicacion ahora: reporte de evaluacion debe ser explicito: esto es backtest walk-forward, no prueba causal de una politica desplegada. Futuro: guardar accion, stake, odds, bankroll y propensities si se quiere OPE real.

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

Aplicacion ahora: implementar ventanas rolling, z-scores, p-values y drift entre sesiones. Futuro: martingales conformales si hay suficientes spins por mesa.

## `2602.13848` - Testing For Distribution Shifts with Conditional Conformal Test Martingales

Decision: implement_now.

Lectura tecnica: propone tests secuenciales de shift contra una referencia fija y corrige error de estimacion de referencia finita. Es relevante porque el motor compara streams nuevos contra fair/reference.

Supuestos: referencia fija y muestra entrante secuencial. En el repo, la referencia fair 1/37 es conocida, pero las sesiones reales pueden ser referencia empirica.

Aplicacion ahora: reportar p-values por ventanas y no tratar la referencia empirica como perfecta. Futuro: CTM si el sistema opera online con alertas continuas.

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

Ahora:
- Agregar suite `randomness` con uniformidad, runs, autocorrelacion serial, transiciones y drift de entropia.
- Calibracion: ECE por ejecucion para confianza, clases y conjuntos top-k; cotas separadas para residuos condicionales en bins fijos. La calibracion conjunta requiere otro diagnostico.
- Agregar p-values a alertas de heatmap.
- Mantener evaluacion walk-forward, comparaciones pareadas y bootstrap.

Futuro:
- OPE con propensities si se registran politicas de apuesta.
- Martingales conformales si hay monitoreo online largo por mesa.
- TSFM solo sobre features derivadas por ventana.

Descartado:
- Prediccion fisica sin sensores.
- Modelos foundation sobre numeros crudos.
- Staking agresivo sin intervalos y drawdown.

## Second-Wave arXiv Search - 2026-06-16

Objetivo: cubrir huecos que quedaron fuera pero son importantes para el proyecto: OPE/bandits, control de falsos positivos, cambio online de distribucion, calibracion multiclase y riesgo de staking.

### `2311.14110` - When is Off-Policy Evaluation Useful in Contextual Bandits?

Decision: read_next.

Lectura: DataCOPE pregunta si un dataset offline permite evaluar una politica objetivo antes de desplegarla. Esto es directamente relevante porque el repo hoy puede hacer walk-forward, pero no puede afirmar OPE causal sin logs de accion y propensiones.

Implementacion ahora: agregar un diagnostico de OPE readiness al reporte de evaluacion. Debe marcar que el nivel actual soportado es backtest walk-forward pareado y listar campos faltantes para IPS/DR/SWITCH.

### `1612.01205` - Optimal and Adaptive Off-policy Evaluation in Contextual Bandits

Decision: read_next.

Lectura: compara IPS, DR y SWITCH para contextual bandits, con foco en tradeoff bias-varianza y dificultad cuando no hay reward model consistente.

Implementacion ahora: no implementar IPS/DR sin propensiones. Si se agregan logs de apuestas, el siguiente paso es un modulo OPE separado con chequeo de overlap.

### `1511.03722` - Doubly Robust Off-policy Value Evaluation for Reinforcement Learning

Decision: read_next.

Lectura: extiende DR a decision secuencial y lo ubica como pieza de safe policy improvement. El valor para el proyecto es metodologico: no evaluar politicas de apuesta nuevas solo con datos generados por otra politica sin modelar esa diferencia.

Implementacion ahora: reporte OPE readiness y separacion explicita entre walk-forward backtest y OPE formal.

### `1801.04756` - A Binning Approach to Quickest Change Detection with Unknown Post-Change Distribution

Decision: implement_now.

Lectura: detecta cambios secuenciales cuando el pre-change es conocido y el post-change desconocido, usando bins y conteos. Ruleta 0-36 ya es un sistema de bins naturales.

Implementacion ahora: agregar test de cambio categorial por ventanas izquierda/derecha al comando `randomness`.

### `2506.01452` - e-GAI for Online FDR Control

Decision: implement_now.

Lectura: el problema es controlar falsos descubrimientos en una secuencia de tests online. El repo ahora genera muchos p-values, por lo que necesita q-values/FDR aunque no implemente e-GAI completo.

Implementacion ahora: Benjamini-Hochberg para reportes batch de randomness y heatmap anomalies. Futuro: alpha-investing/e-values si hay alertas live continuas.

### `1706.05378` - Multi-A/B Testing with Online FDR Control

Decision: future_experiment.

Lectura: combina multi-armed bandits y online FDR para tests monitoreados continuamente. Relevante si el sistema prueba politicas de apuesta en vivo.

Implementacion ahora: no. Requiere eventos online y definicion clara de hipotesis por politica.

### `2602.18573` - Multiclass Calibration Assessment and Recalibration via Linear Log Odds

Decision: future_experiment.

Lectura: propone evaluar y recalibrar modelos multiclase sin acceso interno. Encaja con predictores black-box 0-36.

Implementacion ahora: ya hay full ECE y top-k ECE. No recalibrar aun sin validation set estable.

### `2512.09054` - Normalization-Aware Isotonic Multiclass Calibration

Decision: future_experiment.

Lectura: corrige problemas de isotonic one-vs-rest al respetar normalizacion multiclase. Es prometedor para recalibracion de distribuciones 0-36.

Implementacion ahora: no. Primero medir calibracion y guardar splits.

### `1511.02339` - Markov Chain Order Estimation with Conditional Mutual Information

Decision: implement_now.

Lectura: estima orden Markov en secuencias simbolicas con tests de informacion condicional. Para ruleta, esto pregunta si hay dependencia temporal real.

Implementacion ahora: agregar scan simple de dependencia por lag en `randomness`, con p-values ajustados.

### `1710.01787` - On Kelly Betting: Some Limitations

Decision: reference_only.

Lectura: muestra limitaciones del criterio Kelly cuando importan aproximaciones y drawdown. Relevante para comunicar riesgo, no para predecir numeros.

Implementacion ahora: mantener Kelly como calculadora y no como estrategia default automatica.

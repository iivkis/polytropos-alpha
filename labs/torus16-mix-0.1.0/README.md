# Torus16-Mix 0.1.0

Лаборатория-потребитель публичного API [VeriNeural](../../crates/verineural/). Содержит оператор Torus16-Mix, рекуррентную EMA4-LM и экспериментальное обучение. Все вычислительные графы строятся из общих примитивов Semantic Neural IR; лаборатория не содержит альтернативного CUDA/Lean backend.

**Текущая разработка:** экспериментальный переход к Torus16-Mix 0.2.0 описан в [V0.2_TRANSITION.md](V0.2_TRANSITION.md). Это пока не отдельный выпущенный пакет или доказательство достижения целевого BPC. Зафиксированные результаты и ограничения измерений: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md). Старые подробные записи README доступны в истории Git; этот файл описывает актуальное назначение и запуск, а не хронологию каждого benchmark.

## Оператор и модель

Размерность оператора `d=48`: восемь ролей, по два входных циклических базиса на роль, одна обучаемая физическая матрица `W: [48,48]` на блок и параметризованные `row`, `col`, `bias`, `mix_gate`, `mix_bias`. Роли применяют `W` или `W.T` к смешанным и переставленным активациям; полные виртуальные матрицы не создаются.

```text
r = (7*role+1) mod 48
c = (11*role+3) mod 48
g = sigmoid(sum(x * mix_gate[role]) / sqrt(48) + mix_bias[role])
z = x * col[role]
z' = g * roll(z, -c) + (1-g) * roll(z, -(c+17))
v = z' @ (W.T if role even else W)
out = roll(v, +r) * row[role] + bias[role]
```

`src/lib.rs` определяет оператор, `src/ema4.rs` собирает streaming-модель из четырёх рекуррентных блоков с EMA, LayerNorm, causal Conv9, SwiGLU и tied embedding/head. При словаре из 104 символов исходный TorusShared EMA4 имеет 24 908 обучаемых параметров; standalone Torus16 operator содержит 3 848. Эти параметры относятся к разным моделям.

Параметрические варианты `TorusShared`, `Independent` и `RoleDelta` (rank 2/4/8) используются для контролируемых сравнений. Низкоранговая коррекция Role-Delta не материализует полную матрицу каждой роли. Обучаемые матрицы, рекуррентные состояния, граф и forward/reference должны оставаться согласованными с текущими Rust-контрактами, а не с историческими снимками документации.

## Исходники и данные

| Путь | Ответственность |
| --- | --- |
| `src/lib.rs` | Torus16 projection и публичная композиция |
| `src/ema4.rs` | EMA4 model builder и режимы матриц |
| `src/ema4_reference.rs` | Независимый численный оракул |
| `src/ema4_checkpoint.rs` | Checkpoint/resume и fingerprint архитектуры |
| `src/corpus.rs` | Подготовка корпуса и словаря |
| `examples/ema4_bptt.rs` | Экспериментальное full-parameter truncated BPTT |
| `data/corpus_snapshot.txt` | Зафиксированный UTF-8 исследовательский корпус |
| `tests/` | Проверки модели, численной семантики и GPU parity |

Corpus snapshot доступен локально, без сетевых загрузок. Разбиение train/validation/test, длины оценки и seed необходимо фиксировать для каждого сравнения. Исторический test-фрагмент из 512 символов не является независимым окончательным измерением качества.

## Команды из корневого Taskfile

```sh
# Оператор и Semantic IR
task lab-torus16-mix
task inspect-ir-torus16-mix
task test-lab-torus16-mix

# Forward и модель EMA4
task lab-torus16-ema4
task test-lab-torus16-ema4
task lab-torus16-ema4-gpu
task verify-gpu-torus16-ema4

# GPU parity и границы сертификата
task lab-torus16-mix-gpu
task verify-gpu-torus16-mix
task certify-lab-torus16-mix

# Truncated BPTT, checkpoints и Role-Delta
task test-lab-torus16-ema4-checkpoint
task train-lab-torus16-ema4-bptt
task verify-torus16-role-delta
task compare-torus16-role-delta-ranks

# P0: доступное длительное обучение (BPTT4 по умолчанию)
task verify-torus16-p0
task train-torus16-p0
TORUS16_P0_PHASE=decay task train-torus16-p0

# Опционально: отдельный BPTT8 на тех же бюджетах символов
TORUS16_P0_BPTT=8 task train-torus16-p0

# Исторический benchmarking v0.2 (не P0)
task benchmark-torus16-v02-horizon
```

Команды выполняются разработчиком в `nix develop`, как предписывает [AGENTS.md](../../AGENTS.md). GPU-задачи требуют NVIDIA, CUDA Driver и совместимого `VERINEURAL_CUDA_ARCH`. Некоторые сценарии требуют подготовленного baseline checkpoint; его путь настраивается через `TORUS16_ROLE_DELTA_SOURCE`. Актуальные переменные и проверки находятся **только** в [Taskfile.yml](../../Taskfile.yml), не дублируются здесь.

**Текущий план:** BPTT16/32 больше не используются для достижения NLL <= 1.5 на RTX 5060. Исторический `benchmark-torus16-v02-bptt16` оставлен как CPU preflight, но не является рекомендованным этапом. P0 использует BPTT4/8, отдельные каталоги checkpoint, двухэтапный LR, validation 8192 символов и закрытый test. Для второй фазы сначала должен завершиться `base`; оба запуска должны использовать одинаковые `TORUS16_P0_*` бюджеты, seed и corpus. Данные 512-символьного исторического test нельзя использовать для отбора конфигураций. Детали, границы доказательств и критерии приёмки см. в [плане перехода](V0.2_TRANSITION.md).

## Граница верификации

Результаты `GpuExecution` и CUDA Graph проверяются численно относительно независимых Rust reference. Это **не** доказательство IEEE-754, LLVM/PTX → SASS, аппаратного исполнения, сходимости обучения или глобальных свойств сети. Строгий путь сертификации Torus16 с экспонентой обязан отклонять неподдерживаемые операции; явный approximate-exp runtime остаётся экспериментальным. Не объявлять успешную проверку GPU/Lean без настоящего запуска.

Детальная история архитектурных изменений, latency/throughput, checkpoint parity, сравнений rank и графовых ограничений хранится в [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) и [V0.2_TRANSITION.md](V0.2_TRANSITION.md), а формальные границы библиотеки описаны в [SPEC.md](../../docs/verineural/SPEC.md).

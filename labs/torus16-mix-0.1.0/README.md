# Torus16-Mix 0.1.0

Лаборатория-потребитель публичного API VeriNeural. Источник математики:
исследовательский handoff от 2026-10-25, файл
projects/virtual16_recurrent_operator/operator16.py, режим torus16_mix.
Версия 0.1.0 сохраняет автономный оператор, а модуль `ema4` теперь
собирает также 4-layer streaming EMA4-LM из тех же примитивов.
Отдельное обучение и оценка BPC пока не реализованы.
Не сравнивать эти данные с качеством обученной модели из handoff.

## Семантика

d=48, 8 уникальных ролей, по 2 входных базиса на роль. Одна обучаемая
матрица W формы [48,48]. В каждой роли есть row, col, bias, mix_gate формы
[48] и mix_bias (скаляр). Начальное значение mix_bias=3, gate-весов=0.

~~~
r = (7*role+1) mod 48
c = (11*role+3) mod 48
g = sigmoid(sum(x * mix_gate[role]) / sqrt(48) + mix_bias[role])
z = x * col[role]
z' = g * roll(z, -c) + (1-g) * roll(z, -(c+17))
v = z' @ (W.T if role even else W)
out = roll(v, +r) * row[role] + bias[role]
~~~

roll относится к циклической перестановке **активаций** по последней
оси. Ни одна из 16 виртуальных матриц проекций не материализуется.
Одно умножение на W (или W.T) на роль, а не два. Упомянутые в исходном
README 16 базисов означают 8 пар, не 16 шагов по временной фазе.
Всего у standalone оператора 3848 обучаемых параметров; у полного
обученного EMA-LM из исходного handoff было 24 908. Числа не смешивать.

Вход/выход одной роли имеет форму [1,48]. architecture_model()
выдаёт единый Model с восемью выходами role_0..role_7, одной матрицей
weight, именованными ролями параметров и константами. Внутри IR
вычисляются gate, оба сдвига, mix, shared matmul и affine head. Хост
не выбирает базис за IR. Срезы + concat реализуют roll при постоянном shift.

## Как запускать в devShell

~~~sh
task lab-torus16-mix
task inspect-ir-torus16-mix
task test-lab-torus16-mix
~~~

Первый режим печатает структуру IR и один диагностический Tensor host
reference; второй экспортирует канонический Semantic IR в stdout.
Третий включает независимый scalar oracle 8 ролей x 4 режима gate,
проверку named bindings, тесты поддерживаемых GPU-профилей и
негативную проверку strict-exp compile.

Для аппаратного исполнения NVIDIA:
 
~~~sh
task lab-torus16-mix-gpu
task verify-gpu-torus16-mix
~~~

Первый режим использует `GpuEngine::prepare_model`, подаёт 46 именованных
исходных тензоров и вычисляет все 8 выходов в **одном CUDA Graph**.
Второй проводит независимую проверку GPU с ненулевыми `mix_gate`,
неравномерными `row/col` и разными `mix_bias`; сравнение с Rust
`Tensor` reference имеет абсолютный порог 5e-5 плюс относительный
2e-5. Переменная `VERINEURAL_CUDA_ARCH` позволяет выбрать `sm_86`
или другое совместимое с устройством значение (по умолчанию `sm_52`).
Ни PyTorch, ни host routing/rotation matrix, ни второй backend
вычислений в лабораторию не добавлены.

~~~sh
task certify-lab-torus16-mix
~~~

Эта команда **обязана завершаться ошибкой**, а не выпускать фиктивный
сертификат. Доступны два различных численных режима:

- **strict** (по умолчанию): RNE для Add/Mul/Sub/Div и проверяемая
  structural address policy. Строгая экспонента **не понижена на GPU**,
  поэтому весь Torus16-Mix в этом режиме не компилируется.
- **experimental executable** (явное `with_approximate_exp()`):
  используем `ex2.approx.f32` с отдельно округлённым умножением
  на `log2(e)`. Компилятор допускает только версионированную
  подпись `verineural.exp@1` с точной AST-формой и фиксирует её в
  translation plan. Неизвестные AST всё ещё fail-closed.

Такая экспонента не равна гарантированно `Rust f32::exp` бит-в-бит.
План и PTX audit проверяют разрешённые инструкции, а не реальную
верхнюю границу погрешности для всех F32. Проверка на устройстве
является **численным тестом**, не сертификатом IEEE-754, компиляции
или SASS. `certify` применяет стандартный строгий путь и Lean
по-прежнему отклоняет неподдерживаемую идеальную математику
экспоненты/деления.

Для BroadcastTo и SliceAxis уже есть структурный GPU-профиль
[1,1] -> [1,N], [1,N] SliceAxis(1) и Concat(1) на [1,N].
Остальные формы fail-closed. Ни своего CUDA, ни своего Lean backend,
ни CPU fallback в GPU runner нет.

**Граница лаборатории:** это готовый на уровне кода операторный
forward, не полная обученная EMA4-LM. Не реализованы Conv9,
LayerNorm, 4-layer recurrent state, SwiGLU, embedding/head,
backpropagation, AdamW, оценка BPC и VRAM benchmarks. Число
`24_908` из handoff характеризует **другую, более крупную сеть**;
оно не является количеством обучаемых параметров текущего среза.
Валидация кода на настоящем GPU обязательна до статуса принятого
артефакта.

PyTorch-паритет **намеренно не реализован**, как и требовалось.


## EMA4-LM streaming architecture

Файл `src/ema4.rs` собирает **полную структуру вычисления одного
токена**, соответствующую исследовательскому `OperatorBlock.step`:

- Размерность 48; 4 блока, по 8 ролей Torus16-Mix в каждом;
  физическая матрица W одна **на блок**, виртуальные веса не хранятся.
- По две LayerNorm и две causal depthwise Conv9 на блок.
- EMA: `s[t] = sigmoid(decay_logit)*s[t-1] +
  (1-sigmoid(decay_logit))*sigmoid(role1)*role2`.
- Gate role0, readout role3, два входных/два выходных SwiGLU-оператора
  в ролях 4..7 и повторное использование ролей 0,1 для down.
- One-hot `[1,V]` → обучаемое `[V,48]` embedding → 4 блока
  → final LayerNorm → `[1,V]` tied embedding/head.
- `ModelBuilder::state` / `next_state` определяют 68
  потоковых state tensors: 4 EMA + 4 * 2 * 8 Conv9 history vectors.
  Никакого растущего KV-кэша или скрытого host recurrence.
- При `V=104` ровно **24 908 trainable параметров** (формула
  подсчитана для тех же групп параметров, что у исследовательской сети).
  Нормировки, операторы и Conv9 имеют собственные веса каждого блока.
  Встроенная `Ema4Weights::deterministic` создаёт **тестовые, а не
  обученные** значения. Модуль `ema4_reference` является только
  независимым численным оракулом, не отдельным backend.

```sh
task lab-torus16-ema4
task test-lab-torus16-ema4
task lab-torus16-ema4-gpu
task verify-gpu-torus16-ema4
```

В GPU-режиме один вызов `prepare_model` компилирует общую архитектуру;
дальше `ModelExecution.step` обновляет 68 state buffers на устройстве,
а `set_input` заменяет токены без пересборки. На каждом шаге логиты и
память сравниваются со скалярным Rust reference. Это **реальный
CUDA Graph forward на одном токене за шаг**, а не обучающий запуск.

Для LayerNorm добавлен общий pointwise `verineural.sqrt@1` с
исполняемым `sqrt.rn.f32`, без фиктивного доказательства положительности
произвольных промежуточных значений. Строгая FP32-модель экспоненты
не реализована; обычный `Exp` в полном forward требует явного
`with_approximate_exp()`. Численный прогон не означает Lean/IEEE/SASS
сертификат. Универсальные GPU-операции живут в VeriNeural; вся
EMA4-LM остаётся в `labs/`.

**Не готово:** autograd, optimizer/AdamW, обучение, реальные веса
из PyTorch checkpoint, BPC/качество, батч T>1, GPU throughput и
сопоставление с исследовательской обученной моделью. Четырёхслойный
потоковый inference и её обученное качество — разные вещи.


## Экспериментальное обучение: reverse-mode IR + AdamW

Новый `task train-lab-torus16-ema4` выполняет конкретный CUDA
training vertical slice: один запуск GPU forward → стабилизированный
softmax/cross-entropy и seed dL/dlogits на host → повторный CUDA Graph
для сгенерированного IR-backward → host AdamW → обновление `final_beta`
через `GpuExecution::update_input` без перекомпиляции.
Демонстрационный словарь по умолчанию V=16, teacher-forced токен 0,
target 1, восемь шагов. Выводим training CE и *игрушечный* BPC
(`loss/ln(2)`), **не результат на исследовательском корпусе**.

Общий `verineural::differentiate` не содержит сетевых компонентов.
Он дифференцирует проверенный фрагмент Semantic Neural IR с
fan-out accumulation: Add, Mul, MatMul, BiasAdd, Reshape,
Permute(2D), ReverseAxis(2D), SliceAxis(1), Concat(1),
BroadcastTo([1,N]) и versioned Pointwise Sub/Div/Exp/Sqrt.
Возвращает граф с явным seed-входом и источниками числовых констант.
Неподдерживаемые операции, включая ReLU derivative в этой версии,
отклоняются. Непосредственно NVIDIA backend использует **тот же**
GPU-компилятор и CUDA Graph, без собственной GPU-математики в лаборатории.

`verineural::AdamW` с именованными `Tensor<f32>` хранит моменты,
делает bias correction и decoupled decay; проверяет весь набор
градиентов и финитность **до** мутации. На сегодня параметрический
update исполняется на host; не называем его GPU optimizer.

Текущий эксперимент учит только `final_beta` (48 значений).
Он соответствует **градиенту одного токена** при фиксированных
входных state buffers, то есть TBPTT=1. Это не настоящий
многошаговый BPTT, не обновление всех 24 908 весов, не
обучение исходной EMA4-LM и не проверка сходимости.
`task autodiff-stage` и `task verify-gpu-autodiff` проверяют
универсальные правила, в том числе аналитику против finite differences.
Все новые тесты после добавления подлежат отдельной локальной проверке.


## Full-parameter finite-window BPTT and holdout BPC (experimental)

The architecture-neutral library API is now verineural::unroll_model.
It turns a Model with named Input/Parameter/State/next_state roles into
a single pure time-expanded Semantic Neural IR graph. All timesteps use
ONE shared source per parameter, and every next_state is connected to the
following step. The scalar objective is sum over timesteps of
dot(logits[t], explicit_cotangent[t]); stable softmax/CE is computed by
the lab and supplies these cotangents. Calling differentiate over the
shared 24,908 trainable values includes the recurrent gradients across
the **entire chosen window**. Initial state is detached at each window
boundary: this is truncated BPTT, not infinite-horizon differentiation.

Command sequence:

    task temporal-bptt-stage
    task verify-gpu-temporal-bptt

    task train-lab-torus16-ema4-bptt

The exact research handoff UTF-8 snapshot is checked in at
`labs/torus16-mix-0.1.0/data/corpus_snapshot.txt` (SHA-256:
`c1d0509878099a43c85866c2f582be04fe8253292b137a83c58ebd04fb16ca85`).
No download, unpacking, `TORUS16_CORPUS` or shell setup is needed.
The example resolves the path relative to `CARGO_MANIFEST_DIR`,
not the current working directory. Optional Fish overrides for a
different run: `set -gx TORUS16_BPTT 2`,
`set -gx TORUS16_TRAIN_WINDOWS 2`,
`set -gx TORUS16_VAL_LIMIT 64`; provide
`set -gx TORUS16_CORPUS /absolute/path/to/other.txt` only when
intentionally changing the experiment.

Corpus must be an existing UTF-8 file. Alphabet is the sorted set of
Unicode scalar values in the entire file. The first 90% of characters
are train-only, the final 10% is strict chronological holdout, following
the research data preparation. For the original 524,859-character handoff
snapshot this gives a vocabulary of 104, 472,373 train symbols and
52,486 validation symbols. Changing source text changes the experiment;
nothing is downloaded or guessed silently.

Training uses one persistent, compiled CUDA Graph for multi-timestep
forward and reverse VJP, host stable CE and host AdamW across **all
physical trainable parameters**. Constant numerical source tensors are
excluded from optimization. In the demo, train windows are consecutive;
between windows the 68 final recurrent state tensors are snapshotted to
host memory BEFORE any initial state buffer is overwritten, then loaded
into the next window. This avoids unsafe overlapping CUDA feedback and
preserves values while cutting derivative edges at window boundaries.
One window computes one averaged loss and optimizer update.

After training, a separate streaming ModelExecution is prepared using
the updated parameters. It **resets recurrent state** and scores
next-character prediction exclusively on the final-10%-corpus holdout,
for at most TORUS16_VAL_LIMIT positions. Scored count is printed; this
bounded holdout BPC is not equivalent to the handoff's full validation
benchmark and MUST NOT be compared numerically to its trained checkpoint
BPC without matching all evaluation details. The deterministic default
weights are untrained and the original PyTorch checkpoint is not imported.

Known limitations and trusted boundaries:
- Reverse-mode rules are validated structurally/numerically in a bounded
  fragment, not formally certified GPU derivatives or exact IEEE float.
- Pointwise exp2.approx is explicitly opted in. Mathematical derivative
  of exp does not establish equivalence to NVIDIA's approximation.
- This is an experimental giant graph: compiling all trainable VJPs may
  require substantial GPU/host memory and time. No performance claims.
- GPU training implements CE/AdamW on host and CUDA forward/VJP. It does
  NOT implement a fused GPU optimizer, global loss graph or a general
  recurrent training scheduler.
- Default BPTT=2, train windows=2, holdout scored positions=64. The user
  must increase them for a meaningful training run. There is no guarantee
  the short smoke test improves heldout BPC.
- Reproducible checkpoint export/load and multi-seed studies remain open.


## Experimental GPU Mul → Add fusion (architecturally neutral)

`GpuLaunchConfig::with_pointwise_fusion()` opts into an exact-order,
bounded GPU compiler rewrite for an exclusive pointwise `Mul → Add` chain.
The semantic IR is **not rewritten**, and all selected outputs remain
available. A candidate is rejected unless the producer is used exactly
once by the selected Add, both operations and all operands have the
same checked shape, and the producer is not separately observable.

The CUDA kernel reads three ordinary operands, executes explicit
`mul.rn.f32` followed by `add.rn.f32` in the original operand order
(no `fma.rn.f32`), and stores the Add result. The fused producer
is removed from the actual CUDA Graph launch/memory contracts.
`TranslationKind::MulAdd` validates the two-operation LLVM template,
operand order and PTX instruction counts. The compiler's
`GpuCapabilityPlan` records the removed producer and exact preconditions.
Like the existing PTX audit, this is **structural validation**, not a
formally proved equivalence to executed SASS.

The full-parameter BPTT example now explicitly opts in and prints
`cuda_graph_before`, `cuda_graph_after`, removed kernel count,
fusion candidates, and percentage of real CUDA kernel launch reduction.
The baseline is compiled only, not allocated/run on GPU. For an optional
CUDA-event timing of the **fused graph only** in Fish:

```fish
set -gx TORUS16_BENCH_FUSED 1
task train-lab-torus16-ema4-bptt
set -e TORUS16_BENCH_FUSED
```

This excludes host cross-entropy, AdamW, host-device updates, compilation,
and memory allocations: it is **not** an end-to-end speedup metric.
The default no-fusion GpuLaunchConfig preserves previous behavior.
Check `task pointwise-mul-add-fusion` and
`task verify-gpu-mul-add-fusion` before the full NVIDIA training run.
Unfused and fused outputs are bitwise compared on two operand-order
variants by the ignored hardware test. Full EMA4 training parity
across several epochs and strict FP32 proof remain separate goals.

## Three-op GPU fusion and reproducible performance scopes

The experimental `with_pointwise_fusion()` additionally looks for
exclusive `Mul(a,b) → Add(product,c) → Mul(sum,k)`, where `k` is
one of the **same three external tensors** `a/b/c`. These chains use
one CUDA Graph kernel rather than three, preserving the original
operand ordering via three separate rounded instructions
`mul.rn.f32 → add.rn.f32 → mul.rn.f32`. A fourth external input,
observable intermediate, shared intermediate, nonmatching shape or
overlap with an existing ReLU fusion prevents this rewrite.
The same semantic IR remains canonical. The translation plan and
independent PTX/LLVM structural audit validate operation order,
input addresses, three-port launch ABI, and rounded instruction counts.
These are experimental structural checks, **not** a Lean proof of
IEEE/SASS bitwise equivalence.

Tasks:
`task pointwise-three-op-fusion` (CPU structure and negative tests)
and `task verify-gpu-three-op-fusion` (12 operand-layout NVIDIA cases).
`task train-lab-torus16-ema4-bptt` prints both two-op and three-op
fusion counts, total launched kernels and savings versus the disabled
fusion profile. Setup and each end-to-end training window are timed
separately; throughput includes CPU CE, AdamW, CUDA replay and host
transfers, but excludes compile/setup and validation. This matters:
CUDA kernels removed does not imply proportional training speedup.

Optional Fish commands to compare **GPU-only** unfused and fused event
durations (three warmups, ten measured graph replays each):

```fish
set -gx TORUS16_BENCH_COMPARE 1
task train-lab-torus16-ema4-bptt
set -e TORUS16_BENCH_COMPARE
```

The two graph allocations and benchmarks are performed sequentially,
not simultaneously in device memory. CUDA event timing excludes
host CE/AdamW, host/device copies, compilation and graph setup.
The comparison mode increases setup cost and does not compare whole
training wall-clock performance between different optimized profiles.
Hardware measurements must be gathered on the user's NVIDIA system.


## Batched host/device I/O and long-run throughput

The BPTT example now uses the architecture-neutral
`GpuExecution::update_input_subset(&[(NodeId, &Tensor<f32>)])`
and `GpuExecution::read_outputs_selected(&[NodeId])` APIs.
The former validates **all** updated IDs (no duplicates), shapes and
domains before scheduling pinned H2D transfers on the owned nonblocking
stream, synchronizing exactly once for the batch. Unmentioned inputs
retain their values. The latter rejects invalid/duplicate/unselected
or non-F32 outputs before transfer, queues D2H to session-owned pinned
buffers and synchronizes once. Existing `read_outputs()` is now
batched too; `read_values()` retains its mixed-dtype semantics.

For BPTT=2, the forward CE seed calculation retrieves only **two**
logits rather than the entire 341-output VJP graph. After backward,
all 271 gradients and 68 terminal states are downloaded in one batch.
Weights (271) and recurrent initial states (68) are updated together
after AdamW and host state snapshot. The GPU kernel graph and the
mathematical/AD computation are unchanged.

`task train-lab-torus16-ema4-bptt` uses a **release Rust build** for
host optimizer performance. Its first invocation recompiles; later
invocations reuse Cargo artifacts. `step_profile_window` records
per-stage elapsed wall time for windows 0..2 and every 128th window:
H2D inputs, forward replay, selected logits D2H, host CE,
H2D seeds, backward replay, selected gradient/state D2H, host AdamW,
and H2D updated parameters/states. Stage times do not include
print/log output or setup; they are not CUDA event timings.
For hardware acceptance run `task verify-gpu-batch-transfers` first.
No numerical parity, GPU throughput, or long-run robustness result
is claimed until the local NVIDIA acceptance passes.

## Durable EMA4 TBPTT checkpoint and resume

The BPTT trainer now writes an atomic, versioned checkpoint by default to
`target/torus16-ema4-bptt.ckpt` (ignored by Git). A checkpoint stores all
271 trainable parameter tensors, all AdamW moment tensors/hyperparameters,
68 recurrent initial states at the next window boundary, BPTT size,
vocabulary size, chronological window cursor, and a checksum/fingerprint
of the exact corpus file. It is written every 128 windows and at the
end of the requested training range. On startup, a matching checkpoint
is resumed automatically; corrupt, truncated, wrong-corpus, wrong-BPTT,
wrong-shape, wrong-moment or out-of-range checkpoints fail closed. A
temporary file is synced and atomically renamed, leaving the previous
complete checkpoint available after most interruptions. The fingerprint
is not cryptographic authenticity verification.

`TORUS16_TRAIN_WINDOWS` is a **total target count**, not an additional
number of windows. To continue a previously completed 128-window
checkpoint to 4096 windows in Fish:

```fish
set -gx TORUS16_BPTT 2
set -gx TORUS16_TRAIN_WINDOWS 4096
set -gx TORUS16_VAL_LIMIT 4096
task train-lab-torus16-ema4-bptt
```

Change `TORUS16_CHECKPOINT` to use a separate path; change
`TORUS16_CHECKPOINT_EVERY` to save more frequently (default 128).
To **explicitly start from deterministic initial weights** and overwrite
the previous checkpoint after the next save, set `TORUS16_FRESH=1`
for that run and unset it afterwards (`set -e TORUS16_FRESH`).
No checkpoint is silently imported from the previous pre-checkpoint
runs. This version 1 format applies to this EMA4-LM schema; semantic
architecture changes should bump the format tag. The optimizer snapshot
API is architecture-neutral, while persistence stays lab-owned.
Checkpoint writes add CPU/disk time at save windows and are included
in `end_to_end_step_ms`. Run `task test-lab-torus16-ema4-checkpoint`
for CPU unit tests and compare a resumed NVIDIA run to an uninterrupted
run before claiming numerical parity.

Training now computes per-step softmax CE cotangents directly in the
Semantic IR as `ArgMax → Take` for a stable maximum, `Exp`, `MatMul`
(sum), and `Div/Sub`. The library's architecture-neutral
`differentiate_with_seeds` treats these as terminal adjoints. One CUDA
Graph replay computes both the forward logits and all parameter VJPs;
logits are downloaded only with gradient/state outputs for CE diagnostics.
The separate holdout `ModelExecution` remains forward-only. This removes
the previous two training replays and host CE seed transfer, but does NOT
promise a speedup until tested on NVIDIA hardware. GPU approximate-exp,
optimizer behavior and the generated CUDA code are not Lean-certified.

Short two-token train BPC is not holdout quality; validation BPC is
measured after the requested total training range. No Lean or
machine-code proof of checkpointed AdamW/FP32 training is claimed.

### Device-resident TBPTT state feedback (experimental)

The training loop reuses VeriNeural's generic `GpuExecution::feedback_state`
contract for all 68 recurrent buffers on the owned non-blocking CUDA stream.
Every window transfers only logits and trainable gradients D2H and AdamW
parameters H2D. Recurrent state is read back as a single selected-output batch
only at checkpoint boundaries, before parameter input updates invalidate the
outputs. Checkpoints retain the same versioned state/optimizer schema. This is
an experimental runtime optimization, not an FP32 proof or a guaranteed GPU
utilization/speedup claim; compare checkpoint bytes for uninterrupted and
resumed runs and benchmark sustained throughput on the target hardware.

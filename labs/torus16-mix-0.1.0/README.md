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

### Additional universal binary pointwise fusion (experimental)

VeriNeural's opt-in pointwise fusion now also combines exclusive full-shaped
`Add→Add`, `Add→Mul` and `Mul→Mul` pairs. Every fused kernel retains the
original two separate `*.rn.f32` instructions and operand order; no FMA
contraction, reassociation or overwrite of shared/selected intermediates is
permitted. The binary-chain plan is checked against Semantic IR memory extents
and per-kernel LLVM/PTX structural evidence. These checks are not a hardware
FP32 proof. `cuda_graph_before/after` and `binary_pointwise_fusions` measure
kernel-node reduction; actual GPU utilization and sustained tokens/sec must
be measured separately on the target GPU.

### GPU P2 numerical equivalence acceptance (pending NVIDIA run)

`task verify-gpu-p2-equivalence` is an explicit, developer-run NVIDIA
acceptance gate. It verifies the generic D2D recurrent-state update against
the previous host D2H/H2D staging path (exact FP32 bits across several steps),
then runs all ignored fused/unfused Add/Mul GPU parity suites. Finally it
executes the real EMA4 BPTT backward graph with `TORUS16_VERIFY_FUSION=1`:
the unfused and fused graphs receive identical initial weights, states,
tokens and one-hot CE labels. Every selected logit, parameter gradient and
terminal state must match bit-for-bit, with no simultaneous double GPU-graph
allocation. The gate then compares training as 2 windows + checkpoint/resume
to four continuous windows, checking per-window CE/BPC and optimizer steps,
then byte-for-byte equality of the complete final checkpoint, including
parameters, 68 recurrent states, AdamW moments and cursor.

The gate uses isolated `mktemp` checkpoint paths and the checked-in corpus,
overriding training-related settings inside its own shell command. Its files
are deleted on success and kept on failure for investigation. Normal
`TORUS16_CHECKPOINT` files are not touched. This validates one finite
hardware configuration; it does *not* establish formal FP32, LLVM/PTX,
CUDA JIT/SASS or general training-convergence guarantees. The optional
`TORUS16_VERIFY_FUSION=1` environment setting can also be used with
`task train-lab-torus16-ema4-bptt` to compare all training outputs on
the first pending window before continuing ordinary training.

### Experimental generic GPU AdamW through Semantic IR (opt-in, GPU acceptance pending)

`TORUS16_GPU_ADAMW=1 task train-lab-torus16-ema4-bptt` uses
`GpuAdamWPlan`, an architecture-neutral graph extension in
`crates/verineural/src/gpu_adamw.rs`. It assembles AdamW update
expressions out of the existing public graph primitives (Mul/Add/Sub/Div,
IEEE sqrt and scalar broadcast), rather than installing lab-specific
CUDA kernels or launching compute outside CUDA Graph. Only the two FP32
bias-correction scalars and the next one-hot training inputs are uploaded
per window. First and second moments, trainable weights, and recurrent
states are retained as persistent GPU inputs and are fed back D2D.

Important safety/performance boundary: model weights are declared
`Domain::Finite`, whereas unchecked FP32 arithmetic does not establish
a finite result. The generic `GpuExecution::feedback_state_checked`
therefore reads the *updated weights only* D2H as one pinned batch and
checks the original finite domain **before** any D2D feedback copy.
No gradient or AdamW moment tensors are read each window. At each
checkpoint boundary, weights and both moment tensors are read, checked,
then serialized in the existing checkpoint format. GPU-resident here
means no host optimizer or parameter H2D each step, **not** literally
zero D2H traffic or a guarantee of better throughput. Additional
optimizer IR kernels may erase the saved transfer time; compare sustained
tokens/sec against the default CPU path on the same machine.

The default remains the prior host AdamW until `task verify-gpu-adamw`
passes on NVIDIA hardware. That gate compares GPU AdamW with a host
numeric oracle for several steps (tolerance explicitly reported in
the test), verifies GPU checkpoint/resume **bit-for-bit**, and exercises
the full EMA4 training loop over 2+2 versus 4 uninterrupted windows.
This is hardware testing, not a Lean/IEEE/PTX/SASS proof. An AdamW
result is checked for finite parameters before any state update, but
moment tensors are validated on checkpoint export rather than on every
step. An invalid GPU optimizer output can stop training without a
durable checkpoint for the current window.

A separate `task benchmark-gpu-adamw` runs 256 windows under both
backends with isolated checkpoints and prints actual end-to-end tokens/s
and CUDA kernel counts. The finite-domain D2H validation is included in
GPU-mode throughput; compilation and holdout scoring are not.

### Generic independent CUDA Graph sessions (opt-in, NVIDIA acceptance pending)

For unrelated programs, independent input streams or separate consumer
models, `GpuEngine::prepare_independent` prepares multiple executions
from one compiled `GpuProgram` and validates their source inventories
before CUDA allocation. `GpuExecution::replay_independent_batch` queues
their graphs on distinct non-blocking streams *before* synchronization;
individual `enqueue_replay`/`complete_replay` methods are also available.
A pending launch is not a completed result and rejects readback/updates.
`task verify-gpu-independent-batch` checks bitwise parity with serial
replay, and `task benchmark-gpu-independent-batch` measures host-wall
latency of four independent execution streams. Both are explicit GPU gates,
not part of normal host-only tests. This API does not yet implement
shared-weight batched EMA4 training, gradient aggregation, or higher GPU
utilization for a single recurrent sequence. Tensor-wise finite-domain
validation of GPU AdamW remains D2H; no fully GPU-side finite guard is
claimed.

### GPU AdamW denominator fusion (experimental)

Generic VeriNeural pointwise fusion includes strict-order
`Sqrt→Add→Div` with three full-shaped inputs, targeting the expensive
`sqrt(v_hat) + epsilon` denominator pattern present in many algorithms,
including the optional GPU AdamW. The compiler emits a single CUDA Graph
kernel with three separate rounded PTX operations instead of three
kernels, without reassociation or FMA. Shared/observable intermediate
values are not fused. Check with
`task verify-gpu-sqrt-add-div-fusion`. Actual reduction in EMA4 kernel
count, end-to-end tokens/s and GPU utilization must be measured on the
target device after testing. Existing finite-domain parameter checks
still require D2H until a separately verified on-device predicate exists.

The first full P3 hardware attempt stopped on `UnsupportedOperation:
broadcast_to` for non-`[1,N]` optimizer parameter shapes. The GPU scalar
broadcast lowering now accepts any supported target rank when the source
contains exactly one element. All general non-scalar broadcasting still
fails closed. The `gpu_adamw` tests cover structural rank-1/2/3 acceptance,
reject general broadcasting and compare NVIDIA scalar bit patterns.
Correctness and performance of the full GPU optimizer still require a
fresh run of `task verify-gpu-adamw` and `task benchmark-gpu-adamw`.

### GPU AdamW hardware result (256 windows, one developer run)

`task verify-gpu-adamw` passed the finite-domain, host-oracle and bitwise
restart checks, including 2+2 versus uninterrupted 4 EMA4 training windows.
However, the first direct throughput comparison showed **CPU AdamW 132.83
training tokens/s** with 6,892 post-fusion kernels, while GPU IR AdamW
managed **61.17 tokens/s** with 9,638 kernels on the tested GPU. GPU AdamW
was **2.17 times slower**. These results are for BPTT=2 and 256 windows,
not general performance guarantees. Keep `TORUS16_GPU_ADAMW` unset for
production-length EMA4 training; it is an experimental comparison mode.
A structural negative broadcast test was updated for the expanded scalar
CUDA profile. The revised sources still need a test rerun.

### Experimental four-operation pointwise fusion (hardware verification pending)

The generic compiler now supports exclusive four-stage
`Mul→Mul→Add→Div` chains with a repeated first multiplicand, four
external inputs and separate explicit FP32 RNE operations. This matches
one subgraph of the GPU AdamW second-moment update without embedding
optimizer logic in the kernel generator. The opt-in pointwise fusion
removes up to three kernel launches per eligible chain; actual matches
are reported as `four_op_fusions` in the `cuda_graph_before` log.
`task verify-gpu-four-op-fusion` covers all eight operand-order/reuse
cases, selected/shared intermediate rejection, memory/translation plan
and NVIDIA fused/unfused FP32 bit parity. The existing
`task verify-gpu-adamw` validates optimizer checkpoint integration.
Only a fresh `task benchmark-gpu-adamw` can establish throughput;
the previous GPU AdamW measured 61.17 tokens/s against CPU AdamW at
132.83 tokens/s, so the CPU remains the default. No performance or
hardware result is claimed for this new rule before that run.

### GPU AdamW diamond-fusion follow-up (pending NVIDIA verification)

The first four-op linear fusion passed its standalone NVIDIA GPU parity
test but produced `four_op_fusions=0` in EMA4, because the second
moment's `Add` output is public and must survive D2D feedback and
checkpoint export. The developer measured CPU AdamW **158.17 tokens/s**
and GPU AdamW **66.05 tokens/s** (256 BPTT=2 windows) in that run.
The library now has a second, bounded pointwise `MulDiamondAdd` rule
that preserves this observable output while fusing three internal Mul
kernels. The benchmark reports `diamond_fusions`, distinct from
`four_op_fusions`. New structural and GPU bitwise acceptance is
`task verify-gpu-diamond-fusion`; also rerun `task verify-gpu-adamw`
and `task benchmark-gpu-adamw`. CPU AdamW remains the default until
the GPU path actually wins in sustained end-to-end tokens/s.

## Тюнинг производительности EMA4-BPTT

Для `examples/ema4_bptt.rs` поддерживаются независимые настройки исполнения:

- `TORUS16_GPU_ADAMW=1`: экспериментальный optimizer в Semantic IR и CUDA Graph; **не гарантированно быстрее** host AdamW. На границах checkpoint параметры и моменты всё равно выгружаются. Режим входит в архитектурный fingerprint и не совместим с checkpoint CPU-режима.
- `TORUS16_METRIC_EVERY=N` (по умолчанию 1): с GPU AdamW читать logits и выводить train CE/BPC только на первом, каждом N-м и последнем окне. Градиенты, оптимизатор, рекуррентные состояния, validation и checkpoint работают на **каждом** окне без изменений. При N>1 кривая training BPC становится *выборочной*, а не средним по всем окнам; не сравнивать её со старой полной кривой напрямую. Для CPU AdamW N>1 запрещено.
- `TORUS16_THREADS_PER_BLOCK=64|128|256|512` (по умолчанию 256): допустимые CUDA launch-профили VeriNeural. Размер блока влияет на scheduling и должен подбираться по NVIDIA benchmark, а не по температуре.

`task benchmark-torus16-throughput` измеряет один и тот же BPTT=2 / 256-window GPU AdamW на размере блока 256 с полным и редким чтением logits, затем сканирует блоки 64/128/512 при N=32. Проверяется побитовое совпадение обоих checkpoint при block=256. Это **локальная NVIDIA-приёмка**, результаты зависят от GPU/драйвера и не получены агентом. Отдельный `task benchmark-gpu-adamw` сравнивает CPU и GPU AdamW. Не принимать ускорение на основании числа CUDA kernels или процента GPU-Util: важна `training_tokens_per_second` при совпадающей математике и проверенном resume.

### Экспериментальный профиль LLVM NVPTX

`TORUS16_LLC_OPT=0|2` управляет только уровнем кодогенерации `llc`; по умолчанию `0` (старое поведение). `2` включает экспериментальный `-O2` вместо `-O0` для всех моделей внутри запуска, включая validation. Операции `mul.rn.f32`/`add.rn.f32` и отсутствие fast-math сохраняются в исходном LLVM IR, но это **не доказательство** побитового совпадения PTX/JIT/SASS. Компилятор по-прежнему проводит fail-closed translation validation перед запуском.

`task benchmark-torus16-codegen` прогоняет одинаковое обучение GPU AdamW (BPTT=2, 256 окон, readback /32) с `-O0` и `-O2`, выводит end-to-end tokens/s и CUDA Graph replay, сравнивает выборочные train-метрики и два checkpoint побитово. Если translation validation или хотя бы одна проверка не прошла, `-O2` **не принимать**; логи остаются в созданном временном каталоге. Даже после успешного прогона это локальная численная приёмка, а не Lean/IEEE-754 сертификация. Скорость нужно повторить несколькими запусками: один замер 256 окон слишком шумный для небольших выигрышей.

### Опциональное переиспользование inference CUDA Graph при holdout

`TORUS16_REUSE_HOLDOUT=1` (по умолчанию выключено) компилирует holdout модель один раз до начала учёта времени обучения, затем повторно использует inference CUDA Graph для validation и финального test. Перед каждым скорингом все активные параметры обновляются по имени (при отсутствии берётся исходное значение), а рекуррентные состояния явно сбрасываются к исходным. CUDA Graph replay и вычисление CE не меняются. `holdout_session_prepare_s` показывает однократную стоимость подготовки отдельно от `validation_total_s`, чтобы сравнение полной стоимости запуска было честным. Это удерживает дополнительную GPU-сессию до конца обучения. `task benchmark-torus16-holdout-reuse` сравнивает training output, validation/test BPC и final/best checkpoint побитово между исходным fresh и кешированным путём. До успешной локальной GPU-приёмки никаких гарантий ускорения или FP32/Lean-эквивалентности нет.

### Finetune seed-0 от лучшего checkpoint с понижением AdamW LR

`task finetune-torus16-best-lr5` продолжает **ровно две** конфигурации: `torus_shared:matched:seed0` и `independent:matched:seed0`. Источник `TORUS16_FINETUNE_SOURCE_DIR` (по умолчанию `target/torus16-pilot`) должен содержать исходные `*.best.ckpt`; они копируются только при первом запуске в отдельный `TORUS16_FINETUNE_DIR` (по умолчанию `target/torus16-finetune-lr5-seed0`). Исходные checkpoint никогда не переписываются. `TORUS16_FINETUNE_WINDOWS` (по умолчанию `98304`) задаёт абсолютное число окон, а не число новых шагов; повторное выполнение с `131072` продолжит полученные checkpoint. На новой ветке эксперимента `TORUS16_LR_DECAY_5X=1` меняет только learning rate `0.002 → 0.0004` в восстановленном AdamW (CPU и GPU используют один checkpoint); накопленные моменты, `steps`, веса и рекуррентные состояния сохраняются. Уже переведённый на `0.0004` checkpoint продолжает без повторного снижения. Опциональный путь запрещает fresh-обучение, другие коэффициенты и неявный переход назад. Baseline `compare-torus16-matrix-tying` сбрасывает этот флаг. Лучший checkpoint исходной траектории остаётся альтернативным кандидатом на validation: до нового улучшения он сохраняется в отдельном finetune-каталоге, а test оценивает именно лучший найденный checkpoint. Новый learning rate не означает доказательства сходимости; GPU/FP32 по-прежнему экспериментальны.


**Продолжение BPTT=4:** та же `task finetune-torus16-best-lr5` поддерживает явный профиль `TORUS16_FINETUNE_BPTT=4`: выбирает исходные лучшие checkpoint из `target/torus16-bptt4-seed0`, целевой каталог `target/torus16-bptt4-finetune-lr5-seed0`, `BPTT=4`, validation раз в 1024 окна и checkpoint раз в 512 окон, ровно как при исходном обучении. По умолчанию продолжает до **65 536 окон BPTT=4 = 262 144 символов**, начиная с best checkpoint предыдущих 32 768 окон. Это сравнимый общий объём символов с BPTT=2 на 131 072 окнах, но не равное число обновлений AdamW. Для старого BPTT=2-профиля ничего не меняется. Команда для BPTT=4: `TORUS16_FINETUNE_BPTT=4 task finetune-torus16-best-lr5`. Увеличение `TORUS16_FINETUNE_WINDOWS` продолжает checkpoint из отдельного finetune-каталога без его перезаписи; переменная задаёт *абсолютное* число окон.

Важное ограничение: прежние `VAL_LIMIT=512` и `TEST_LIMIT=512` зафиксированы fingerprint checkpoint. Продолжение с другим интервалом без отдельного evaluation-only инструмента запрещено. Test уже многократно просматривался и не является независимым подтверждением.

## Torus16-Mix 0.2.0: Role-Delta (экспериментальный срез)

Вариант Role-Delta использует тот же разделяемый W[48,48], что `torus_shared`,
и две независимые обучаемые матрицы A[48,q], B[q,48] на роль в каждом из
четырёх слоёв. Выход вычисляется в общем VeriNeural Semantic IR как
`mixed @ T_r(W) + scale * ((mixed @ A_r) @ B_r)`, затем применяются те же
roll, row scale и bias. Полная виртуальная матрица не материализуется.
Поддерживаются `TORUS16_MATRIX_MODE=role_delta_2|role_delta_4|role_delta_8`
(также `role_delta` означает rank=4); `TORUS16_DELTA_SCALE` (0,16],
по умолчанию 1.0, задаёт **фиксированную** величину λ/q, а не дополнительный
обучаемый gate. `TORUS16_ROLE_INIT=matched` обязателен.

Инициализация: общий W и все старые параметры берутся из прежнего baseline,
A детерминированно и раздельно по ролям инициализируется малыми ненулевыми
значениями, B = 0. В начале коррекция математически равна нулю; на первом
шаге градиент A равен нулю, а B может получать градиент. Суммарные
числа обучаемых параметров при словаре из 104 символов: 31 052 (q=2),
37 196 (q=4), 49 484 (q=8). Режим не изменяет старые shared/independent
IR или checkpoint; для каждого rank отдельный путь `target/*role_delta*`.
Fingerprint охватывает ранговые порты, начальные A/B, масштаб, seed и
параметры обучения. Resume несовместимого checkpoint запрещён.

Верификация в этом коммите: структурная Rust-проверка форм, имён,
параметров и отделения режимов плюс отключённый по умолчанию GPU-тест
исходных logits/EMA state; запустить `task verify-torus16-role-delta` на
NVIDIA машине. GPU, strict FP32, градиенты, снижение BPC и Lean-сертификация
нового составного оператора **не подтверждены локальным прогоном**.
Weight-only warm-start и отдельный evaluation-only runner ещё не сделаны,
архитектурный BPTT остаётся 2–8. Не смешивать этот срез с dynamic EMA,
ReZero или увеличением глубины до прохождения измерительных ablation.

### Role-Delta обратный проход: независимая проверка

`task verify-torus16-role-delta` дополнен парой проверок:
`role_delta_scalar_vjp_matches_finite_differences` сравнивает
аналитические градиенты W/A/B независимого scalar oracle с центральными
конечными разностями без GPU. `role_delta_gpu_vjp_matches_scalar_oracle`
проверяет реальные forward и reverse-mode VJP тех же портов через
VeriNeural Semantic IR и CUDA Graph, отдельно для чётной/нечётной роли,
нулевой и ненулевой B. При B=0 градиент A обязан равняться нулю,
а B должен получать ненулевой обучающий сигнал; при B≠0 обе
матрицы имеют градиент. Проверяется и транспонирование общей W.

Пользователь локально подтвердил предыдущие структурную и начальную
GPU-parity проверки 26.10.2026 в Nix devShell. Новые тесты с
градиентами добавлены, но **ещё не запускались**. Численная GPU-проверка
не равнозначна формальному Lean-доказательству и не удостоверяет SASS.

# VeriNeural: журнал реализации и проверки

Исторические инженерные заметки, записи приёмки и подробности уже предпринятых изменений. Это **не** нормативная спецификация и **не** подтверждение успешности тестов на текущем HEAD. Источник требований: [SPEC.md](SPEC.md), состояние этапов и ограничения: [ROADMAP.md](ROADMAP.md), реальная поддержка: исходники и результаты проверок.

Материал сохранён из прежних разделов документов при KISS/DRY-консолидации, чтобы не терять описания регрессий, условий и результатов. Он может относиться к более ранним коммитам.

## Хронологические заметки из ROADMAP

**Stage 10, стабилизация публикации (код, локальная приёмка ожидается):** повторная замена существующего auto-build/ на Linux переведена на атомарный renameat2(RENAME_EXCHANGE), который fail-closed при отсутствии поддержки ядра/ФС. Первое создание auto-build/ требует одного rename. Старый bundle удаляется уже после переключения, ошибка его очистки не превращается в фиктивную неудачу доказательства. Симлинк вместо каталога auto-build запрещён. Лаборатории используют один библиотечный certified_example_main вместо копий CLI-кода. Добавлены unit-тесты swap, первого выпуска, несогласованных backup и внешних ссылок. Всё ещё отсутствует подтверждение выполнения Lean/LLVM/Graphviz, а первоначальная публикация семи root-alias и legacy migration не являются одной атомарной операцией. Tracked Stage9 артефакты в корнях лабораторий будут изменены при default migration; для проверки использовать --artifact-root /tmp/<unique-name>.

**Stage 10, регрессии после первой локальной приёмки:** `task check` прошёл, три `verify-certified-stage10` проверки Lean/SHA-256 прошли; полный `task test` остановился на некорректном домене входа теста `sqrt`, `task clippy` остановился на `too_many_arguments`, Cargo предупредил о совпадении имён двух `certified` examples. В следующем срезе test input изменён на `NonNegativeFinite` (контракт `sqrt` остаётся строгим), replay переиспользует один `AutoBuildPlan`, а имена executable examples различаются. Локальная повторная приёмка этих исправлений ожидается.

**Stage 10, первая успешная локальная end-to-end приёмка (2026-10-23, `84980ea`):** разработчик подтвердил `task check`, `task test --workspace` (все запущенные тесты прошли, отдельные tool-dependent тесты по умолчанию ignored), `task clippy` с `-D warnings` и `task verify-certified-stage10` (3/3 включая ложную цель, чужую identity и tamper). `task certified-build` и `task verify-certified-build` через Cargo CLI успешно собрали и перепроверили *обе* независимые модели с отдельными absolute artifact roots: XOR (`6bc265ca583d2e4c6803fecf74c5b7df0f39f3817d7902d425ccc6deeda77797`) и CoupledField (`0788971dd228bc51de0105780817d45d9141c832f1fc18d279cf386d83c47f44`). Lean checker: 4.29.1, по одной обязательной named goal на архитектуру. Это подтверждение Lean ideal-Int + структурного GPU lowering и offline replay, **не** GPU hardware/JIT/SASS equivalence. В этих логах нет второго build в тот же root, проверки неверного внешнего SHA-256 pin или запуска CUDA Graph. Добавлен четвёртый ignored acceptance-тест, воспроизводящий повторную публикацию, детерминированность SHA-256, верный/неверный pin и post-reissue replay; его локальный результат ещё ожидается. Первый root alias migration остаётся неатомарным; универсальному consumer пока нужен явно оформленный `examples/<name>_certified.rs`.

**Приёмка P0/P1/P2 и rotating GPU, новый код (локальный прогон ожидается):**
`task verify-api-acceptance` объединяет существующие P0 Lean/consumer gates без изменения
математических контрактов; `task verify-lab-rotating-gpu` запускает проверку
типизированного GPU `ArgMax/Index` и новый `labs/rotating-matrices-0.1.0/tests/gpu_parity.rs`.
Независимый CPU-оракул адресует все восемь пространственных маршрутов формулами
координат и проверяет многотокенный deterministic GPU forward, readout, state reset
и политику `ArgMax First` против одного переиспользуемого CUDA Graph. CPU smoke-test
требует различимых результатов всех маршрутов. Всё GPU/Lean acceptance — пока
**не проверено локально**; `task gpu-state-replay` и `task lab-rotating` остаются
обязательными отдельными аппаратными проверками. Это численная validation,
не доказательство FP32/LLVM/PTX/JIT/SASS.
\n**Stage 10 и P1 proof-boundary hardening (код; локальная приёмка ожидается):**\nкорневая миграция семи root alias при ошибке теперь откатывает установленные ссылки\nи возвращает исходные tracked legacy-файлы из backup; fault-injection tests\nпроверяют rollback fresh и legacy-публикации. Атомарность видимости\n*семи различных* root paths всё ещё не гарантируется, зато неуспешная миграция\nне оставляет частично перенесённые артефакты в штатном однопроцессном профиле.\nLean `PointwiseLe` generator переиспользует общие композиционные леммы\n(`refl/trans/add with nonnegative/Relu`) поверх DAG и самостоятельно доказывает\nнеотрицательность составной sibling-ветки из ReLU/Add/Mul, без присвоения\nruntime Domain статуса Lean-гипотезы. Преднамеренно ложные отношения\nпродолжают отклоняться kernel-check в certified-пути. Это поддерживаемый\nrelational фрагмент, не произвольный theorem-dependency framework;\nобязательная проверка structural LLVM/PTX/memory evidence и явные\n`fp32_machine_equivalence=not-proven`, `nvidia_jit_sass_equivalence=not-proven`\nостаются trust boundary. Команда `task verify-stage10-proof-boundary` запускает\nвсе связанные проверки при совместной локальной приёмке.\n\n**Совместная приёмка после Stage 10/P1 усиления:** `task verify-all-acceptance`\nобъединяет check/test/clippy, P0/P1/P2, root migration fault-injection,\nLean relational positive/negative claims, GPU memory/translation audit,\nStage10 certified replay, state feedback и rotating NVIDIA parity/training.\nОтдельный тест с намеренной заменой `add.rn.f32` на `add.ftz.f32`,\n`mul.rn.f32` на `fma.rn.f32` / `mul.approx.f32` и нарушенным\nколичеством операций обязан быть отклонён существующим FP32 PTX auditor.\nReport дополнительно различает проверенные условные Lean memory obligations,\nструктурный LLVM/PTX audit и отсутствие отдельного RNE bounds/cuda-runtime proof.\nВсе эти новые проверки **ожидают запуска пользователем**; полноценная\nформальная JIT/SASS или arbitrary-predicate theorem composition не заявлена.\n

**Torus16-Mix structural row profile (код, локальная приёмка ожидается):**
GPU lowering переиспользуемых операций дополнен узким профилем `SliceAxis(axis=1)`
для `[1,N]`, `BroadcastTo` из одного элемента в `[1,N]` и
`Concat(axis=1)` для двух однострочных тензоров. Последняя операция
переиспользует flat-concat адресацию только при доказуемой эквивалентности
форм. Планы памяти фиксируют исходные размеры и сдвиг, translation
validator проверяет LLVM-шаблоны и PTX FP32 policy; генератор Lean
выводит арифметическое обязательство для границ среза, не утверждая
полного доказательства CUDA. `crates/verineural/tests/structural_last_row.rs`
проверяет plan round-trip, поддерживаемые и отвергаемые формы и содержит
отдельный ignored NVIDIA CUDA Graph replay. Команды:
`task test-lab-torus16-mix`, `task verify-gpu-structural-last-row`.
Локальные Cargo/Lean/LLVM/CUDA проверки агентом **не выполнялись**.
`Exp`/`Div` и весь Torus16-Mix forward по-прежнему fail-closed,
поскольку строгая FP32 экспонента не реализована; для завершения лаборатории
нужны отдельная численная спецификация и проверенный lowering, не
скрытое approx-замещение. Никакие крупные нейросетевые блоки в библиотеку
не добавлены.

### Torus16-Mix: полный executable operator forward, экспериментальная numeric-область (2026-10-25)

Потребитель labs/torus16-mix-0.1.0 описывает единый 8-output операторный
граф с одной shared W и входным gated mixing до каждого MatMul.
Восемь выходов можно запускать в одном CUDA Graph через именованный
ModelExecution. Никаких предметных Torus/EMA/LLM-компонентов в ядре
VeriNeural нет.

Новые универсальные GPU-paths: pointwise subtraction (sub.rn.f32),
division (div.rn.f32), natural exponential via explicit experimental
opt-in GpuLaunchConfig::with_approximate_exp() (mul.rn.f32 x log2(e),
ex2.approx.f32). Доступны только версии AST Sub/Div/Exp с проверкой
канонического выражения и имени. Чужой PointwiseSpec, включая
совпадающее имя с изменённой математикой, отвергается.
LLVM/PTX translation validation проверяет регистры, инструкции,
rounding-режимы и численную метку ExpApprox, но не доказывает IEEE-754
error bound или PTX/SASS. Strict-сборка и certified-build по-прежнему
отклоняют Exp. Здесь нет молчаливого approximate fallback.

Приёмка: task test-lab-torus16-mix; task lab-torus16-mix-gpu;
task verify-gpu-torus16-mix. Аппаратный тест с ненулевым mix_gate
и нессиметричной W проверяет восемь выходов против независимого
Tensor reference. Авторские Cargo/Lean/CUDA проверки не запускались,
пользователь должен подтвердить результаты в локальном devShell.

Вне реализации: полная 4-layer EMA4-LM с Conv9/LayerNorm/SwiGLU,
recurrent state, backprop/AdamW, BPC и GPU memory benchmarks. Нельзя
выдавать готовность operator forward за воспроизведение обученного
результата 24 908-параметрической сети.

### EMA4-LM: 4-layer streaming consumer architecture (программный срез, приёмка ожидается)

В `labs/torus16-mix-0.1.0/src/ema4.rs` собрана полная конфигурация
**одного авторегрессионного inference-шaга** исходной 4-layer EMA4-LM
на универсальных VeriNeural primitives, без предметных компонентов в core:
tied character embedding/head, два LayerNorm и causal depthwise Conv9 на блок,
EMA с learnable decay, gated Torus16-Mix ролей 0..3, четыре SwiGLU
проекции ролей 4..7 и down через повторно используемые роли 0..1.
Потоковое состояние в `ModelBuilder`: 4 EMA vectors и 64 Conv9 history
vectors. Каждый `next_state` привязан к реальной зависимости общего IR,
исполняется persistent CUDA Graph через `ModelExecution.step()`;
никакой host recurrence в GPU path. Для словаря 104 параметров 24 908,
как в структуре исходного handoff (но значения не загружены из checkpoint).

Требуемая математическая операция `sqrt` расширена **универсальным**
versioned `PointwiseExpr::Sqrt`/`NeuralGraph::sqrt_ieee`
с LLVM/PTX `sqrt.rn.f32` и template audit. Отличие от
доменного `Operation::Sqrt` намеренно: pointwise-путь не придумывает
`NonNegativeFinite` для вычисленной дисперсии, разрешает NaN по
IEEE-семантике, а Lean proof для всего графа остаётся unsupported.
Поддерживаемая численная область всего forward использует explicit
experimental `ExpApprox`, в strict/certified-режиме fail-closed.

Независимый scalar Rust reference в `src/ema4_reference.rs`
проверяет последовательный причинный EMA/Conv9 step и logits.
Численная NVIDIA-приёмка сравнивает три шага с сохранением GPU
состояния (включая `mem_0`), task-команды
`lab-torus16-ema4`, `test-lab-torus16-ema4`,
`lab-torus16-ema4-gpu` и `verify-gpu-torus16-ema4`.
Запуск Cargo/Lean/CUDA задач агентом **не производился**.
Не реализованы autograd, AdamW, обучение, streaming quality/BPC,
checkpoint importer и сопоставление GPU FP32 с исходными обученными
весами. Это полный *streaming forward* на Rust/IR, не законченная
система обучения.

### Универсальный reverse-mode IR и первый trainable CUDA Graph vertical slice

Код `crates/verineural/src/autodiff.rs` строит reverse-mode VJP
в **той же** идентичности Semantic Neural IR: исходный forward
является неизменным префиксом, а производные являются обычными
узлами Add/Mul/MatMul/Permute/Reshape/Slice/Concat/Broadcast и
фиксированных versioned Pointwise. Учёт повторного использования
узлов через суммирование вкладов, seed явный. Проверка
достижимости и всех поддерживаемых reverse-правил выполняется
до построения производной; неизвестные операции fail-closed.
Контракт производной математический, **не** доказанная Lean
связь с точной FP32 производной приближённой `ex2.approx`.
Для выбранных параметров отсечение не относящихся к ним
ветвей уменьшает IR, не меняя семантику требуемого VJP.

`crates/verineural/src/optimizer.rs` добавляет архитектурно-
нейтральный **host AdamW** с атомарной валидацией named parameter
set, non-finite и shape guard. GPU backward исполняется без
нового backend через обычный CUDA Graph. Приёмка:
`task autodiff-stage`, `task verify-gpu-autodiff`,
`task train-lab-torus16-ema4` и структурный тест EMA4.
Аппаратная проверка ещё не была запущена при внесении кода.

Лабораторная команда демонстрирует *teacher-forced* CE,
GPU dL/d(final_beta), host AdamW и обновление параметра
в persistent GPU execution (только 48 trainable значений,
один токен, фиксированное состояние). Выводит toy-BPC,
который нельзя сравнивать с исследовательской метрикой.
Недостающие крупные части: unroll и 2+ шаговый truncated BPTT,
all-parameter train/optimizer на GPU, loss reduction внутри
Semantic IR, полноценная загрузка корпуса, стабильность градиента
на реальных данных, Lean численная сертификация reverse-pass.


### Finite-window BPTT and real-corpus EMA4 training: implementation awaiting local acceptance

Architecture-neutral verineural::unroll_model composes a named recurrent
Model across 1..=16 timesteps by cloning ordinary typed Semantic IR
operations, sharing single parameter sources, and wiring next_state
producers to later State consumers. It introduces explicit per-step
Input and cotangent sources and a scalar dot-product objective, so
the existing generic differentiate() produces reverse-mode VJPs
through time without lab backward kernels. Initial recurrent states
are explicit window-boundary sources, cutting the gradient graph there.
The new temporal.rs performs strict binding validation and rejects
unregistered sources, wrong shapes and missing transitions.
**New opt-in temporal reuse (code; CUDA acceptance pending):**
`unroll_model_hoisted` reuses a pure original-model operation across
window timesteps only when all of its transitive sources are named
`Parameter` inputs. `State` and per-timestep `Input` dependencies are
never hoisted. It shares the same public Neural IR and generic VJP/GPU
lowering; no architecture-specific kernels or new arithmetic are added.
The established `unroll_model` remains unchanged for old checkpoint
fingerprints. This transformation can alter the reverse AD accumulation
tree, so forward bitwise parity and gradient numeric parity require
separate NVIDIA acceptance via `task verify-gpu-temporal-bptt`.
A specific consumer training experiment is
`task benchmark-torus16-role-delta-hoisting`; speedup is not
confirmed until locally measured. Lean/strict-FP32/SASS semantics
are not upgraded by this graph sharing.
A tiny two-step CUDA test tests the *through-time* gradient analytically:
2.25 with the cross-time edge, versus 1.875 when detached.

The EMA4 consumer lists **all** nonconstant trainable parameters
(24,908 at vocab=104, including a single tied embedding/head and one
physical W per block), then uses a full-parameter window VJP and
generic host AdamW to update them. Between windows a snapshot of 68
terminal states is copied out before any state inputs are updated,
avoiding unsafe D2D overlap while retaining recurrent data and
explicitly detaching the past for TBPTT. Training uses a stable
host cross-entropy seed, compiled GPU forward+backward via CUDA Graph
and a held-out reset-state streaming validation pass.

The lab's CharCorpus::from_text uses a sorted Unicode character alphabet
and a 90/10 chronological split. The original handoff corpus_snapshot.txt is versioned in
labs/torus16-mix-0.1.0/data/ and selected by default through
CARGO_MANIFEST_DIR; TORUS16_CORPUS is optional for deliberate overrides.
SHA-256: c1d0509878099a43c85866c2f582be04fe8253292b137a83c58ebd04fb16ca85.
Expected handoff data: 524859 characters, 104 symbols. Commands:
task temporal-bptt-stage, task verify-gpu-temporal-bptt,
task train-lab-torus16-ema4-bptt (no external corpus path required).
Training default is BPTT=2 and two windows; the validation position
count is explicitly reported and by default capped at 64. No GPU run
of this new stage was performed by the agent, and full all-parameter
GPU compilation cost/VRAM are unknown.

Unfinished: proving floating-point reverse-mode correctness,
measuring end-to-end optimizer throughput, implementing in-GPU
AdamW/CE, ensuring parity of the research checkpoint/architecture,
exact handoff 51,648 validation positions protocol, checkpoint I/O,
long multi-seed learning and evaluating whether holdout BPC improves.


### Experimental Mul→Add NVIDIA GPU fusion for universal reverse-mode graphs (code, acceptance pending)

`GpuLaunchConfig::with_pointwise_fusion()` adds an opt-in typed,
exclusive-use `Mul → Add` compiler rewrite. Kernel combination is
valid only when Mul has exactly one reachable use, is not itself a
selected output, all three inputs/outputs have equal shape, and no
existing ReLU-chain fusion consumes either node. The unmodified
Semantic IR remains authoritative. A newly generated CUDA Graph launches
one kernel (separate `mul.rn.f32` then `add.rn.f32` without FMA) rather
than two. The exact original position of Mul in Add is encoded in the
translation plan. GPU memory contracts include all three actual source
addresses, explicit `GpuOptimizationCapability::MulAddFusion` records
the removed producer, and structural LLVM/PTX audit checks the template,
rounding and instruction counts. Unsupported cases remain unfused.

Acceptance: `task pointwise-mul-add-fusion`,
`task verify-gpu-mul-add-fusion`; selected-output/fan-out negative
tests plus bitwise NVIDIA parity for both Add operand positions.
The EMA4 full-parameter BPTT lab opts in, compares the fused and
non-fused **kernel counts before execution** and prints the reduction
(and optionally CUDA-event replay time). No measured hardware gain,
full training gradient parity or Lean/IEEE/JIT equivalence is claimed
until separately checked. More general backward fusion and
end-to-end train throughput/memory benchmarking remain open.

### Experimental three-operation fusion with bounded three-source GPU ABI

The existing opt-in `GpuLaunchConfig::with_pointwise_fusion()` additionally
selects a `Mul→Add→Mul` chain when both inner nodes have a sole reachable
consumer, no selected output or earlier fusion references them, every
operation/source has the same element shape, and the final Mul reuses
one of the three external sources. A fourth distinct source is explicitly
not supported in this bounded profile. Internal operation order and
both binary operand positions are encoded in versioned
`TranslationKind::MulAddMul`; LLVM template emits exactly
`mul.rn.f32`, `add.rn.f32`, `mul.rn.f32`, without FMA.
Memory contracts/capabilities explain two removed kernels and all
three source addresses. Structural LLVM/PTX audit checks exact operand
wiring, ordered instructions and byte-indexed pointer patterns.
No Lean proof, strict IEEE JIT/SASS guarantee or end-to-end speedup
claim arises from this pass.

Added CPU and ignored NVIDIA bitwise acceptance tests for all twelve
operand layouts, plus explicit fan-out, selected-output and fourth
distinct input negatives. EMA4 BPTT training logs real kernel counts,
CUDA-only replay comparison (optional TORUS16_BENCH_COMPARE), total
training wall time, tokens/s and per-window time. Device GPU benchmark
does not measure AdamW or transfers; full train step timing includes
those but excludes compilation. This new code has not undergone local
Cargo/NVIDIA acceptance and exact kernel counts remain unmeasured.


### Batched GPU I/O for generic Semantic IR programs (implementation, acceptance pending)

`GpuExecution::update_input_subset` sends prevalidated unique source
updates through pinned CUDA H2D staging and a single synchronization.
`GpuExecution::read_outputs_selected` reads only explicitly selected
F32 results, validates before enqueueing, retains output order and
state snapshot behavior, and synchronizes once after a session-owned
pinned D2H batch. Existing all-F32 `read_outputs` delegates to this
batch reader. No legacy stream or standalone kernel path introduced.
The EMA4 BPTT consumer now reads only supervised logits at the forward
phase and gradients plus terminal states after backward; it batches
input/seeds and 271 parameter + 68 recurrent-state updates. The
training task now uses Cargo --release, and labels CPU/GPU wall-clock
stage durations. All previous Semantic IR, autodiff and CUDA Graph
kernels remain unchanged.

Acceptance: `task verify-gpu-batch-transfers` (bitwise output parity,
duplicate/source/shape rejection without partial mutation, selected
reverse order, single input API compatibility). Then compare
end-to-end windows and `holdout_reset_streaming_bpc` with the
previous unfused-upload baseline on the same device. GPU time and
speedup are **not** established by source changes. Durable training
checkpoint + AdamW moment resume and faster GPU-side optimizer
are separate future milestones, not claimed here.


### Durable lab-owned EMA4 TBPTT restart snapshots (experimental, hardware acceptance pending)

Versioned binary snapshots persist exact named trainable parameters,
AdamW first/second moments and hyperparameters through the reusable
`AdamW::checkpoint()/from_checkpoint()` API, plus EMA4 initial-state
values at the next training window boundary, cumulative window cursor,
BPTT/vocabulary, exact corpus fingerprint, and a truncated-file
checksum. The checkpoint format and filesystem I/O remain private to
the EMA4 research lab, not VeriNeural core. Strict shape/name
validation and optimizer moment/step checks fail closed before resume;
atomic same-directory temporary file replacement preserves the previous
complete checkpoint on interruption. No extra Cargo dependency.
Automatic path: `target/torus16-ema4-bptt.ckpt`. Environment knobs:
`TORUS16_CHECKPOINT`, `TORUS16_CHECKPOINT_EVERY`,
`TORUS16_FRESH=1`. `TORUS16_TRAIN_WINDOWS` means total target windows.

CPU acceptance task: `task test-lab-torus16-ema4-checkpoint`
(round-trip, corrupted/truncated/wrong corpus/schema rejection, resumed
AdamW update matching uninterrupted update). GPU-level continuous vs
interrupted/resumed training equality remains to be measured. The
non-cryptographic fingerprint and file checksum do not guarantee
authenticity or proof of neural model correctness.

**P1 GPU feedback runtime safety (code, GPU acceptance pending):**
`GpuExecution::feedback_state` now rejects feedback before a completed
replay/benchmark or after source updates; errors from metadata preflight do not
mutate state. Observable selected outputs aliasing state destinations receive
CUDA D2D snapshots before feedback. Raw execution preserves all selected outputs;
named `ModelExecution` preserves only public outputs, avoiding extra state
snapshots for internal allocation roots. New replay/benchmark invalidates the
previous snapshots; every public read API uses the protected address. The
`ModelExecution::step` path avoids redundant manual snapshots. NVIDIA checks:
`task verify-gpu-feedback-runtime`, plus `task check`, `task test`, and
`task clippy`. No Lean proof of CUDA/JIT/SASS or hardware success is claimed.

**P1 device-domain guard и feedback CUDA Graph (экспериментальный код, NVIDIA-приёмка ожидается):** для динамических обязательств `Finite/NonNegativeFinite` отдельный runtime CUDA Graph проверяет битовый образ каждого F32 элемента на exponent=255 (NaN/Infinity), а для NonNegativeFinite дополнительно на отрицательный ненулевой битовый образ. `-0.0` допустим. Через D2H передаётся **один 32-bit status**, а полные значения читаются только при ошибке для получения точного host-oracle domain error. До проверки никакая feedback D2D-запись не выполняется. Стабильный набор D2D feedback копий кешируется во втором CUDA Graph и запускается на том же явном non-blocking stream; при смене списка связей проводится синхронизация и пересоздание. Обе вспомогательные графовые операции остаются доверенным NVIDIA runtime/PTX срезом, **не являются** автоматически Lean-certified семантикой. Без `task verify-gpu-adamw`, `task verify-gpu-feedback-runtime` и новых benchmark нельзя заявлять ни аппаратную корректность, ни ускорение. GPU AdamW остаётся экспериментальным, host остаётся default; общий multi-tensor/region fusion и proof of hardware FP32 correctness не реализованы.

### P2 GPU training numerical equivalence acceptance (code, NVIDIA run pending)

`task verify-gpu-p2-equivalence` runs the ignored multi-step exact
D2D-vs-host state-feedback regression and existing pointwise-fusion GPU
parity tests. The `TORUS16_VERIFY_FUSION` opt-in path runs the real EMA4
training graph both unfused/fused with identical next-window token and CE
labels, checks every logit, parameter gradient and terminal state as FP32
bits, and drops the baseline CUDA session before preparing the fused one.
Isolated 2+resume-to-4 and fresh 4-window executions compare CE/BPC and
AdamW step logs and byte-for-byte checkpoint contents. The check must not
reuse the user's primary checkpoint or infer any hardware result before the
actual NVIDIA run. This verifies a bounded numerical execution fixture,
not a formal FP32/PTX/JIT/SASS theorem; longer-horizon and
architecture-agnostic equivalence remain open.

### P3 experimental GPU AdamW (code only, NVIDIA acceptance pending)

`GpuAdamWPlan` composes the existing graph's gradient nodes with universal
tensor arithmetic for AdamW, creates explicit first/second moment sources
and bias-correction scalar inputs, and exposes D2D feedback plans. EMA4
opt-in `TORUS16_GPU_ADAMW=1` appends the optimizer to the same reverse
Semantic IR/CUDA Graph, eliminating per-window host AdamW, gradient
readback and trainable parameter uploads. One batched D2H finite-domain
validation of updated weights remains compulsory, because model parameter
sources require `Domain::Finite` and current FP32 graph operations cannot
prove it statically. Checkpoints preserve the legacy durable schema,
including moment tensors and cursor. No measured speedup or GPU utilization
claim; there are extra optimizer IR kernels. `task verify-gpu-adamw`
covers structural compilation, host numerical comparison, GPU bitwise
restart and EMA4 2+2 vs 4. Numeric discrepancy relative to CPU is
reported separately from strict GPU restart identity; CUDA/JIT/SASS and
arbitrary FP32 equivalence remain unproved.

### Independent CUDA Graph multi-stream scheduling (code; hardware acceptance pending)

The architecture-neutral `GpuEngine::prepare_independent` validates every
member of a batch of *independent* source inventories before allocating CUDA
resources, then prepares owned `GpuExecution` instances. Each owns its
CUDA Graph, allocations and explicit `CU_STREAM_NON_BLOCKING` stream in
the retained shared primary context. `GpuExecution::enqueue_replay` queues
work without a stream barrier; `complete_replay` synchronizes and publishes
the correct output epoch. `GpuExecution::replay_independent_batch` queues
all members before waiting, draining earlier launched sessions on enqueue
failure and waiting for every stream after a synchronization failure. The
single-session synchronous `replay` API remains unchanged.

While a graph is queued, all input updates, output readbacks, feedback,
benchmarks and repeat launches fail closed with `GpuError::ReplayPending`.
A failed CUDA enqueue/synchronization still permanently poisons the affected
session, and CUDA stream teardown synchronizes before freeing allocations.
This is real concurrent *independent* graph submission, not shared-weight
minibatch training, not gradient accumulation and not a guarantee of hardware
overlap. `task verify-gpu-independent-batch` checks bitwise serial/batch
parity, epochs, pending restrictions and new inputs; `task
benchmark-gpu-independent-batch` reports wall-clock serial vs queued replay
for four independent sessions. The GPU runtime remains CUDA Graph-only,
with no legacy streams or kernel launches outside Graph. Hardware acceptance
and throughput have not been executed by the agent.

**Still open:** a verified GPU-side compact finite-domain guard (the GPU
AdamW path still uses full D2H validation of changed Finite parameters),
more general multi-operation/region kernel fusion, tensorized *shared-weight*
minibatches with correct state separation and gradient aggregation, and a
sustained EMA4 end-to-end throughput comparison. Scheduling independent
programs does not automatically accelerate single-sequence EMA4 BPTT.

### Strict-order three-operation Sqrt→Add→Div fusion (code, hardware acceptance pending)

The opt-in architecture-neutral pointwise fusion pass now recognizes
`Div(numerator, Add(Sqrt(x), epsilon))` only when both intermediate nodes
are unobserved and have exactly one reachable consumer and all three source
tensors have identical shape. The original operand order of `Add` is
preserved. The generated three-input CUDA Graph kernel emits separate
`sqrt.rn.f32`, `add.rn.f32`, `div.rn.f32` instructions, retaining each
FP32 rounding boundary. The fusion removes two kernels per matched chain
but may not alter wall-clock time beneficially. A source-derived capability
plan explains both removed nodes, memory contracts validate three source
extents, and translation validation checks LLVM operand wiring and ordered
PTX instructions before device execution. No Lean IEEE/SASS theorem follows
from these structural checks.

`task verify-gpu-sqrt-add-div-fusion` exercises operand order, observed
intermediates, fan-out negatives, translation-plan serialization and actual
NVIDIA bitwise fused/unfused results. The generic GPU AdamW plan's unit
test checks that this fusion reduces its denominator expression. All GPU
and Cargo acceptance remain to be run by the developer. Full arbitrary
pointwise-region fusion and runtime-domain predicates remain open.

### P3 acceptance follow-up: rank-general scalar GPU broadcast (code, pending rerun)

The GPU `BroadcastScalar` memory contract and lowering already read source
index 0 and write one lane per output index. The former artificial target
shape restriction `[1,N]` has been lifted for scalar sources with exactly
one element, allowing zero-rank AdamW coefficients to broadcast to tensors
of any supported shape, including rank-1, rank-2 and rank-3 parameters.
General broadcasting of non-scalar sources remains GPU-unsupported and
fails closed, even where the Semantic IR shape rule permits it. CPU structural
and NVIDIA bit-pattern acceptance cases were added to `gpu_adamw`.
The AdamW GPU parity test now uses checked-domain feedback, retains both
checkpoint snapshots across restart, and specifies `f32` for floating
bit-pattern expectations. Hardware acceptance and throughput are not yet
established by these source repairs.

### P3 GPU AdamW hardware acceptance and measured throughput (developer-run)

On the tested NVIDIA system, `task verify-gpu-adamw` passed CUDA checks for
FP32 tolerance against the CPU AdamW oracle, GPU optimizer bitwise restart,
and finite-domain feedback validation. The four-window EMA4 checkpoint/resume
versus continuous training check passed byte-for-byte. `task check` and strict
`task clippy` passed; `task test` exposed a stale negative scalar
broadcast test in `structural_last_row.rs`, updated to accept scalar target
shape `[2,6]` while rejecting genuine axis-wise non-scalar broadcasting.

The first 256-window end-to-end benchmark on that system found **host AdamW
132.83 tokens/s** (3.854478 s, 6,892 post-fusion CUDA kernels per replay),
versus **GPU IR AdamW 61.17 tokens/s** (8.369581 s, 9,638 post-fusion
kernels). GPU AdamW was **2.17x slower** for this BPTT=2, width-48 model;
2,746 additional tiny kernels per replay are consistent with high GPU
scheduling overhead. This is not a profiler-proven breakdown. CPU AdamW
remains the default; GPU AdamW is experimental. Independent CUDA Graph
scheduling speedup is a separate benchmark, not single-sequence EMA4 speedup.

**Next performance acceptance criterion:** generic region/multi-output
pointwise fusion or true shared-parameter minibatch training must improve
wall-clock training throughput; a lower kernel count alone is insufficient.
Preserve FP32 order/rounding, memory contracts and hardware parity tests.

### Bounded four-op pointwise fusion, four-source CUDA Graph lowering (code, hardware pending)

The opt-in architecture-neutral compiler now recognizes the exclusive
shape-preserving subgraph `Div(Add(Mul(Mul(a,b), a|b), c), d)`.
It emits a single four-source CUDA Graph kernel with four separate
`mul.rn.f32`, `mul.rn.f32`, `add.rn.f32`, `div.rn.f32` instructions.
The repeated first multiplicand is reused in registers, and both operand
order selectors are represented explicitly in a versioned
`TranslationKind::MulMulAddDiv`. Each intermediate must have exactly
one reachable use, be unselected, and have the exact root shape. Shared
or observable intermediates fail to fuse. The capability report records all
three removed kernels, the memory contract bounds all four full-shape
sources, and the LLVM/PTX translation validator checks operand binding
and rounded instruction sequence. The CUDA runtime `launch4` captures
the kernel on the existing owned non-blocking stream, never the default
stream and never outside CUDA Graph capture.

This general pattern occurs in the second-moment update and bias
correction of the optional GPU AdamW, without introducing a specialized
optimizer kernel. Structural and independent NVIDIA bitwise comparisons
over all eight operand-order combinations are in
`task verify-gpu-four-op-fusion`; `task verify-gpu-adamw` checks optimizer
integration and restart. The end-to-end gate is
`task benchmark-gpu-adamw` using the same 256-window fixture as before.
The existing CPU baseline measured 132.83 tokens/s while the previous
GPU implementation reached 61.17 tokens/s. A smaller post-fusion kernel
count is not proof of faster execution; do not select GPU mode by default
unless measured sustained throughput improves. Full arbitrary-size
pointwise region fusion, source-sharing across arbitrary fanout, and
formal IEEE/SASS equivalence remain unsupported.

### Four-source selected-root diamond fusion after hardware regression (code, GPU pending)

The developer's first run of the bounded linear `Mul→Mul→Add→Div` fusion
passed its standalone NVIDIA FP32 parity test, but `task test` and the GPU
AdamW structural test failed: it expected one linear fusion while observing
zero. EMA4 reported `four_op_fusions=0` for both paths. The cause was
structural, not a CUDA rounding failure: the intermediate updated
second-moment `v` is a selected output for device feedback/checkpoints.
The linear rule must not erase that observable value. Measured on that
same run: CPU AdamW 158.17 tokens/s (3.236955 s/512 tokens, 6892 kernels);
GPU AdamW 66.05 tokens/s (7.752278 s, 9638 kernels). The GPU mode
remained significantly slower and is not recommended by default.

A separate architecture-neutral `MulDiamondAdd` rewrite now fuses
`Add(Mul(a,b), Mul(Mul(c,d), c|d))` when exactly three Mul values are
exclusive and full-shape. Unlike the linear Div-root rewrite, its Add
root remains a live output even if selected and can feed both the
next graph operation and feedback/checkpoint. The capability plan
checks the complete four-source memory contract and each removed
Mul producer; the versioned translation plan and LLVM/PTX audit
check every intermediate RN instruction and both operand orders.
The new kernel is captured via the existing four-input nonblocking
CUDA Graph API, with no standalone launch or lab-specific kernel.
`task verify-gpu-diamond-fusion` checks all eight selector combinations
plus selected/shared intermediate negative cases and NVIDIA bit parity.
GPU AdamW's structural fixture now requires exactly one diamond fusion
and **zero** linear Div-root fusion. EMA4 reports `diamond_fusions`.
The GPU AdamW acceptance and end-to-end benchmark must be rerun
after this change; no speedup is claimed yet. GPU-side finite-domain
validation without D2H remains open.

### Оптимизация GPU AdamW: групповой runtime и горизонтальное объединение IR

**Реализованный экспериментальный код, NVIDIA-приёмка и ускорение не подтверждены:**

- **P0:** EMA4 в режиме `TORUS16_GPU_ADAMW=1` не экспортирует градиенты как публичные outputs. Градиенты остаются внутренними зависимостями обновления AdamW. CPU-режим сохраняет исходный список выходов.
- **P0:** runtime `CudaSession` проверяет несколько тензорных доменов одной kernel-node на bucket и копирует независимые feedback-связи побитовым grouped CUDA Graph kernel вместо отдельной D2D node на тензор. Для обоих путей размерные bucket'ы и таблица дескрипторов фиксируются при захвате. Динамический guard возвращает на CPU status **до** любой записи; aliasing и невалидные адреса отвергаются заранее. CUDA/PTX остаётся TCB.
- **P1 (ограниченный срез):** `GpuLaunchConfig::with_internal_pointwise_fusion()` разрешает горизонтальное объединение двух независимых внутренних Add/Mul узлов одинаковой формы, с не более чем 64 просматриваемыми предыдущими кандидатами. Обе FP32 операции остаются независимыми и записывают два отдельных буфера. Используется существующий translation/memory/capability audit. В EMA4 опция включена только для экспериментального GPU AdamW. **Произвольный region fusion / grouped multi-tensor compute пока не реализованы.**
- **P2:** device-conditional feedback и tensorized shared-weight minibatching **не реализованы**. Первый требует согласованной семантики deferred host error и аппаратной проверки CUDA conditional nodes; второй требует доказанной границы агрегирования градиентов. Нельзя выдавать `prepare_independent` за shared-weight minibatch.

**Приёмка пользователем:** `task check`, `task test`, `task clippy`, `task verify-gpu-feedback-runtime`, `task verify-gpu-dual-pointwise-outputs`, `task verify-gpu-adamw`, `task benchmark-gpu-adamw`. В GPU тесты добавлены heterogeneous sizes, атомарный отказ guard, internal dual-output parity, а в обычные unit tests добавлены проверки bucket-плана и сохранения битов таблицы дескрипторов. Ускорение признаётся только по повторённой end-to-end производительности; количество kernels не является достаточным доказательством.


### Последующий runtime-срез: индекс выделений, стабильные планы, CUDA-event диагностика

В `CudaSession` таблица собственных device allocations теперь имеет адресный индекс `BTreeMap`, а порядок освобождения остаётся прежним. В неизменном захваченном `guard_graph` и `feedback_graph` валидированная таблица дескрипторов переиспользуется без повторного полного прохода по буферам. Любая смена адреса, размера или доменного требования выполняет новый fail-closed preflight и recapture. Проверка **самих FP32-значений** обязательна на каждом checked feedback, вне зависимости от кеша метаданных.

Для аппаратной локализации оставшихся затрат добавлен `GpuExecution::feedback_state_checked_device_profiled` и диагностический флаг лаборатории `TORUS16_GPU_FEEDBACK_EVENTS=1`. Флаг добавляет CUDA events вокруг **одного** исполнения grouped feedback Graph, не повторяя обновление. `feedback_device_event_ms` относится к stream elapsed time; `feedback_copy_sync` и `feedback_total` остаются host-wall измерениями. Первый запуск может включать ожидание host recapture между event record и graph launch; выводы следует делать на прогретых окнах. Включённый флаг добавляет явную event synchronization, поэтому **throughput в таком режиме не сравнивать** с обычным CPU/GPU benchmark. Без флага прежний путь не создаёт event-объекты и не получает новый барьер.

**Приёмка в dev: ожидается developer-run, здесь не подтверждена.** `task check`, `task test`, `task clippy`, `task verify-gpu-feedback-runtime`, `task verify-gpu-adamw`, затем `task benchmark-gpu-adamw` без event-флага. Отдельно `TORUS16_GPU_FEEDBACK_EVENTS=1 task benchmark-gpu-adamw` для сравнений `feedback_device_event_ms` vs host feedback time (скорость при флаге не учитывать). GPU tests проверяют повторный feedback, смену плана, профиль events и атомарный отказ домена. P2 device-conditional feedback и shared-weight minibatching пока не реализованы: эта работа не заменяет их и не расширяет Lean/FP32 сертификат.

**Проверка developer-run 2026-10-25 для предыдущего среза:** `task check`, `task test`, GPU feedback (3/3), GPU AdamW (4/4), побитовое checkpoint/resume совпадение прошли. `task clippy` остановился на `clippy::nonminimal_bool` в индексированной проверке адресов; эквивалентный предикат `Option::is_none_or` исправлен отдельным коммитом и ожидает повторной проверки. Один end-to-end benchmark на 256 окнах, BPTT=2: CPU 162,51 токена/с (3,150631 с), GPU IR 246,55 токена/с (2,076640 с), отношение скорости ≈1,52; измеренный `feedback_total` 0,6510 мс. `feedback_events=false`, то есть CUDA-event диагностика в этом прогоне **не выполнялась**; `feedback_device_event_ms=0` здесь не является длительностью GPU-копирования. Прирост требует нескольких повторов и сравнения медиан; Lean/IEEE/SASS доказательства этот запуск не добавляет.

### Продолжение: кешированный feedback preflight на уровне GpuExecution

Профиль разработчика с `TORUS16_GPU_FEEDBACK_EVENTS=1` подтвердил `feedback_device_event_ms=0.0285` мс при общем `feedback_total=0.6255` мс (256 окон, BPTT=2), `feedback_validation_d2h=0.1489` мс, `feedback_copy_sync=0.0751` мс. Остаток `0.4013` мс измерениями пока **не атрибутирован**. Проход по исходнику обнаружил повторное вычисление `feedback_copy_plan`, построение output lookup, списков domain-check и transfers на каждом вызове; это **гипотеза** для ускорения, а не доказанная причина всего остатка.

Новый ограниченный runtime-патч кеширует подготовленную схему `GpuExecution` по точному списку `(state,next)` и флагу runtime-domain checks, вместе с размерами, доменными требованиями и D2D transfers. Cache miss запускает полный предыдущий preflight; проверка актуальных FP32 битов на GPU остаётся **обязательной на каждом checked feedback**, включая повторы. Добавлена метрика `feedback_plan_host` для отделения host preflight/lookup от прочих этапов, а также GPU-regression `finite → finite → overflow` при неизменных связях для подтверждения отсутствия кеширования результата проверки значений.

**Проверки после патча пользовательские, ещё не выполнялись:** `task check`, `task test`, `task clippy`, `task verify-gpu-feedback-runtime`, `task verify-gpu-adamw`, `task benchmark-gpu-adamw`. Никакого device-conditional feedback, shared-weight minibatching или Lean/IEEE/SASS доказательства эта оптимизация не добавляет.

### Torus16-Mix: парное обучение LM с независимыми и преобразованными общими матрицами (код, developer-run ожидается)

**Исследовательская гипотеза (не вывод):** одна физическая матрица `W` для восьми ролей Torus16-Mix в каждом рекуррентном блоке может приближаться по качеству к восьми независимо обучаемым матрицам при меньшем числе параметров. Реализованное сравнение касается **разделения весов внутри Torus16-Mix**, а не одной общей матрицы для всей четырёхслойной EMA4-LM.

Лабораторный API `ema4::MatrixMode::{Independent,TorusShared}`, `architecture_model_for(vocab, mode)`, `Ema4Weights::{bindings_for,trainable_parameters_for,parameter_count_for}` строит обе версии исключительно через публичные Rust/VeriNeural ModelBuilder, Semantic Neural IR, autodiff BPTT, CUDA Graph и общий GPU IR AdamW. В `Independent` каждый из восьми projection role в каждом из четырёх EMA4-блоков имеет собственную матрицу 48×48; в `TorusShared` одна матрица 48×48 на блок преобразуется существующими роль-специфическими roll, transpose, row/col scale и gate. Остальные блоки, история Conv9, EMA память, embedding/head, роли, корпус и loss без изменений. **Все независимые W инициализируются побитовой копией W общего варианта**, чтобы начальные вычисления отличались только идентичностью объектов параметров, не значениями. Отсутствие численной ошибки GPU не является формальной теоремой о FP32.

На исследовательском корпусе 104 символа из существующего `corpus_snapshot.txt`: одинаковый хронологический 90/10 split, порядок токенов, нулевые состояния в начале и единые AdamW (lr 0.002, wd 0.01), BPTT=2. Параметров при 104-символьном словаре: **24 908** для `TorusShared` (4 матрицы), **89 420** для `Independent` (32 матрицы), прочие 15 692 параметра одинаковы. Эффект оценивается по `train_bpc` и его кривой (последовательные блоки по 1024 окна), по финальному `holdout_reset_streaming_bpc` на одном и том же validation fragment и по tokens/s/времени. Различия в числе параметров отражать явно: эксперимент сравнивает expressivity vs parameter efficiency, не равный budget и не доказывает превосходство архитектуры по одному seed.

Команда: `task compare-torus16-matrix-tying` (по умолчанию 16 384 окна × BPTT 2 и 512 validation chars, GPU AdamW; выполнение только локально на машине разработчика). Для продолжительного эксперимента доступны окружения `TORUS16_TRAIN_WINDOWS` (до 131072, с проверкой длины корпуса), `TORUS16_BPTT`, `TORUS16_VAL_LIMIT`, `TORUS16_CHECKPOINT_EVERY`, `TORUS16_MATRIX_ABLATION_DIR`, `TORUS16_CORPUS`. Результаты и отдельные checkpoints сохраняются в `target/torus16-matrix-ablation/{independent,torus_shared}.*`. Для одиночного прогона `TORUS16_MATRIX_MODE=independent|torus_shared` задаёт режим у существующего `ema4_bptt`, а старые вызовы по умолчанию сохраняют `torus_shared`. Модели отличаются явно именованными trainable W; попытка привязать независимые bindings к общей модели должна отвергаться.

Приёмка: `task check`, `task test`, `task clippy`, опционально `task verify-torus16-matrix-initial` для bitwise NVIDIA parity на начальных logits/EMA state, далее пользовательский `task compare-torus16-matrix-tying`. Проверять успех обоих прохождений, валидность checkpoint каждого режима, конечность train/holdout BPC, динамику train loss и полезность параметрического сжатия. **Никакой PyTorch parity, PyTorch зависимости или lab-specific CUDA kernel**. Никаких Lean-доказательств обучения/сходимости/оптимальности и статуса FP32/SASS certified не заявлять; результаты до локального прогона неизвестны.

### Torus16-Mix P1/P2: проверка обучения и воспроизводимые прогоны (патчи, developer-run ожидается)

Новый исследовательский срез сохраняет **только Rust + публичный VeriNeural API**: двухшаговый BPTT, autodiff, собственные CUDA Graph ядра VeriNeural, GPU IR AdamW, строгий режим проверки текущих доменов. Отдельных лабораторных kernels, Python/PyTorch либо PyTorch-parity нет.

**P1:**
- `Ema4Weights::deterministic_seeded(vocab, seed)` предоставляет воспроизводимую матричную инициализацию по целочисленному mixing, без скрытого RNG. `seed=0` сохраняет старые значения. `RoleInit::{Matched,Diverse}` для Independent: Matched оставляет побитово одинаковые исходные матрицы всех восьми ролей (контролируемое исследование weight tying), Diverse даёт независимые по роли воспроизводимые начальные значения. Остальные слои, embedding, decay, norm и роли не меняются между парными прогонами.
- `task verify-torus16-matrix-gradients`: отдельная NVIDIA численная проверка, что VJP по общей матрице совпадает в пределах явной FP32 погрешности с суммой восьми VJP по независимым ролям при Matched.
- `task verify-torus16-matrix-training`: аппаратный gate на настоящем 4-layer BPTT2/IR GPU AdamW и fused/unfused выходах/градиентах; в каждом режиме выполняет два окна, возобновляет до четырёх и сравнивает checkpoint с непрерывными четырьмя окнами побитово. Наблюдения не равнозначны Lean-сертификату GPU/FP32.
- `task compare-torus16-matrix-tying` по умолчанию 65 536 окон × BPTT2, 4096 holdout chars, validation каждые 8192 окна (и в конце) и checkpoint каждые 4096. Для полного train диапазона доступен `TORUS16_TRAIN_WINDOWS` до 1_000_000 с реальным ограничением длиной корпуса. Устанавливать `TORUS16_MATRIX_SEEDS="0 1 2"` для повторов; `TORUS16_INCLUDE_DIVERSE=1` включает третий, независимо инициализированный режим. Каждый seed берётся одинаковым в парных Matched-прогонах. Больше окон и повторов нужны для статистически содержательных выводов о holdout BPC, но также повышают время и расход GPU.

**P2:**
- Checkpoint **v2** содержит `architecture_fingerprint`, вычисленный из упорядоченных операций, входных зависимостей, формы, dtype, численных контрактов и профиля FP32 Semantic IR, выходов/состояний, seed/role_init и битовых значений начальных именованных тензоров. Сигнатура FNV64 служит **только детектором несовместимости**, не криптографической аттестацией и не доказательством корректности вычисления. Изменение математики, seed, значений constants или режима инициализации приводит к fail-closed reject до продолжения обучения.
- Checkpoint **v1 больше не читается** (небезопасная автоматическая миграция запрещена). Старые исследования остаются на диске; новый эксперимент начинается с чистого checkpoint в новой папке или при явно выбранном `TORUS16_FRESH`.
- При `TORUS16_EVAL_EVERY` выполняется reset-state chronological holdout на **одинаковом** фрагменте validation без обновления параметров. Validation boundary всегда сопровождается snapshot состояния и optimizer; печатаются `validation_window`, `holdout_reset_streaming_bpc`, `train_curve_window`. Лучший по этой метрике checkpoint сохраняется в `<name>.best.ckpt`; основной checkpoint продолжает фиксировать **последнюю** точку обучения и `best_validation_bpc` для корректного resume.
- `task compare-torus16-matrix-tying` возобновляет отдельные запуски из `<mode>-<init>-seed<seed>.ckpt` при наличии совместимого checkpoint (по умолчанию); `TORUS16_MATRIX_ABLATION_FRESH=1` принудительно создаёт свежий. Файлы `*.log`, `*.train_curve.log`, `*.validation_curve.log` хранят измерения; предшествующие interrupted log windows могут перекрываться при resume, поэтому для чистого графика использовать `validation_window` и подтверждённый cursor checkpoint, не считать каждую строку уникальным шагом. Счётчик времени при нулевых новых окнах не выдаётся за вновь измеренную производительность.

**Критерий приёмки:** `task check`, `task test`, `task clippy`, `task verify-torus16-matrix-initial`, `task verify-torus16-matrix-gradients`, `task verify-torus16-matrix-training`, затем `task compare-torus16-matrix-tying`. Это рекомендации для запуска на машине разработчика, **здесь ни Cargo, ни GPU обучение не выполнялись**. При ненулевых расхождениях/ошибках остановить исследование и исправлять причины. Никаких заявлений о сравнительном качестве моделей до фактических BPC-логов.

### Torus16-Mix: приоритетные исправления после P1/P2 code review (developer-run ожидается)

**P0: формат и компиляция.** Убран лишний позиционный аргумент вывода `experiment_matrix_mode=...`; теперь число аргументов соответствует шаблону. Эта правка сама по себе не означает успешный `cargo check`.

**P1: корректная постановка эксперимента.** `architecture_fingerprint` вычисляется **после построения фактического execution_graph** с BPTT-unroll, CE cotangents, reverse autodiff и (для GPU пути) общим `GpuAdamWPlan::append`; дополнительно включает профили optimizer backend, seed, role init, подписанные именами и ролями источники исходного `Model`, начальные значения входных параметров/констант, `TORUS16_VAL_LIMIT`, `TORUS16_TEST_LIMIT` и `TORUS16_EVAL_EVERY`. При resume полный v3 fingerprint и перечисленные параметры проверяются **до первого GPU replay**. Смена только длины `TORUS16_TRAIN_WINDOWS` не ломает продолжение, как и раньше. Checkpoint v1/v2 не мигрируют неявно. Fingerprint — FNV64 compatibility checksum, не криптографическое доказательство и не FP32/SASS-сертификат.

**P1: межархитектурная GPU-проверка.** `task verify-torus16-matrix-training` снимает **битовый снимок logits каждого шага BPTT2 и всех terminal recurrent states в первом training replay**, затем сравнивает их `cmp` между Matched Independent и TorusShared при одинаковых токенах и начальных параметрах. Fused/unfused output-gradient parity, независимые checkpoint/resume для каждого режима и 4-шаговая непрерывная контрольная траектория сохраняются. `task verify-torus16-matrix-gradients` проверяет **все четыре блока**, каждый — против суммы восьми role-specific VJP, с явно ограниченной FP32-погрешностью. Тест в этом режиме сравнивает один recurrent inference step; BPTT-градиенты двух архитектур не заявляются битово эквивалентными.

**P2: выбор и финальная оценка.** `holdout_bpc` оценивает интервалы одного и того же исторического validation split. Префикс длины `TORUS16_VAL_LIMIT` используется **только** для периодической validation/выбора лучшего checkpoint. Отдельный суффикс длины `TORUS16_TEST_LIMIT` используется **только** для итогового `test_reset_streaming_bpc` на параметрах **best-validation checkpoint**, а не последнего обучающего шага. Пересечение интервалов запрещено fail-closed. Значения по умолчанию у `task compare-torus16-matrix-tying`: 4096 validation + 4096 test chars; для GPU-smoke: 2 + 2. `TORUS16_FRESH` удаляет только старый best-checkpoint именно выбранного пути после успешной подготовки нового GPU-сеанса, чтобы исключить случайное использование прежнего best. Добавлены отрицательные тесты incompatibility для изменённого training IR, validation/test schedule и реально другой схемы Independent вместо синтетического `identity+1`.

Порядок пользовательской приёмки: `task check`, `task test`, `task clippy`, `task verify-torus16-matrix-initial`, `task verify-torus16-matrix-gradients`, `task verify-torus16-matrix-training`. Только затем долгий `task compare-torus16-matrix-tying`; при ошибке любого gate исследование останавливается. Реальные результаты BPC и прохождение CUDA-тестов пока **не подтверждены**; только статическая проверка опубликованных изменений.

## P0: reuse идентичных GPU kernel specialization (подготовленный срез, локальная приёмка ожидается)

`GpuEngine::compile_outputs` сопоставляет LLVM NVPTX-функции по **точному совпадению сгенерированного тела** после исключения исключительно синтетического имени `vn_node_{id}`. Таким образом, разные Semantic IR узлы с идентичным ABI, FP32-инструкциями, порядком операций и захваченными константами имеют один PTX entrypoint, сохраняя **отдельные CUDA Graph kernel nodes**, указатели, длины и зависимости. `GpuProgram::kernel_count()` продолжает возвращать число исполняемых graph nodes, `unique_kernel_count()` и `kernel_entry(node)` показывают число уникальных функций и отображение узла на владельца специализации.

`TranslationPlan` версии `v2` сохраняет `entry_node` для каждого логического узла и проверяет каждый его semantic/memory contract по LLVM/PTX-функции владельца. Missing/extra entrypoints, forward/unknown owner и нарушение FP32/shape templates отклоняются. Reader сохраняет совместимость с v1, в котором `entry_node = node`; старый v1-артефакт нельзя неявно переинтерпретировать как план с общими функциями. В опциональном `VERINEURAL_GPU_PREPARE_PROFILE` выводятся `graph_kernels` и `unique_functions` для атрибуции JIT-затрат.

Интернирование **не является kernel fusion**: число CUDA Graph nodes и потенциальная цена launch/replay остаются прежними. Цель среза — сократить модуль LLVM/PTX и работу JIT за счёт устранения идентичных entrypoints. Новые структурные и негативные регрессии находятся в `crates/verineural/tests/kernel_specialization_reuse.rs`; настоящий NVIDIA replay-тест помечен `#[ignore]` и требует отдельного запуска разработчиком. Показатели ускорения JIT, tokens/s, BPTT=16, численная GPU parity и Lean-сертификация аппаратного кода **не подтверждены**. После проверки этого среза отдельно профилировать узлы CUDA Graph и разрабатывать контролируемый fusion, не изменяющий FP32-порядок и контракты.

## Архивные отчёты о реализованных API-срезах из SPEC §3.3–3.5

### 3.3. Именованный frontend (реализованный узкий срез)

`ModelBuilder` предоставляет `input`, `parameter`, `block`, `output`, `proof_goal` и `finish`. `NeuralGraph::compose_inferred` выводит структурный контракт непосредственно из IR, проверяет границы и скрытые зависимости, атомарно фиксируя только корректные блоки. `Model::bind` проверяет полный набор именованных тензоров, формы и домены; `Model::plan` передаёт все выходы и явные цели в существующий `ArchitectureAutoCompiler`.

Это не отдельный IR и не Lean-доказательство без запуска checker. Роли input/parameter/state сохраняются как metadata над тем же IR, а temporal mutation реализуется только runtime-слоем из §3.5. Встроенного optimizer-а и автоматического доказательства FP32/SASS нет.

### P2 rotating-matrices: минимальные операции и подготовка consumer (код; локальная приёмка ожидается)

В `Tensor<T>` доступны `from_vec`, `from_slice`, `zeros` с общей `Shape::new`-валидацией. Host f32 reference дополнен `sub/div/exp/reduce_max` и явным правосторонним `broadcast_to`. Rust graph вводит `reduce_max(input, axis)` и `broadcast_to(input, Shape)`; `sub/div/exp` десахарятся в типизированный `PointwiseSpec` AST, в котором сохраняются идентификатор, версия и математическое выражение. Это не готовый softmax, пользователь компонует его из узлов сам.

Shape validation, обязательный `DType::F32`, детерминированная row-major перестановка и canonical IR/DOT видимость покрываются Rust-кодом и тестами. `reduce_max` сохраняет первый максимум, при NaN выдаёт NaN; broadcast повторяет только singleton-оси. Произвольный NumPy-like implicit broadcasting для арифметики не разрешается. `PointwiseExpr::Sub/Div/Exp` **не имеют** модели ideal-Int в Lean: доказательный backend отклоняет их до генерации Lean. `ReduceMax/BroadcastTo` также Lean/GPU unsupported, как и новые pointwise операции на GPU. Никакого фиктивного сертификата/заявления FP32/IEEE equivalence нет. Эти ограничения должны быть сняты отдельным доказуемым вертикальным срезом.

Лаборатория перенесена на `Tensor::from_slice`; повторная публикация certified bundle, SHA-256 и Graphviz остаётся исключительно за Stage 10 (`Model::certify`), без локального `write_architecture_manifest`, ручного SHA/copy. Structural/index routing P0 имеет узкий GPU lowering; локальная сертификация всё ещё требует реального Lean/LLVM прогона и не означает IEEE-754/JIT/SASS доказательства.

### 3.5. P1 state transition и named CUDA Graph replay (API-срез; локальные проверки ожидаются)

`ModelBuilder::state` регистрирует источник с ролью State; `next_state` объявляет один чистый переход с проверкой формы, типа и домена. Эти связи входят в identity и являются самостоятельными IR roots. `block_multi` выводит структурные контракты именованных выходов атомарно, без неявных математических доказательств.

`GpuEngine::prepare_model` принимает полные initial bindings и компилирует общий CUDA Graph. `ModelExecution` владеет копией только именованной runtime-metadata (source roles, outputs, state links), а не заимствует `Model`; поэтому consumer может владеть finished model и execution в одной структуре без self-referential lifetime. `set_input/set_parameter/set_state` обновляют только запрошенный буфер; `run` исполняет replay без изменения state; `step` после replay переносит `next_state → state` через явный `cuMemcpyDtoDAsync_v2` на non-blocking CUDA stream. Runtime ABI теперь сохраняет dtype: `TensorValue::F32` и `TensorValue::Index` используются `GpuExecution::read_value/read_values`, `GpuEngine::execute_value/execute_values` и `ModelExecution::output_value`; CUDA `Index` читается как `i32` и проверяемо преобразуется в host `usize`. Старые F32-only `read_output/read_outputs/execute/output(name)` сохранены для совместимости, но при `DType::Index` fail-closed через dtype mismatch вместо переинтерпретации битов как `f32`. `state` остаётся F32, потому что публичные source tensors текущего профиля F32. Прежний `GpuExecution` сохранён.

**Аппаратная приёмка P1, реализация теста:** `task gpu-state-replay` запускает в том же CUDA Graph модель с двумя публичными выходами и скрытым feedback-результатом. Независимые точные FP32 fixtures сравнивают результаты `run()` и пяти последовательных `step()`, проверяют накопление state после D2D, повторное использование parameters, обновления по имени, reset и fail-closed для неверных source role/shape/domain. CUDA Graph создаётся только при `prepare_model`, а не между шагами. Задача не входит в обычный `task test`, требует CUDA/LLVM и должна быть подтверждена локальным запуском.

Граница гарантий: IR остаётся чистым DAG; temporal replay не является Lean-доказательством рекуррентного алгоритма, исполнения CUDA или FP32/SASS. Unsupported GPU operations отклоняются. Для rotating-matrices добавлен узкий GPU lowering профиль structural/indexing P0: rank-2 transpose, rank-2 reverse по axis 0/1, contiguous concat по axis 0, rank-2 last-axis `ArgMax(First)` и scalar `Take` по axis 0. Для этого профиля memory contracts теперь порождают проверяемые Lean-обязательства границ фактических row-major адресов; `Take` требует явную предпосылку `selected < choices`. Остальные формы fail-closed; deterministic rotating-matrices runtime исполняет единый architecture graph через `ModelExecution`: sequence-wide parameters/dictionary загружаются именованно, token обновляется по шагам, а recurrent state переходит `next_state → state` D2D без host rebinding. Host чтение `next_state` остаётся только диагностикой Trace, не источником следующего шага. Stochastic/fixed-route training также использует named `ModelExecution`: host sampling выбирает один из восьми route-specific графов, но spatial rotation базового core строится `permute/reverse_axis` внутри IR, без host-derived `rotated_weight`. Все объявленные state sources обязательно входят в GPU execution roots, даже если `next_state` игнорирует прежнее состояние. До первого replay проверяется layout D2D feedback; публичные выходы, делящие GPU allocation с state, получают повторно используемые device snapshots до обратной связи. Поэтому `output(name)` после `step` относится к чистому вычислению до обновления state, включая `Reshape`-алиасы. Несколько feedback-копий с опасными взаимными alias-перекрытиями по-прежнему отклоняются. При ошибке CUDA D2D-очереди, переноса output snapshots либо stream synchronization `GpuExecution` становится необратимо `ExecutionPoisoned`: любые последующие replay, чтение, rebinding и feedback отклоняются до создания нового execution. Предварительная ошибка проверки формы/домена/aliased layout **не** отравляет сессию. Это fail-closed поведение, а не транзакционный rollback уже поставленных DMA: частично обновлённые значения после ошибки нельзя считать достоверными. Отдельный CUDA fault-injection не заявлен проверенным: Rust-тесты имитируют результат transfer/sync через общий health gate.

**P1 session-wide CUDA lifecycle (код, приёмка ожидается):** `ExecutionHealth` дополнен `OutputNotReady` и общей проверкой ошибок H2D, D2D, D2H, `cuGraphLaunch`, stream sync и benchmark. После успешной валидации перед началом обновления входа или GPU replay выходы становятся неготовыми; только успешно завершённый `replay()` переводит их в готовое состояние. Ошибка CUDA IO/launch/sync необратимо переводит execution в `ExecutionPoisoned`. Ошибка проверки формы/домена без обращения к CUDA не инвалидирует предыдущий результат и не отравляет сессию. Важное ограничение: CPU fault-injection тестирует health state, не драйвер NVIDIA. `output(name)` до первого replay либо после изменения входа выдаёт `OutputNotReady`, не читает неинициализированную память.

**P2 индексирование `dependency_scope()` (код, приёмка ожидается):** временный индекс `output NodeId -> declared module inputs` строится один раз на обход; обход далее идёт по обычным DAG-рёбрам и индексированным декларациям. Для нескольких module invocations с одним output зависимости объединяются, не теряются. Добавлены тесты сравнения с прежним обходом и неиспользуемых портов.

**P2 append transactions:** `NeuralGraph` заменяет полное клонирование IR при `apply_module`, `compose_inferred`, `compose_multi_inferred`, `scan_fixed` и `apply_multi_module` на контрольные длины append-only коллекций `nodes/modules/multi_modules/module_order`; RAII rollback срабатывает при `Err` и при unwind. Вложенный успешный блок откатывается вместе с ошибочным внешним. Identity графа и правила захвата сохраняются. Это не обещание асимптотической скорости GPU: сокращаются Rust CPU allocations при композиции.

**Stage 10 (реализованный API-срез, без подтверждённой приёмки):** `Model::certify(CertifiedBuildConfig::new(package_root))` и `Model::verify_certified(config, pinned_sha256)` используют те же Lean/GPU/SHA256 backend'ы, что `ArchitectureAutoCompiler`. Все явно объявленные ProofGoal обязательны; все named outputs требуют поддерживаемого lowering. `identity.txt` внутри SHA256SUMS связывает модель, роли и цели; его сериализация больше не зависит от Rust `Debug` и process-local `NodeId.graph`, а старый v1 Debug-shaped identity считается миграционно совместимым только если полные декларации совпадают после удаления исключительно allocation-specific `graph` чисел. `--artifact-root` в лабораторных примерах разрешается относительно package root. Появился Cargo external subcommand `cargo-verineural` (`build/verify --certified`, выбор package через `--manifest-path` и artifact root), который запускает единственный тонкий consumer `examples/<name>_certified.rs` (уникальное имя executable target в Cargo workspace); для использования как `cargo verineural` бинарник должен быть доступен в PATH, без установки доступна Taskfile-команда `task certified-build`. В корне выбранного package после сборки создаются относительные Unix symlink на опубликованный `auto-build/`: canonical IR, DOT, PNG, Lean, SHA256SUMS, report, identity. Существующие артефакты с проверенным id/version архивируются, чужие не заменяются. Первая миграция корневых файлов не полностью атомарна; Stage 10 нельзя считать завершённым до приёмки CLI и доказательного пути. Это не является формальным FP32/SASS proof.

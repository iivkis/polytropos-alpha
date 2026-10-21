# polytropos-alpha

Исследовательский проект **VeriNeural**: архитектурно-независимый конструктор нейросетевых вычислений и система их формальной верификации. На Rust пользователь задаёт **собственную математику**, собирает её в переиспользуемые вложенные блоки и крупную архитектуру. Из одного Semantic Neural IR должны генерироваться исполняемые программы для **NVIDIA GPU** и **Lean 4-модели** с локальными и глобальными доказательствами.

**VeriNeural не является готовой LLM и не будет поставлять Transformer, KV Cache, набор архитектурных LLM-компонентов, чат/serving/API, распределённую инфраструктуру и деплой.** Аналогичные пользовательские алгоритмы возможны при поддержке соответствующих универсальных примитивов. Минимальный CUDA runtime нужен исключительно для выполнения и проверки вычислений.

Требование к завершённости: не просто доказать отдельные примитивы, а связать их в **содержательную глобальную теорему об архитектуре** с явными предпосылками, численной семантикой и границами доверия GPU. Совпадение CPU/GPU в примере не является доказательством IEEE-754 либо работы железа.

## Состояние

**Этап 2/10 проверен локально пользователем:** форматирование, Rust check, 9 тестов, Clippy и пример прошли. **Этап 3/10: базовый Lean 4 proof backend подтверждён локальным запуском:** 14 Rust-тестов пройдены, Clippy и `cargo check` успешны, Lean проверил 25 обязательств. После корректировки `rustfmt` повторный `fmt-check` прошёл. **Этап 4/10: LLVM NVPTX/CUDA backend подтверждён пользователем на NVIDIA GPU: 6/6 FP32 значений совпали с CPU; 17 тестов, Lean 25 обязательств, fmt/check/clippy пройдены.** **Этап 5/10: объединённый отчёт локально подтверждён пользователем (Lean + NVIDIA GPU + SHA-256 evidence bundle).** **Этап 6/10: MatMul, BiasAdd и residual FFN локально подтверждены пользователем.** **Этап 7/10: контролируемый CUDA launch и экспериментальное Relu-fusion реализованы в коде; локальный прогон ожидается.** GPU и IEEE-754 формально не доказаны.

- `crates/verineural/` — импортируемая Rust-библиотека, не приложение.
- `Tensor<T, Shape>` — владеющий данными тензор, проверяющий размеры и длину буфера.
- `NeuralGraph` — append-only DAG из типизированных узлов; проверяет арность, формы, принадлежность узлов графу и ограничения доменов.
- `NeuralModule` и `MathContract` — компонуемые модули с проверкой формы и консервативно выводимого домена; ошибочная композиция отклоняется без изменения графа.
- `FP32Policy` — требования к FP32 округлению, FMA, subnormal и порядку редукции; пока **не доказанная семантика** выполнения.
- `OperationSemantics` — три целевых семантических слоя и исходная метка `GpuSemantics::NotLowered`; отдельный GPU compiler формирует LLVM IR/PTX для поддерживаемых узлов, без претензий на доказанное исполнение.
- CPU reference для Add, Mul, MatMul, BiasAdd, ReLU, Sqrt и Reshape. Это расчёты Rust `f32`, не доказанные exact-FP32 kernels.

## Rust API

Поддерживаемый входной язык пока **явный графовый Rust DSL**, а не произвольные Rust-функции, макросы или исходный код сторонних crates.

Пример построения графа:

```rust
use verineural::{Domain, NeuralGraph, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2, 2])?;
    let input = graph.input(shape, Domain::Finite);
    let positive = graph.relu(input)?;
    let result = graph.sqrt(positive)?;
    assert_eq!(graph.node(result)?.shape.dims(), &[2, 2]);
    Ok(())
}
```

Полный пример с пользовательским `NeuralModule` и `MathContract` находится в [examples/stage2_graph.rs](crates/verineural/examples/stage2_graph.rs).

**Граница гарантии:** `graph.input(shape, Domain::Finite)` регистрирует *предположение пользователя*, а не автоматически проверяет реальные данные. Для конкретного `Tensor<f32>` используйте `validate_domain` до исполнения. Гарантии модулей проверяются по консервативно выводимой информации графа; арифметическое переполнение означает, что два конечных входа не дают автоматически гарантированно конечный выход. Узлы `sqrt` без установленного неотрицательного конечного домена отклоняются. Общие редукции и GPU Sqrt пока не реализованы; GPU MatMul и BiasAdd добавлены в коде этапа 6, но новый end-to-end прогон ещё ожидает локального подтверждения. Наличие IR и контрактов не заменяет доказательство Lean. Для доказуемого подмножества теперь доступен отдельный генератор Lean; доказательством считается только результат успешного запуска Lean, а не генерация исходного файла.

## Локальная разработка

```sh
nix develop
task --list
task check
task test
task clippy
task fmt-check # cargo fmt + проверка; если .rs изменились, отдельный commit и git push origin
task example-stage2
task proof-stage3
task gpu-stage4
task cert-stage5
task verify-stage5
task neural-stage6
task gpu-stage7
```

**Поведение `task fmt-check`:** только при полностью чистом рабочем дереве запускаются `cargo fmt --all` и `cargo fmt --all -- --check`. Если rustfmt изменил отслеживаемые `.rs`, команда проверяет `origin/<текущая-ветка>`, добавляет только Rust-файлы, создаёт коммит `style(verineural): apply cargo fmt` и выполняет обычный fast-forward `git push` без `--force`. При грязном дереве, отставшей/опережающей ветке, ошибке форматирования или push происходит отказ; при отсутствии изменений коммит и push не выполняются. Прямой вызов `cargo fmt` и `task fmt` ничего автоматически не отправляет. При отказе push созданный локальный коммит сохраняется.

Согласно [AGENTS.md](AGENTS.md), сборка, тесты и другие тяжёлые команды выполняются **пользователем локально**, а не ИИ-агентом без отдельного разрешения. Для будущего GPU-исполнения необходим совместимый NVIDIA GPU и системный NVIDIA-драйвер.

[SPEC.md](docs/verineural/SPEC.md) описывает всю конечную систему; [ROADMAP.md](docs/verineural/ROADMAP.md) — последовательность 10 этапов. LLVM IR/PTX генерируются для ограниченного GPU-подмножества. Stage 5 добавляет отчёт с уровнями доказательств и проверками целостности, но не формальный сертификат GPU-исполнения.

## Proof Engine: этап 3

В `lean/VeriNeural/Basic.lean` находятся определения плоских тензоров `Fin n → Int` и леммы для Add/Mul/ReLU/Reshape. `LeanProofEngine::generate(&graph, output)` порождает Lean-модель достижимой части графа и теоремы о формах, поэлементной семантике и интерфейсах модулей. `LeanProofEngine::verify(&model, path)` записывает Lean-файл и запускает `lean`. Только успешное завершение **без диагностических сообщений** возвращает `CheckedLean`.

Пример:

```sh
task proof-stage3
```

После проверки Lean-файл сохраняется в `target/verineural/verification/stage3.lean`. Если исполняемый `lean` отсутствует либо найдена неподдерживаемая операция, proof step завершается ошибкой. Никакого `sorry`, `admit` или новых аксиом в кодогенераторе нет.

**Границы доказательства:**
- Модель работает над **идеальными `Int`**, без переполнения, NaN, rounding, FMA и FTZ. Это ещё **не доказательство Rust f32**.
- В проверенном примере этапа 3 использованы **Add, Mul, ReLU, Reshape**; доказательства относятся к математической модели, формам и неотрицательности. На момент завершения этапа 3 MatMul и Sqrt отклонялись. MatMul/BiasAdd позднее добавлены в коде этапа 6, Sqrt по-прежнему не поддерживается proof backend'ом.
- Объявление `Domain::NonNegativeFinite` на входе становится явной гипотезой Lean, а не доказанным фактом о данных. `Finite` в идеальной целочисленной модели не заменяет численный контракт FP32.
- Lean-модели строятся по графовому Rust DSL, а не автоматически по произвольному телу Rust-функции. Генератор IR → Lean, бинарник Lean и стандартная библиотека входят в доверенную границу; формального доказательства корректности самого кодогенератора пока нет.
- Успешная Lean-проверка даёт условные математические теоремы для поддерживаемого графа, но **не сертификат корректности GPU**.

## GPU Compiler и CUDA Driver: этап 4

`GpuEngine::compile(&graph, output)` генерирует LLVM NVPTX IR из достижимой части графа. `GpuEngine::execute(&program, inputs, output_dir)` сначала проверяет формы и реальные входные домены, затем понижает IR в PTX с помощью `llc -march=nvptx64`, загружает системную `libcuda.so.1` и выполняет kernels через CUDA Driver API. Выделенные GPU-буферы, PTX-модуль и CUDA-контекст освобождаются и при ошибке.

В подтверждённом примере этапа 4 исполнялись поэлементные `Add`, `Mul`, `ReLU` (отдельные kernels), `Reshape` переиспользует тот же буфер. Число элементов не больше `i32::MAX`. На момент этапа 4 MatMul и Sqrt отклонялись; MatMul/BiasAdd позднее добавлены в коде этапа 6, а Sqrt пока остаётся неподдерживаемым для GPU. Все данные для примера конечны, используется сравнение с независимой CPU-реализацией по битам.

```sh
nix develop
task fmt-check
task check
task test
task clippy
task gpu-stage4
```

Артефакты: `target/verineural/gpu-stage4/graph.ll` и `graph.ptx`. Нужны Linux, NVIDIA GPU, установленный драйвер и LLVM `llc` из `flake.nix`. Архитектуру можно указать через `GpuEngine::new("llc", "sm_86")`; значение по умолчанию `sm_52`.

**Граница гарантии:** успешное совпадение CPU/GPU является численной проверкой примера, не математическим доказательством IEEE-754 либо исполнения GPU. Lean этапа 3 работает над идеальными `Int`, LLVM/PTX, JIT-драйвер и железо ещё не входят в доказанную цепочку.

## Proof + Execution Integration: этап 5 (локально подтверждён)

`CertificateEngine::run(&graph, output, &inputs, &cpu_reference, dir)` использует один Rust Semantic Neural IR для генерации Lean 4 и LLVM NVPTX/PTX, вызывает Lean checker, исполняет kernels через CUDA Driver API и сравнивает выходные FP32 **по битам** с независимым CPU reference. При ошибке старые `report.json` и `SHA256SUMS` удаляются. Успешный результат даёт проверяемый evidence bundle, но **не машинный сертификат**.

```sh
nix develop
task fmt-check
task check
task test
task clippy
task cert-stage5
task verify-stage5
# Для защиты от подмены манифеста подставить напечатанный ранее хеш:
VERINEURAL_STAGE5_SHA256=PASTE_DIGEST task verify-stage5
```

Результат в `target/verineural/stage5/`: `graph.ir`, `proof.lean`, `graph.ll`, `graph.ptx`, `inputs.bits`, `output.bits`, `report.json` и `SHA256SUMS`. Манифест фиксированного состава связывает все семь артефактов SHA-256. При первом запуске запомните SHA-256 **манифеста** отдельно от каталога. Без внешней фиксации можно переписать и файлы, и их хеши, подписи нет. `task verify-stage5` пересчитывает хеши и повторно запускает Lean **без GPU**, но не перепроверяет наблюдение CUDA и не доказывает связь Lean ↔ машинный код.

В JSON установлены разные статусы: `mathematical=proven`, `composition=proven` **только** для условной идеальной `Int` модели Lean; `numerical=validated` и `gpu_execution=validated` только для записанных входных FP32 битов; `compilation=trusted`; `fp32_semantics=unsupported`, `machine_code_correctness=unsupported`. Записываются предположения и реальные входы. TCB: Rust сериализатор и Lean-генератор, Lean kernel, LLVM `llc`, CUDA драйвер/JIT, GPU, CPU oracle, `sha256sum`.

Этот этап не включает доказательство корректности lowering, памяти GPU, IEEE-754, или авторства JSON. Неподдерживаемые операции/отвергнутый Lean/несовпадение результатов прерывают выпуск отчёта. Для чистой локальной проверки при необходимости удалите прежний каталог `target/verineural/stage5` перед повторным прогоном.

## Neural primitives: этап 6 (локально подтверждён)

Rust DSL расширен rank-2 row-major MatMul (`graph.matmul`) и прибавлением rank-1 bias к каждой строке (`graph.bias_add`, без произвольного broadcasting). `Dense`, `ResidualAdd` и `ResidualFfn` в `crates/verineural/src/modules.rs` служат **демонстрацией универсальной композиции**, а не обязательными компонентами разрабатываемой типовой LLM. Контракты проверяют точные формы; неверное построение модуля не меняет граф. Параметры являются обычными входами, а не зашитыми в GPU-выполнение константами.

`ResidualFfn::new(batch, features, hidden)` принимает `X [batch, features]`, `W1 [features, hidden]`, `b1 [hidden]`, `W2 [hidden, features]`, `b2 [features]`. Результат: `X + (ReLU(X × W1 + b1) × W2 + b2)`. CPU reference независимо вычисляет эти операции над `Tensor<f32>`.

```sh
git pull --ff-only
task fmt-check
task check
task test
task clippy
task neural-stage6
```

`neural-stage6` выполняет тот же IR через Lean 4 и NVPTX → PTX → CUDA и создаёт отдельный bundle в `target/verineural/stage6/`. Старые артефакты `task cert-stage5` не затрагиваются.

**Граница гарантии:** Lean фиксирует MatMul/BiasAdd над идеальными `Int`, проверяет формы, интерфейсы модулей и поузловые равенства по определениям. Для обращения по индексам идеальная модель использует `flatGet` (при выходе за границы возвращает ноль); формального доказательства всех адресов матричного ядра и соответствия модели Rust/GPU нет. Поузловые равенства не доказывают численную корректность GPU MatMul, memory safety, rounding/FMA/FTZ или машинный код. Накопление матричной суммы слева направо; оптимизации tensor cores не применяются намеренно, но strict FP32 не доказан.

**Граница текущего среза:** иерархическая архитектура с пользовательскими глобальными теоремами, произвольные новые математические операции, циклы/состояние и доказательство FP32/GPU lowering ещё не реализованы. Attention, KV Cache, MHA/GQA, MoE и другие готовые архитектурные LLM-блоки **не планируются как обязательный каталог**.


## Controlled GPU fusion: этап 7 (локальная проверка ожидается)

Из единого Rust IR доступны две CUDA-конфигурации: обычная без изменений графа и **экспериментальная** с локальным слиянием `Relu(Add(x,y))`, `Relu(Mul(x,y))` или `Relu(BiasAdd(matrix,bias))`. Оптимизация допустима только если бинарный producer имеет одного потребителя в достижимом графе. Shared DAG nodes и промежуточный `Reshape` не сливаются; `MatMul` не фьюзится. Слияние пропускает один промежуточный kernel и device allocation, но профит нужно измерять на целевой GPU.

Публичный API:

```rust
use verineural::{GpuEngine, GpuLaunchConfig, ReluFusion};

let baseline = GpuEngine::default()
    .with_launch_config(GpuLaunchConfig::new(128, ReluFusion::Disabled)?);
let experiment = GpuEngine::default()
    .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental)?);
```

Разрешены 64, 128, 256 и 512 threads/block. Неизвестные размеры отклоняются. Фактически применённые оптимизации доступны через `GpuProgram::fused_relus()`, launch profile через `launch_config()`. В `report.json` (по-прежнему схема stage5.v1) сохраняются `gpu_launch.threads_per_block`, `relu_fusion` и `applied_fusions`; этот JSON привязан к SHA256SUMS.

```sh
task gpu-stage7
```

Пример запускает один произвольный граф с двумя профилями, проверяет исходную Lean-модель графа и отдельные **идеальные целочисленные** леммы `lean/VeriNeural/Fusion.lean`, выполняет CUDA и требует битового совпадения обоих результатов с CPU. Папки: `target/verineural/stage7/baseline`, `.../fused`, отдельный `.../fusion.lean`. **Важное ограничение:** отдельный `fusion.lean` пока не входит в SHA256SUMS отчёта Stage 5, хотя проверяется Lean. Время замеряется для всей последовательности компиляции и проверки, это **не GPU-kernel benchmark**, и различие 64/256 threads также влияет на сравнение.

Это **не** доказательство эквивалентности оптимизированного FP32/PTX, отсутствия гонок или корректности машинных инструкций. Compiler/JIT/GPU остаются в TCB; `relu_fusion` разрешается только в явном экспериментальном режиме. Целевые формальные memory/race/translation-validation доказательства, tiling, scheduling, fused MatMul и подтверждённые FP32-оптимизации ещё не реализованы.

## Git-процесс

Изменения интегрируются в `main` **только fast-forward**, без pull request, merge-коммитов, squash и force push. Полные правила для агентов — в корневом [AGENTS.md](AGENTS.md).

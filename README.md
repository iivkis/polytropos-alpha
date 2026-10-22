# polytropos-alpha

Исследовательский проект **VeriNeural**: архитектурно-независимый конструктор нейросетевых вычислений и система их формальной верификации. На Rust пользователь задаёт **собственную математику**, собирает её в переиспользуемые вложенные блоки и крупную архитектуру. Из одного Semantic Neural IR должны генерироваться исполняемые программы для **NVIDIA GPU** и **Lean 4-модели** с локальными и глобальными доказательствами.

**VeriNeural не является готовой LLM и не будет поставлять Transformer, KV Cache, набор архитектурных LLM-компонентов, чат/serving/API, распределённую инфраструктуру и деплой.** Аналогичные пользовательские алгоритмы возможны при поддержке соответствующих универсальных примитивов. Минимальный CUDA runtime нужен исключительно для выполнения и проверки вычислений.

Требование к завершённости: не просто доказать отдельные примитивы, а связать их в **содержательную глобальную теорему об архитектуре** с явными предпосылками, численной семантикой и границами доверия GPU. Совпадение CPU/GPU в примере не является доказательством IEEE-754 либо работы железа.

## Состояние

**Этап 2/10 проверен локально пользователем:** форматирование, Rust check, 9 тестов, Clippy и пример прошли. **Этап 3/10: базовый Lean 4 proof backend подтверждён локальным запуском:** 14 Rust-тестов пройдены, Clippy и `cargo check` успешны, Lean проверил 25 обязательств. После корректировки `rustfmt` повторный `fmt-check` прошёл. **Этап 4/10: LLVM NVPTX/CUDA backend подтверждён пользователем на NVIDIA GPU: 6/6 FP32 значений совпали с CPU; 17 тестов, Lean 25 обязательств, fmt/check/clippy пройдены.** **Этап 5/10: объединённый отчёт локально подтверждён пользователем (Lean + NVIDIA GPU + SHA-256 evidence bundle).** **Этап 6/10: MatMul, BiasAdd и residual FFN локально подтверждены пользователем.** **Этап 7/10: CUDA Graph-only, reusable CUgraphExec, экспериментальный chain fusion (3→1 kernels), CUDA-event benchmark и корректное владение CUDA context локально подтверждены пользователем (34 Rust-теста, все Stage 4–7 проверки). Новый срез: обновление входов без пересборки графа и benchmark четырёх размеров реализованы, ожидают локального прогона.** GPU и IEEE-754 формально не доказаны.

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
task verify-gpu-stage7
task gpu-stage7-bench
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


### NVIDIA Driver в NixOS devShell

Для GPU-кода необходима **реальная системная** `libcuda.so.1` из установленного NVIDIA-драйвера, а не stub-библиотека из CUDA Toolkit. На NixOS она обычно доступна по пути `/run/opengl-driver/lib/libcuda.so.1`. `CudaSession` сначала пробует стандартное имя через `dlopen`, затем этот абсолютный путь; глобальный `LD_LIBRARY_PATH` не изменяется.

При ошибке загрузки на хосте проверяйте:

```sh
nvidia-smi
ls -l /run/opengl-driver/lib/libcuda.so.1
```

Отсутствие файла или неработающий `nvidia-smi` означает проблему конфигурации драйвера на хосте; `cuda_nvcc` и `cuda_cudart` из `flake.nix` не устанавливают драйвер. Сначала должен пройти `task gpu-stage4`, только затем имеет смысл запускать `task cert-stage5`, `task verify-stage5`, `task gpu-stage7` и `task verify-gpu-stage7`: команды `verify-*` не восстановят манифесты после неудачного GPU-прогона.

## CUDA Graph-only execution (после запрета legacy streams)

VeriNeural теперь **не использует CUDA Legacy Streams**: CUDA-kernels захватываются в **CUDA Graph** и исполняются через `cuGraphLaunch` на явно созданном `CU_STREAM_NON_BLOCKING`. Host↔device копирование использует `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` с pinned host buffers на том же потоке, а ожидание выполняется `cuStreamSynchronize`. Память GPU выделяется до захвата, чтобы CUDA Graph не перехватывал операции выделения. Если CUDA Graph API отсутствуют или создание графа завершилось ошибкой, выполнение останавливается без legacy fallback. `GpuEngine::prepare` один раз компилирует/загружает PTX, выделяет buffers, захватывает граф и создаёт `CUgraphExec`; `GpuExecution::replay` затем повторно вызывает `cuGraphLaunch` без повторной сборки и инстанцирования графа. Обычный `execute` использует тот же путь с одним replay. Для графов из одних входов/Reshape GPU kernels отсутствуют, поэтому запускается только необходимая передача данных без пустого графа. Поле `gpu_launch.execution` в SHA-256-отчёте по-прежнему фиксирует graph-only non-blocking-stream режим.

Старые версии этапов 4–6 выполнялись с `stream = 0`, но после перехода на CUDA Graph-only пользователь повторно подтвердил Stage 4–7 на NVIDIA GPU. Изменение runtime не доказывает FP32/LLVM/PTX/железо. Повторная передача входов через `GpuExecution::update_inputs` сохраняет прежний `CUgraphExec` и адреса device-буферов: метод сначала валидирует **весь** набор входов и только потом выполняет асинхронные копии на принадлежащем session non-blocking stream. Перед возвращением синхронизирует копии и возвращает завершённые pinned staging buffers в **пул повторного использования**. Пул сохраняется на время жизни `GpuExecution` и очищается при его освобождении; pending buffers никогда не переиспользуются до `cuStreamSynchronize`. Вместо новых private CUDA context на каждое выполнение `CudaSession` удерживает ссылку на **общий primary context** данного GPU через `cuDevicePrimaryCtxRetain`/`cuDevicePrimaryCtxRelease`. Потоки, PTX-модули, графы и память остаются отдельными для каждого сеанса. Контекст активируется через push/pop на время Driver API операций, а все kernels и transfers по-прежнему работают только на явно созданном non-blocking stream. Неверные формы, чужие handles, пропущенные или не удовлетворяющие входному домену данные не изменяют буферы. Ошибка CUDA после начала передач не обещает атомарного rollback.

## Controlled GPU fusion: этап 7 (Graph reuse и chain fusion проверены локально)

Из единого Rust IR доступны две CUDA-конфигурации: обычная без изменений графа и **экспериментальная** с локальным слиянием `Relu(Add(x,y))`, `Relu(Mul(x,y))`, `Relu(BiasAdd(matrix,bias))`, а также эксклюзивной цепочки `Add/Mul → BiasAdd → Relu`. Для цепочки оба промежуточных producer должны иметь ровно одного достижимого потребителя; тогда три kernels сводятся к одному трёхвходному kernel. Shared DAG nodes и промежуточный `Reshape` не сливаются; `MatMul` не фьюзится. Отдельные Lean-леммы фиксируют идеальную `Int`-семантику двух новых цепочек, но не доказывают FP32/PTX.

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

Пример запускает один произвольный граф с двумя профилями, проверяет исходную Lean-модель графа и отдельные **идеальные целочисленные** леммы `lean/VeriNeural/Fusion.lean`, выполняет CUDA и требует битового совпадения обоих результатов с CPU. Папки: `target/verineural/stage7/baseline`, `.../fused`, отдельный `.../fusion.lean`. **Доказательная привязка:** отдельный `target/verineural/stage7/SHA256SUMS` связывает `fusion.lean`, `memory_baseline.lean`, `memory_fused.lean` и оба вложенных `SHA256SUMS`. `task verify-gpu-stage7` перепроверяет оба манифеста, сохранённые Lean-файлы, математические правила fusion и индексные обязательства без повторного CUDA-запуска. Для защиты от совместной подмены сохраняйте напечатанный хеш внешне; при повторной проверке передайте `VERINEURAL_STAGE7_SHA256=<digest>`. Это не доказывает корректность PTX/FP32. Stage 7 теперь сравнивает baseline и fusion при одинаковых 256 threads/block. После отдельной проверки Lean/CPU evidence создаются два подготовленных `CUgraphExec`; 25 warmup и 1000 измеряемых replay засчитываются CUDA events на том же non-blocking stream. В этот интервал **не входят** Lean, LLVM/NVPTX compilation, создание CUDA context, JIT/module setup, allocations и host↔device copies. Это измерение времени GPU stream для graph replay, а не доказательство ускорения на других GPU или других размерах. На примере **всего шести FP32-элементов** fusion оказался медленнее baseline (примерно 14.788 против 9.726 мкс в среднем). Причина пока не установлена; при крошечных kernels измерение может включать паузы CPU submission.

### Предварительная проверка GPU-памяти и Lean-обязательства индексации (новый срез)

`GpuProgram::memory_contracts()` описывает **каждый реально запускаемый** kernel после fusion: исходные узлы и количество читаемых FP32-элементов, записываемый диапазон, индексную схему `Pointwise`, `BiasAdd` или `MatMul`. Компилятор отклоняет несоответствие размеров источников, матричных размерностей, нулевое число столбцов, возможное переполнение `u32` при построении сетки и ссылки на будущие узлы. CUDA runtime перед `cuStreamBeginCapture` сверяет логические диапазоны с **реально выделенными device-буферами** и отказывает, если выход пересекается со входом либо два kernel узла пишут в один буфер. Zero-copy Reshape остаётся допустимым алиасом только для чтения.

Из тех же проверенных форм и оптимизаций `GpuProgram::memory_lean_source()` генерирует Lean 4-файл с конкретными размерностями. `lean/VeriNeural/Memory.lean` содержит общие леммы о доступе pointwise и bias (`index % columns < columns`), адресах обоих входов row-major MatMul и уникальности flat index для двух потоков одного блока. Команда `task gpu-stage7` проверяет **отдельные** `memory_baseline.lean` и `memory_fused.lean` реальным Lean checker'ом и фиксирует их в Stage 7 SHA-256 manifest. `task verify-gpu-stage7` повторно проверяет файлы без GPU.

**Граница доказательства:** Lean доказывает арифметические свойства натуральных индексов при указанных предпосылках. Соответствие LLVM/PTX этим индексам, отсутствие гонок между произвольными GPU streams, runtime/device pointer safety и FP32 **не** доказаны. Проверка соответствия форм и выделений в Rust входит в TCB. Формальное завершение Stage 7 и переход к Stage 8 пока не объявлены; новые проверки ожидают локального запуска пользователем.

### Обновление входов и масштабируемые GPU-измерения

`GpuEngine::prepare` возвращает `GpuExecution` с уже инстанцированным графом. Новый `GpuExecution::update_inputs(&[(NodeId, Tensor<f32>)])` заменяет **все** достижимые исходные входы (с проверками графа, форм, доменов и полноты набора), не меняя `CUgraphExec`; далее используются обычные `replay()` и `read_output()`. Передачи и синхронизация выполняются вне CUDA-event timed interval. Замена части входов отдельным вызовом пока не поддерживается.

```sh
task gpu-stage7-bench
```

Отдельный пример `stage7_benchmark.rs` компилирует одну и ту же цепочку Add → BiasAdd → Relu в обычном и fused вариантах для **1 024, 16 384, 262 144 и 1 048 576** FP32 элементов. Используются одинаковые 256 threads/block, пять серий на размер и режим, в каждой по 25 прогревочных и 1000 измеряемых replay; порядок режимов чередуется, печатаются min/median/max и отношение медиан. До измерения пример сравнивает оба режима с CPU по битам, обновляет входы без пересоздания CUDA Graph, повторно сравнивает результаты и проверяет, что NaN-вход в домене `Finite` отвергается без замены предыдущих данных. Среди тестовых значений есть `±0` и минимальные нормальные `f32`-числа; это только конечный набор наблюдений, **не** доказательство полной IEEE-754 семантики.

Вывод benchmark измеряет CUDA-event интервал на GPU stream. Паузы между host submissions могут влиять на результаты, особенно у коротких kernels; компиляция, context/JIT, аллокации, копирование и CPU/Lean проверки в измерение не входят. Артефакты лежат в `target/verineural/stage7-workload/` и **не** входят в сертифицированные evidence bundles этапа 7. Пользователь подтвердил полный GPU-прогон предыдущей версии: на 1 048 576 элементах CUDA-event медианы 28.884 мкс (baseline) и 14.629 мкс (fused), то есть ≈1.97× быстрее без профилировщика. При Nsight Systems трассировка node ухудшает наблюдаемые времена, поэтому запуск с `nsys` используется для диагностики, а не итоговых performance-цифр. В **новом, ещё не проверенном локально** варианте benchmark отдельно печатает время подготовки `prepare` (включая llc/JIT/Graph capture) и min/median/max **полного цикла** `update_inputs → replay → read_output` с H2D/D2H и CPU wall-clock, не включая проверки CPU-эталона в измеряемый интервал.

Это **не** доказательство эквивалентности оптимизированного FP32/PTX, отсутствия гонок или корректности машинных инструкций. Compiler/JIT/GPU остаются в TCB; `relu_fusion` разрешается только в явном экспериментальном режиме. Целевые формальные memory/race/translation-validation доказательства, tiling, scheduling, fused MatMul и подтверждённые FP32-оптимизации ещё не реализованы.

## Stage 8: численная верификация FP32

На коммите `4414613` пользователь локально подтвердил расширенный целочисленный срез Stage 8: 49/49 Rust-тестов, fmt/check/clippy, Lean, baseline/fused CUDA Graph для аддитивного графа и MatMul-графа, повторную проверку без GPU и SHA-256. Получено 20/20 побитовых совпадений с независимым `i64`-эталоном. Подтверждённые границы идеальной целочисленной модели: `±3 145 728` для `Add → BiasAdd → Relu → Reshape` при входах `±2^20` и `±1 544` для `MatMul → Mul → BiasAdd → Relu` при входах `±8`. Зафиксированный манифест этого прогона: `66978a69a62d8ca0e642c25b30daa067024d3ec74546e0ee5e40af67777c2040`.

### Точное целочисленное подмножество

`ExactFp32Plan` выводит из **того же Semantic Neural IR** диапазоны для `Input/Add/Mul/MatMul/BiasAdd/Relu/Reshape`. Входные `f32` должны быть конечными целыми числами; отрицательный ноль отвергается. `compile` использует исходную границу `2^20`, а `compile_with_input_bound` позволяет сузить её для произведений и редукций. Любой математический результат, произведение или частичная сумма MatMul, выходящие за `[-2^24, 2^24]`, приводят к fail-closed.

`lean/VeriNeural/Numeric.lean` содержит универсальные леммы диапазонов для сложения, умножения, row-wise bias, ReLU, reshape и left-to-right MatMul. Независимый `i64`-эталон не использует `f32`-арифметику до финального точного преобразования представимого целого.

```rust
use verineural::ExactFp32Plan;

let plan = ExactFp32Plan::compile_with_input_bound(&graph, output, 8)?;
let expected = plan.integer_reference(&inputs)?;
let model = plan.lean_source()?;
```

### Дробный dyadic RNE и диапазонные численные контракты

Дробный `Mul/MatMul` срез локально подтверждён пользователем после formatting-коммита `6806fe2`: `cargo check`, 54 Rust-теста, Clippy, `task numeric-stage8` и `task verify-numeric-stage8` прошли. Baseline/fused fractional MatMul побитово совпал с независимым integer RNE oracle, Lean проверил отдельные multiply/add witnesses, а новый SHA-256 manifest равен `87f3d550c94aee99e5733626eb888f8c449a319af7f762a1ac84c8d450396746`. Контрольный MatMul сохраняет `q_in=20 → q_out=40`, ideal 140 737 521 909 762 ticks, rounded 140 737 521 909 760 ticks, observed error 2 ticks и conservative `ε=41 943 040` ticks.

`RneFp32Plan` остаётся concrete-trace механизмом для `Input/Add/Mul/MatMul/BiasAdd/Relu/Reshape`: входы декодируются из binary32-битов, операции моделируются integer RNE oracle, а MatMul явно использует separate rounded `fmul → fadd` в left-to-right порядке. Машинная связь с LLVM/PTX/JIT/GPU по-прежнему не доказана и проверяется лишь побитовым сравнением конкретных запусков.

Подтверждённый `RneRangePlan::compile(&graph, output, source_fraction_bits, source_bound_ticks)` поднимает ту же модель от одного concrete trace к **переиспользуемому диапазонному контракту**. На `27db27b` пользователь подтвердил fmt, 56 Rust-тестов, полный `task numeric-stage8` и offline `task verify-numeric-stage8`; manifest 32 evidence-файлов имеет SHA-256 `725e652961cb14a65b2b0d22578fadfad18aff71fb705356c147ef426f44588e`. Для каждого reachable node выводятся четыре величины: dyadic scale, bound идеального значения, абсолютный `ε` и bound фактического значения `ideal_bound + ε`. Worst-case ULP определяется из максимального pre-round magnitude диапазона, а overflow, слишком тонкая product lattice и неподдержанные операции fail-closed.

В Lean введён `ApproxContract` и доказаны reusable transfer rules: `binary32_rne_shift_le_of_bound` ограничивает допустимый RNE shift по входному диапазону; `approx_contract_scale`, `approx_contract_add_rne`, `approx_contract_mul_rne`, `approx_contract_relu` переносят bound/ε через операции. Генерируемый `numeric_range.lean` использует **произвольные Int-значения**, а не зафиксированные примеры, и инстанцирует эти теоремы конкретными bounds Semantic Neural IR.

**ReLU threshold-margin slice локально подтверждён пользователем после Lean fix `2a6531e`.** `RneRangePlan::classify_relu_input(relu, observed_f32)` использует масштаб и ε аргумента конкретного ReLU, возвращает `Positive` при `observed > ε`, `NonPositive` при `observed + ε ≤ 0`, в остальных случаях `Indeterminate`. Дополнительно возвращается допустимый интервал идеального pre-activation в tick-единицах. Узел обязан входить в output slice; finite normal binary32 должен точно лежать на dyadic lattice и в выведенном actual bound. `-0`, subnormal, NaN, Infinity, off-lattice и out-of-range отклоняются.

Lean доказывает устойчивость обеих ветвей **условно на `ApproxContract`**: `approx_contract_relu_positive_branch` и `approx_contract_relu_nonpositive_branch` порождают поузловые правила для всех reachable ReLU. Новый `numeric_threshold.lean` включает условные positive/nonpositive obligations и конструктивное доказательство неоднозначности у нуля: одно observed значение совместимо с идеальными значениями по разные стороны порога. `threshold_certificates.txt` помечает наблюдения как синтетические, а не GPU-измерения. `task numeric-stage8` успешно проверил GPU baseline/fused и все 13 Lean-файлов; `task verify-numeric-stage8` повторно подтвердил 48 SHA-256 артефактов без CUDA-запуска. Подтверждённый manifest SHA-256: `60355e39deceac555771e31a3c79924d3e361f14060dea7d958608a41261229f`. Эти результаты относятся к условной dyadic/RNE модели и записанным GPU-трассам, связь с машинным FP32/LLVM/PTX/GPU по-прежнему остаётся TCB. В данном прогоне повторный результат полного `task test` не представлен.

**FP32 contraction slice локально подтверждён пользователем после formatting-коммита `83ecaec`.** GPU backend теперь задаёт `add.rn.f32` и `mul.rn.f32` через side-effecting LLVM inline PTX для Add/Mul/BiasAdd и каждого шага MatMul. После `llc` `audit_fp32_ptx` проверяет, что PTX содержит именно ожидаемое количество округлённых FP32 add/mul, отклоняет `fma`, `mad`, `ftz`, неявные `add.f32`/`mul.f32` и неподдержанные операции. Ошибка аудита блокирует CUDA execution. Пример Stage 8 дополнительно сравнивает baseline/fused `Mul → BiasAdd → ReLU` для FMA-чувствительных f32-битов: раздельные RNE дают +0, contracted multiply-add даёт ненулевой положительный результат. Сохранённые LLVM/PTX пары перепроверяются офлайн, включая audit report, и входят в SHA-256 manifest. Успешны 71/71 Rust-тестов, оба FMA-чувствительных GPU-профиля, статический аудит 10 пар PTX/LLVM, 13 Lean-файлов, 55 evidence-файлов и offline verify. Подтверждённый SHA-256 manifest: `4b511265597e8f66f7de3ef2bd7f245e90a5c05c01dbdd9f655269dd57cae652`. **Это статическая проверка выданных PTX-инструкций и конкретный GPU-oracle, не математическое доказательство компилятора, порядка команд после JIT, SASS, аппаратной IEEE-754 или всего FP32-диапазона.** FTZ/subnormal/special values и полноценная FMA-семантика остаются вне модели.

**Subnormal/FTZ boundary slice подтверждён локальным прогоном после formatting-коммита `d6011bb`.** Добавлен ограниченный независимый `subnormal_mul_rne_bits` для конечных FP32, чья точная целочисленная мантиссная продукция после RNE попадает в интервал subnormal/малейшего normal; `subnormal_add_exact_bits` поддерживает лишь точное сложение на lattice `2^-149` в этом же интервале. `subnormal_relu_bits` задаёт узкую битовую конвенцию ReLU. Unsupported FP32 cases возвращают `None`, без расширения старого `RneFp32Plan` за пределы его normal-domain модели. `numeric_subnormal.lean` содержит доказанные half-tick ties-to-even и переход subnormal → min-normal для фиксированных чисел; это не полная FP32 семантика. CUDA Graph baseline/fused probe проверяет 8 adversarial lanes, различающих FTZ/DAZ, ties и сохранение subnormal-битов. Пользователь подтвердил 73/73 Rust-теста, Cargo check/Clippy, baseline/fused по 8 FTZ-чувствительных значений, 14 Lean-файлов, 63 SHA-256 артефакта и offline verify. Подтверждённый manifest SHA-256: `4af0ead91a2c210f1dff6c19399beba42e2e189eda8fe3305f06e5e1ca65dce1`. Драйвер/JIT/SASS остаются в TCB.

**Exceptional-FP32 boundary slice локально подтверждён пользователем после Lean fix `c6b8d40`.** `special_mul_bias_relu_boundary` задаёт независимый bit-oracle для **строго перечисленных** NaN/Infinity/signed-zero, `maxFinite × 2` overflow и операций с единицей. Неподдерживаемая арифметика и signaling NaN отвергаются (`None`), quiet NaN проверяется только по классу без предположения о payload/sign, остальные допустимые результаты сравниваются по 32 битам. В Stage 8 добавлены `baseline_special`/`fused_special` CUDA Graph профили (13 входных строк), сохраняемые GPU-выходы и независимая офлайн-перепроверка каждого наблюдения. `numeric_special.lean` доказывает только классификацию конкретных binary32 слов как zero/infinity/quiet NaN/finite, а не арифметику IEEE-754. Успешны все **14 GPU-профилей**, в том числе `baseline_special`/`fused_special` по 13 специальных значений; Lean принял **15 файлов**, `task verify-numeric-stage8` повторно проверил все **72 SHA-256 evidence-файла**. Подтверждённый manifest: `431dc70b2eadb5147b94c82b0567dd91f3680eb9c80cdda01bed1d5f9d7fe727`. В данном журнале нет повторного полного `task test` для финального HEAD; 75/75 Rust-тестов пока не утверждаются. Полная IEEE-754-семантика, payload-policy, PTX/JIT/SASS и машинное FP32 соответствие остаются TCB/незакрыты.

Первый module-boundary пример использует `ResidualFfn(1, 2, 2)` при source `q=10`, `|ticks| ≤ 16`. Две MatMul, bias, ReLU и residual add выводят `q_out=30`, ideal bound `34 095 104` ticks, `ε=170` ticks и actual bound `34 095 274` ticks. `RneRangePlan::module_contracts()` экспортирует эту сводку на границе вызова `residual_ffn`; `range_contract.txt` и `numeric_range.lean` включены в Stage 8 evidence bundle.

**Stage 8 Numeric Translation Validation: крупный вертикальный срез локально проверен на `c63cd1f` (кроме Clippy).** `TranslationPlan` формируется из Semantic IR, guarded fusion и memory contracts. `TranslationPlan::validate` до CUDA JIT проверяет по каждому kernel LLVM-функции и PTX entry points, конкретные формы, источники, адресные шаблоны, RNE add/mul, порядок MatMul reduction и ReLU guard. Ошибка ведёт к fail-closed, а не к запуску подозрительного GPU-кода. Сохранены `translation.plan` и `translation.report` для всех 14 GPU-профилей, а общий `translation_summary.txt` повторно проверяется offline вместе с LLVM/PTX и SHA-256. Пользователь подтвердил `task fmt-check` (форматирующий коммит `c63cd1f`), `task check`, **80/80 Rust-тестов**, **14/14 CUDA-профилей**, Lean, **101 SHA-256 evidence-файл** и успешный `task verify-numeric-stage8`. Закреплённый manifest: `7f4d473333b7360a5fbc75496b7bdbabc441c007fd862d29190719ddf03afa54`. На `c63cd1f` `task clippy -D warnings` обнаружил один `collapsible_if`, исправленный в следующем коммите; повторный результат Clippy пока не подтверждён. Это **структурная проверка известного LLVM/PTX подмножества**, а не доказательство эквивалентности FP32-инструкций, CUDA JIT/SASS, гонок или физической памяти. Status: Lean=ideal Int/dyadic conditional, structural translation=checked-by-validator, GPU=numerical probe, JIT/SASS=trusted.

В предыдущем MatMul/BiasAdd bundle `task numeric-stage8` проверял двенадцать Lean-артефактов: `numeric.lean`, `numeric_matmul.lean`, `numeric_rne.lean`, `numeric_rne_matmul.lean`, `numeric_range.lean`, `numeric_architecture.lean`, `numeric_architecture_dag.lean`, `numeric_nested.lean`, `numeric_nested_deep.lean`, `architecture_property.lean`, `architecture_property_nested.lean` и новый `architecture_property_matmul_bias.lean`. MatMul/BiasAdd global-property bundle локально подтверждён: 46 evidence-файлов и 12 Lean-артефактов; `numeric-stage8` и offline verify прошли на `eaf701e`. Текущий подтверждённый threshold bundle дополнительно включает `numeric_threshold.lean` (13-й Lean-файл) и `threshold_certificates.txt`; `task verify-numeric-stage8` повторно проверяет Lean и SHA-256 без GPU.

```sh
task numeric-stage8
task verify-numeric-stage8
```

**Граница гарантии:** whole-module slice локально подтверждён пользователем после formatting-коммита `489f5eb`: check/test/clippy, Stage 5 certificate, `task numeric-stage8` и offline verify прошли; 57 Rust-тестов были зелёными, а Stage 8 manifest стал `0fdf33efe2bc38014016d5777b35590753622cd3d32b51c74a0405b24bb60896`. `module_<n>_global_contract` действительно проверен Lean для полного `ResidualFfn` и fail-closed при undeclared capture.

Architecture-composition slice локально подтверждён пользователем после formatting-коммита `a942802`: прошли `cargo check`, 58 Rust-тестов (31 unit + 10 Stage 7 + 17 Stage 8), Clippy, полный `task numeric-stage8` и offline `task verify-numeric-stage8`. `architecture_global_contract` проверен Lean для цепочки двух `ModuleInvocation`: доказанный `module_0_global_contract` поставляет output `ApproxContract`, которые становятся входными доказательствами `module_1_global_contract`. Для двух последовательных `ResidualFfn(1,2,2)` при q10/source bound 16 получены q50, ideal bound `53 928 163 016 704` ticks и `ε=184 963 072` ticks. `numeric_architecture.lean` и `architecture_contract.txt` входят в 34-файловый Stage 8 evidence bundle; подтверждённый manifest: `134c321f0af061ab3fe5165ad3f5beec5ebf8e6e9f0bd79f8ca38b389f46d9d6`. DAG/nested slice локально подтверждён пользователем после formatting-коммита `8a6fe41`: прошли 60 Rust-тестов, Clippy, полный Stage 8 Lean/GPU/evidence прогон и offline verify. Fork/fan-in дал q30 / `68 190 208` / `ε=348`, direct nested `NestedResidualPair` дал q50 / `53 928 163 016 704` / `ε=184 963 072`; 38-файловый manifest: `916e64b60599d1bc1aa95d315d57a96fb733a1708b58307973dbc6bb285fa384`. Deep-nesting slice локально подтверждён пользователем после formatting-коммита `41e6c37`: прошли 61 Rust-тест, Clippy, полный Stage 8 Lean/GPU/evidence прогон и offline verify; depth=3 theorem reuse принят Lean. Подтверждённый 40-файловый manifest: `c244df753c4a3cbffa3e00830c33d057c94b0ed9285539ab6f9591c5c2809121`. Global-property slice локально подтверждён пользователем после formatting-коммита `c3927a4`: 63 Rust-теста, Clippy, Lean/GPU и offline verify прошли; `architecture_global_nonnegative` доказана для top-level Add/Mul DAG. Подтверждённый 42-файловый manifest: `e41a973676c30f2614a30abf99db767bdf1b9aa456a38d33f5e03e031b8077ae`. Nested-global-property slice локально подтверждён пользователем после formatting-коммита `6112ed8` и Clippy fix `44693a6`: 65 Rust-тестов, Clippy, `architecture_property_nested.lean` и offline verify прошли. Подтверждённый 44-файловый manifest: `5a5a049e7252daf83dca11603b07389dd748bad21ae066c68954bd8133d76880`. MatMul/BiasAdd ideal-Int slice подтверждён пользователем после formatting-коммита `ae24512` и Lean fix `eaf701e`: 67 Rust-тестов, fmt/check/clippy, GPU и все 12 Lean-файлов, включая `architecture_property_matmul_bias.lean`, и offline SHA-256 verify прошли. `architecture_global_nonnegative` композиционно доказывает nonnegativity MatMul left-to-right reduction и row-wise BiasAdd broadcast при явно заданных знаковых input/weight/bias assumptions. Подтверждённый 46-файловый manifest SHA-256: `ad5f0aa1927ce9765cf9980e49f072b80e82e1231aaaa82820c0b9100f02d4ff`. Недостающая bias-hypothesis fail-closed; FP32/GPU nonnegativity не доказана. Также остаются FMA/contract behavior, FTZ/subnormal, NaN/Infinity, signed zero, overflow за текущим fail-closed профилем, условные ветви и translation validation. Stage 8 не завершён.

## Git-процесс

Изменения интегрируются в `main` **только fast-forward**, без pull request, merge-коммитов, squash и force push. Полные правила для агентов — в корневом [AGENTS.md](AGENTS.md).

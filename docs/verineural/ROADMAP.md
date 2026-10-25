# VeriNeural: 10 крупных этапов реализации

**Назначение:** план реализации [целевой спецификации](SPEC.md). В отличие от SPEC.md, здесь этапы перечислены намеренно. Целевая спецификация охватывает **архитектурно-нейтральный конструктор** для собственных нейросетевых алгоритмов, в том числе альтернативных Transformer моделей языка; не стандартную LLM, её готовые блоки или инфраструктуру.

**Последовательность:** ядро конструктора на Rust и Semantic Neural IR → Lean/GPU → связанный отчёт → расширяемые операции и иерархические блоки → управляемая GPU-компиляция → FP32/композиционные доказательства → собственная крупная архитектура → certified build. Сначала реальные end-to-end результаты, затем глубина гарантий.

**Обязательные принципы:** KISS, DRY, Pragmatic Engineering, Risk-based Verification, Modular Verification, Architecture Neutrality и Hierarchical Composition. Предпочитать проверяемое узкое подмножество широкой декларативной поддержке.

**Вне обязательной области:** готовые Transformer, Attention/MHA/GQA, KV Cache, специализированные LLM-блоки, чат/API/serving, распределённая инфраструктура, деплой. Минимальный CUDA runtime для исполнения и проверки результатов остаётся необходимым. Dense/FFN на раннем этапе — **пример композиции общих операций**, а не переход к разработке фиксированной LLM.

## 1. Инициализация workspace и проектного контракта — РЕАЛИЗОВАН (локальная проверка не подтверждена)

**Работы:** Cargo workspace, импортируемый crate `verineural`, корневые `flake.nix` и `Taskfile.yml`, правила `AGENTS.md`, документация полной цели и план. Без реализации GPU, Lean или архитектуры.

**Готово, когда:** новый crate корректно подключён к workspace, пути единообразны, спецификация и правила сохранены. Локальную сборку выполняет пользователь; факт сборки не заявляется, пока пользователь её не провёл.

## 2. Rust Neural API и Semantic Neural IR — РЕАЛИЗОВАН (локальные проверки пройдены)

**Текущая реализация:** динамические проверяемые формы, владеющий данными `Tensor<T, Shape>`, операции Add/Mul/MatMul/ReLU/Sqrt/Reshape, символический DAG через Rust builder, структурные и доменные контракты `NeuralModule`, явная FP32 policy и метка отсутствующего GPU lowering. Входные домены пока задаются как предположения, которые необходимо проверять на конкретных данных. GPU, Lean и формальные численные доказательства принадлежат следующим этапам. Поддерживается ограниченный Rust DSL, не анализ произвольного Rust-кода.

**Работы:** `Tensor<T, Shape>`, `NeuralModule`, `NeuralGraph`, `MathContract`, `FP32Policy`; API чистых операций, проверка форм, граф и зависимости; три семантических слоя — Ideal, FP32, GPU. Поддерживаемое подмножество Rust/операций объявлено явно. Не плодить crates ради названий из спецификации.

**Готово, когда:** Rust-функция/модуль формирует валидный Semantic Neural IR с математическим контрактом и ошибками неверных форм/доменов, а архитектурные компоненты могут композиционно выстраиваться в граф.

## 3. Lean 4 Proof Engine: операции, контракты и композиция — БАЗОВЫЙ СРЕЗ ЛОКАЛЬНО ПРОВЕРЕН (включая форматирование)

**Текущая реализация:** Rust IR → генератор Lean 4 → запуск проверяющего Lean kernel, явные теоремы о формах, поэлементной идеальной целочисленной семантике и неотрицательности ReLU/Reshape, в том числе при композиции модулей. Исходные предположения о неотрицательности остаются гипотезами теорем; отсутствие Lean или любая ошибка/предупреждение приводят к отказу. На момент локальной проверки этапа 3 proof backend охватывал только Add/Mul/Relu/Reshape над Int; MatMul/BiasAdd позднее добавлены в коде этапа 6. Sqrt, FP32/IEEE-754, произвольная пользовательская математика и GPU-исполнение по-прежнему не доказаны. Этот результат не удостоверяет численное поведение Rust f32 или аппаратного GPU.

**Работы:** минимальные формальные определения тензоров, индексов, форм, элементарных операций и модулей; генерация Lean-моделей и proof obligations из Semantic Neural IR; библиотека проверенных лемм; связь `requires`/`ensures` между модулями. Запрет `sorry`/необоснованных аксиом.

**Готово, когда:** одна Rust-функция порождает теоремы и проверяемые Lean-доказательства вычислительных и композиционных инвариантов, а недоказанный контракт отклоняется.

## 4. GPU Compiler и CUDA Runtime: исполняемый вертикальный срез — ЛОКАЛЬНО ПОДТВЕРЖДЁН

**Текущая реализация:** Semantic Neural IR → LLVM NVPTX IR → `llc` → PTX → CUDA Driver API (`libcuda.so.1`). Поэлементные Add, Mul и ReLU выполняются отдельными kernels; Reshape переиспользует GPU-буфер. Реализованы проверки форм и входных доменов, выделение и освобождение памяти, синхронизация и сравнение с независимым CPU reference. В проверенном срезе этапа 4 MatMul/Sqrt отклонялись (MatMul и BiasAdd позднее добавлены в код этапа 6); код GPU и FP32 не доказаны Lean. Пользователь подтвердил GPU-прогон (6/6 значений FP32 совпали по битам), 17/17 Rust-тестов, Clippy без ошибок и Lean 25 обязательств.

**Работы:** GPU IR с памятью и потоками, LLVM IR → LLVM NVPTX → PTX, загрузка/запуск через CUDA Driver API, буферы, синхронизация, элементарные tensor kernels, связь с исходным графом.

**Готово, когда:** тот же модуль компилируется из Rust в PTX и локально исполняется на GPU с независимой численной проверкой. Этот этап ещё не означает формально доказанное аппаратное исполнение.

## 5. Сведение доказательств и исполнения, первичный сертификат — ЛОКАЛЬНО ПОДТВЕРЖДЁН

**Текущая реализация:** один Rust IR используется Lean и LLVM NVPTX/CUDA backend'ами; после проверки Lean и совпадения FP32 выходов по битам сохраняется JSON отчёт, исходный IR, Lean, LLVM IR, PTX, конкретные входы и результаты, SHA-256 манифест и внешний хеш для закрепления. Повторная команда проверяет целостность относительно манифеста и повторно запускает Lean. IEEE-754, PTX и машинное исполнение не доказаны; TCB и статусы обозначены явно. При ошибке отчёт удаляется. Пользователь подтвердил успешный локальный проход Stage 5 (Lean и CUDA, формирование и повторная проверка отчёта). Это не устраняет перечисленные границы доверия.

**Работы:** сертификаты по слоям mathematical, numerical, composition и compilation; привязка IR, Lean-теорем и GPU-артефактов хешами; отчёт о TCB, явная область входов, принцип fail-closed. Без недоказанных гарантий машинного кода.

**Готово, когда:** один пример проходит Rust → IR → Lean + PTX → запуск → проверяемый отчёт, различающий `proven`, `validated`, `trusted` и `unsupported`.

## 6. Расширяемые вычислительные примитивы и иерархический конструктор — БАЗОВЫЙ СРЕЗ ЛОКАЛЬНО ПОДТВЕРЖДЁН

**Текущая реализация:** Rust IR поддерживает row-major MatMul и BiasAdd ([M,N] + [N]) с CPU reference, Lean-моделью над идеальными `Int` и простыми CUDA kernels. Добавлены `Dense`, `ResidualAdd`, `ResidualFfn` как **демонстрации** `NeuralModule` и проверок контрактов, а также пример Rust → IR → Lean + PTX → NVIDIA GPU → отчёт. Это не заготовка обязательного Transformer/FFN-стека. Lean пока проверяет модель/формы и отдельные свойства, **не** корректность FP32/GPU lowering и **не** произвольные пользовательские глобальные теоремы.

**Работы:** универсальные строительные средства (тензорная арифметика, линейная алгебра, reshape/indexing/gather/scatter, редукции, сравнения, условные вычисления и пользовательская математика по мере потребности); интерфейс добавления операций с семантикой и контрактами; параметризованные повторно используемые блоки, вложенные подграфы, экспортированные порты и проверяемая связь между ними. Для циклов/состояния, если они нужны пользователю, определить явную семантику и безопасный поддерживаемый фрагмент. Каждый **новый поддерживаемый** оператор должен иметь GPU-lowering и Lean-модель либо честный статус `unsupported`.

**Новый P0 structural/indexing slice, ожидающий локальной проверки:** IR различает `DType::F32` и `DType::Index`; добавлены `permute`, `reverse_axis`, бинарный `concat` с variadic public facade, детерминированный `argmax` с policy `First` и `take`. Shape/dtype/domain validation выполняется до изменения графа, host reference задаёт row-major semantics и bounds checks, canonical IR v3 и DOT сохраняют dtype/axis/permutation/tie policy. Lean покрывает identity/rank-2 transpose permutation, reverse, concat, first-tie argmax и take; неподдерживаемые permutations и весь GPU/numeric lowering нового среза fail-closed. Это закрывает API-дыру, из-за которой routing consumer вынужден был представлять индекс как host integer и подсовывать производный rotated tensor как внешний параметр, но не объявляет GPU lowering уже выполненным.

**Готово, когда:** разработчик описывает нетиповой математический оператор/композицию на Rust, собирает из него несколько переиспользуемых блоков и крупный модуль; всё компилируется из одного IR в исполняемый PTX и Lean-модели, локальные нетривиальные контрактные цели проверяются, ошибки форм/предпосылок отклоняются. Не требуется набор Attention, MoE или KV Cache.

## 7. Управляемый NVIDIA GPU Backend и доказательства преобразований — GRAPH REUSE / CHAIN FUSION / МАСШТАБНЫЙ BENCH ЛОКАЛЬНО ПОДТВЕРЖДЕНЫ; ОПТИМИЗАЦИЯ RUNTIME В КОДЕ

**Подтверждённый срез:** CUDA Graph-only на явном non-blocking stream; подготовленный `CUgraphExec`, многократный replay и обновление входных тензоров без изменения GPU-графа; конфигурируемая сетка 64/128/256/512 threads/block и experimental chain fusion `Add/Mul → BiasAdd → Relu`, сокращающий три kernels до одного при эксклюзивных зависимостях. Пользователь локально подтвердил fmt/check/test/clippy, Lean, CUDA, SHA-256, побитовую валидацию CPU/GPU, benchmark на 1K/16K/256K/1M и отказ неверных входных данных. На 1M элементов без профилирования медиана CUDA-event replay составила 28.884 мкс baseline и 14.629 мкс fused (≈1.97× быстрее); на малых массивах эффект невелик и результаты шумные. При Nsight Systems `--cuda-graph-trace=node` график replay существенно меняется из-за overhead профилировщика. Отдельные kernels агрегированы под именами `vn_node_3/4/5`, поэтому суммарная сводка без разбивки по размеру не доказывает конкретную скорость fused kernel на каждом размере.

**Новый срез, ожидающий локальной проверки:** каждый `CudaSession` теперь удерживает один reference на device primary CUDA context (`cuDevicePrimaryCtxRetain/Release`) вместо создания и разрушения private `CUcontext` для каждого `GpuExecution`. NVIDIA primary context разделяется между сессиями на том же GPU, а stream/module/graph/allocations остаются уникальными. Система по-прежнему использует `cuCtxPushCurrent_v2`/`cuCtxPopCurrent_v2` и не передаёт операции в legacy/default stream. Отдельный пул pinned staging буферов рециклирует host allocations **только после stream synchronization**, сохраняя их живыми при async H2D/D2H копировании. Дополнительные результаты `task gpu-stage7-bench` сравнивают затраты `GpuEngine::prepare` (включая llc/JIT/allocations/graph capture), CUDA-event replay и wall-clock полного обновления данных H2D→replay→D2H. Ожидаемое преимущество этих оптимизаций — снижение накладных расходов подготовки и циклических трансферов, **не** ускорение отдельных вычислительных kernels.

**Обязательное ограничение:** CUDA Graph-only и явный `CU_STREAM_NON_BLOCKING`, никаких CUDA Legacy/NULL/default streams, kernel launch вне Graph или fallback. Оптимизация fusion остается экспериментальной; Lean-леммы фиксируют только идеальную `Int`-математику. Сравнение `f32` по битам не доказывает IEEE-754 и PTX; GPU driver/JIT/оборудование и компиляция LLVM остаются доверенной границей.

**Новый срез проверки индексов и буферов (локальная проверка ожидается):** Rust компилятор порождает для каждого выбранного kernel memory contract с реальными входными диапазонами и выходной длиной; валидирует соответствие тензорным формам, topology DAG, отсутствие недопустимого переполнения CUDA grid. Перед графовым capture runtime дополнительно сверяет все read/write spans с выделенными device allocations, запрещает in-place output и общие writable output buffers. Lean 4 универсально доказывает границы pointwise, modulo-bias и двух row-major MatMul-адресов при активном output lane и корректном j. Из реально скомпилированных baseline/fused graph порождаются *отдельные* `memory_baseline.lean` и `memory_fused.lean` с конкретными параметрами; оба проверяются Lean, входят в Stage 7 SHA-256 manifest и повторно проверяются без GPU. Это **арифметические доказательства индексации при предпосылках**, а не доказательство корректности LLVM/PTX адресов, реальной памяти CUDA, race freedom или IEEE-754.\n\n**Оставшаяся работа этапа 7:** доказательства/translation validation для IR → GPU IR → LLVM IR → PTX, инварианты адресации/памяти/гонок и синхронизации, допустимость каждого преобразования в заявленной семантике; расширение универсального scheduling/tiling/fusion там, где полезно и подтверждено измерениями. Stage 8 по формальной FP32-композиции не объявлен выполненным и не подменяет незакрытые обязательства Stage 7.

**Готово, когда:** один нестандартный составной блок работает только через CUDA Graph без CUDA Legacy Stream, имеет несколько контролируемых вариантов GPU lowering, измеряемое исполнение и проверяемые свидетельства корректности **каждого разрешённого** преобразования в поддерживаемой семантике. Непроверенные преобразования отклоняются в certified-режиме; GPU драйвер/JIT/аппаратура честно остаются в TCB, где их поведение не доказано.

## 8. FP32 Verification Engine и численно осведомлённая композиция — ARCHITECTURE COMPOSITION ПОДТВЕРЖДЕНА

**Подтверждённый целочисленный срез:** на коммите `4414613` пользователь локально подтвердил 49/49 Rust-тестов, fmt/check/clippy, Lean-доказательства диапазонов, baseline/fused CUDA Graph для аддитивного графа (6/6 + 6/6 совпадений), baseline/fused MatMul-графа (4/4 + 4/4), SHA-256 и повторную Lean-проверку без GPU. Всего 20/20 значений побитово совпали с независимым `i64`-эталоном. `ExactFp32Plan` покрывает `Input/Add/Mul/MatMul/BiasAdd/Relu/Reshape`; `compile_with_input_bound` доказывает консервативные границы произведений и всех left-to-right частичных сумм MatMul внутри `[-2^24,2^24]`. Подтверждённые итоговые bounds: `3 145 728` для аддитивной цепочки при входах `±2^20` и `1 544` для `MatMul → Mul → BiasAdd → Relu` при входах `±8`. Манифест подтверждённого прогона: `66978a69a62d8ca0e642c25b30daa067024d3ec74546e0ee5e40af67777c2040`. Это остаётся доказательством идеальной `Int`-модели плюс численной проверкой конкретного GPU-запуска, а не доказательством IEEE-754/LLVM/PTX.

**Подтверждённый дробный RNE/ε срез:** additive профиль был подтверждён на `2930a4f`, а fractional `Mul/MatMul` после formatting-коммита `6806fe2` также прошёл `cargo check`, 54 Rust-теста, Clippy, `task numeric-stage8` и offline `task verify-numeric-stage8`. Baseline/fused fractional MatMul совпал с separate-mul/add integer RNE oracle; q20→q40 trace имеет observed error 2 ticks и conservative `ε=41 943 040` ticks. Актуальный подтверждённый manifest: `87f3d550c94aee99e5733626eb888f8c449a319af7f762a1ac84c8d450396746`. LLVM/PTX/JIT/GPU соответствие формальной модели всё ещё остаётся TCB.

**Подтверждённый range/module slice:** на `27db27b` прошли fmt, 56 Rust-тестов, `task numeric-stage8` и offline `task verify-numeric-stage8`; 32-файловый manifest SHA-256: `725e652961cb14a65b2b0d22578fadfad18aff71fb705356c147ef426f44588e`. `RneRangePlan` принимает source dyadic scale и interval bound, затем консервативно выводит per-node `ideal_bound`, `epsilon`, `actual_bound=ideal_bound+epsilon` и output scale для `Input/Add/Mul/MatMul/BiasAdd/Relu/Reshape`. Worst-case RNE step вычисляется из максимального pre-round magnitude. Lean `ApproxContract`, `binary32_rne_shift_le_of_bound`, `binary32_rne_within_maxStep`, `approx_contract_scale/add_rne/mul_rne/relu` являются reusable theorem rules с произвольными Int-значениями. Генерируемый `numeric_range.lean` инстанцирует их bounds каждого reachable node. `RneRangePlan::module_contracts()` экспортирует boundary summary; контрольный `ResidualFfn(1,2,2)` при q=10 и source bound 16 получает q_out=30, ideal bound 34 095 104 ticks и `ε=170` ticks. `numeric_range.lean` и `range_contract.txt` входят в Stage 8 manifest.

**Подтверждённый whole-module slice:** после formatting-коммита `489f5eb` пользователь подтвердил check/test/clippy, Stage 5 cert/verify, 57 Rust-тестов, `task numeric-stage8` и offline verify. Lean принял `module_0_global_contract` полного `ResidualFfn(1,2,2)`; undeclared graph capture regression также прошёл. Stage 8 manifest этого прогона: `0fdf33efe2bc38014016d5777b35590753622cd3d32b51c74a0405b24bb60896`.

**Подтверждённый architecture-composition slice:** после formatting-коммита `a942802` пользователь подтвердил `cargo check`, 58 Rust-тестов (31 unit + 10 Stage 7 + 17 Stage 8), Clippy, `task numeric-stage8` и offline `task verify-numeric-stage8`. Lean принял `architecture_global_contract`, который для совместимой линейной цепочки `ModuleInvocation` вызывает уже доказанные whole-module теоремы и передаёт их output `ApproxContract` дальше по архитектуре. Контрольный граф из двух последовательных `ResidualFfn(1,2,2)` при q10/source bound 16 дал q50, ideal bound 53 928 163 016 704 ticks и `ε=184 963 072` ticks. `numeric_architecture.lean` и `architecture_contract.txt` входят в 34-файловый evidence bundle; manifest SHA-256: `134c321f0af061ab3fe5165ad3f5beec5ebf8e6e9f0bd79f8ca38b389f46d9d6`.

**Подтверждённый DAG/nested slice:** после formatting-коммита `8a6fe41` пользователь подтвердил `cargo check`, 60 Rust-тестов (31 unit + 10 Stage 7 + 19 Stage 8), Clippy, полный `task numeric-stage8` и offline `task verify-numeric-stage8`. Lean принял fork/fan-in `architecture_global_contract` и nested whole-module reuse. Контрольный граф `ResidualFfn_A(X) + ResidualFfn_B(X)` через `ResidualAdd` дал q30, ideal bound 68 190 208 ticks и `ε=348` ticks; `NestedResidualPair` сохранил q50 / 53 928 163 016 704 / ε=184 963 072. Подтверждённый 38-файловый manifest SHA-256: `916e64b60599d1bc1aa95d315d57a96fb733a1708b58307973dbc6bb285fa384`.

**Подтверждённый deep-nesting slice:** после formatting-коммита `41e6c37` пользователь подтвердил `cargo check`, 61 Rust-тест (31 unit + 10 Stage 7 + 20 Stage 8), Clippy, `task numeric-stage8` и offline verify. Lean принял `numeric_nested_deep.lean`; depth=3 `DeepNestedWrapper → NestedResidualPair → ResidualFfn` сохранил q50 / 53 928 163 016 704 / ε=184 963 072. Подтверждённый 40-файловый manifest SHA-256: `c244df753c4a3cbffa3e00830c33d057c94b0ed9285539ab6f9591c5c2809121`.

**Подтверждённый global-property slice:** после formatting-коммита `c3927a4` пользователь подтвердил 63 Rust-теста (33 unit + 10 Stage 7 + 20 Stage 8), fmt/check/clippy, `task numeric-stage8` и offline verify. Lean принял `architecture_global_nonnegative` для Add/Mul fork/fan-in DAG с тремя явными nonnegative source assumptions; отсутствие обязательной гипотезы отклоняется. Подтверждённый 42-файловый manifest SHA-256: `e41a973676c30f2614a30abf99db767bdf1b9aa456a38d33f5e03e031b8077ae`.

**Подтверждённый nested-global-property slice:** на formatting-коммите `6112ed8` пользователь подтвердил 65 Rust-тестов (35 unit + 10 Stage 7 + 20 Stage 8) и полный Stage 8 Lean/GPU/evidence прогон, manifest `5a5a049e7252daf83dca11603b07389dd748bad21ae066c68954bd8133d76880`; после исправления `44693a6` fmt/check/test/Clippy/offline verify снова прошли. Lean доказал depth=3 `architecture_global_nonnegative` через direct child modules без скрытых dependencies.

**Подтверждённый MatMul/BiasAdd global-property slice:** после formatting-коммита `ae24512` и Lean zero-base fix `eaf701e` пользователь подтвердил 67 Rust-тестов (37 unit + 10 Stage 7 + 20 Stage 8), fmt/check/clippy, `task numeric-stage8` с GPU и 12 Lean-файлами, а также offline `task verify-numeric-stage8`. Reusable ideal-Int леммы `flatGet_nonnegative`, `matmul_nonnegative` (каждая left-to-right частичная сумма) и `biasAdd_nonnegative` (broadcast через `i % n`) приняты Lean. Модульный `architecture_global_nonnegative` доказан для `MatMul(2×3,3×2) → BiasAdd(2×2,2)`; недостающая bias-hypothesis отклоняется. Подтверждённый 46-файловый SHA-256 manifest: `ad5f0aa1927ce9765cf9980e49f072b80e82e1231aaaa82820c0b9100f02d4ff`. Обязательства машинного LLVM/PTX/GPU correspondence остаются TCB.

**Threshold-margin slice (локально подтверждён после Lean fix `2a6531e`):** Rust `RneRangePlan::classify_relu_input` декодирует синтетическое/наблюдаемое pre-ReLU binary32 на dyadic lattice узла и выводит интервал идеального аргумента `[actual−ε, actual+ε]` только при предпосылке `ApproxContract`. При `actual−ε>0` ветвь `Positive`; при `actual+ε≤0` ветвь `NonPositive`; иначе `Indeterminate`. Прочие узлы, недостижимые ReLU, off-lattice, subnormal, -0, NaN/Inf и выход за actual bound fail-closed.

Lean `approx_contract_relu_positive_branch` / `approx_contract_relu_nonpositive_branch` дают общие условные доказательства; `RneRangePlan::lean_source()` порождает специализированные правила для каждого ReLU. `numeric_threshold.lean` дополнительно предъявляет два допустимых идеальных аргумента разных знаков для одной неопределённой точки. `threshold_certificates.txt` явно указывает на синтетические pre-activation observations, доказательные гипотезы и доверенный LLVM/PTX/GPU слой. Пользователь подтвердил полный `task numeric-stage8` с восемью успешными baseline/fused GPU-профилями и Lean, затем offline `task verify-numeric-stage8`: 13 Lean-файлов и 48 файлов SHA-256 bundle приняты. Внешне закреплённый SHA-256 manifest: `60355e39deceac555771e31a3c79924d3e361f14060dea7d958608a41261229f`. В этом логе нет повторного полного `task test`; machine FP32, LLVM/PTX/JIT/GPU correspondence не доказаны.

**FP32 FMA/contraction audit slice (локально подтверждён после `83ecaec`):** для поддерживаемых CUDA Graph kernels Add/Mul/MatMul/BiasAdd и guarded fused chains LLVM генерирует side-effecting `add.rn.f32`/`mul.rn.f32` inline PTX, а отдельный аудитор проверяет итоговый PTX после `llc`. Отказ происходит при FP32 `fma/mad`, FTZ, неявном округлении, неразрешённых FP32 arithmetic opcode либо несовпадении ожидаемого количества инструкций. Поскольку исходное LLVM IR и PTX являются входами текстового аудитора, он не доказывает их полную семантическую эквивалентность и не покрывает SASS/JIT. Stage 8 запускает baseline/fused FMA-discriminating `Mul → BiasAdd → ReLU` fixture с независимым сравнением `f32::mul_add` и отдельного округления. Обе пары LLVM/PTX, входные/выходные биты и статический отчет добавлены в SHA-256 evidence; offline verify перепроверяет сохранённый PTX. Пользователь подтвердил Cargo check, 71 Rust-тест, Clippy, FMA-специфичные baseline/fused GPU fixtures, 13 Lean-файлов и offline SHA-256 verify 55 артефактов. Подтверждённый manifest: `4b511265597e8f66f7de3ef2bd7f245e90a5c05c01dbdd9f655269dd57cae652`.

**Gradual-underflow/FTZ subnormal boundary slice (локально подтверждён после `d6011bb`):** отдельный `subnormal_mul_rne_bits` читает IEEE754 significand/exponent из 32-битного представления без host float arithmetic, вычисляет точное tiny произведение в `u128` и округляет к `2^-149` lattice методом RNE, отказываясь при выходе из поддержанного subnormal/min-normal диапазона. `subnormal_add_exact_bits` допускает только точные операции в том же диапазоне и `subnormal_relu_bits` отсекает отрицательный аргумент в +0. Не расширяет `RneFp32Plan` и `ApproxContract`: они пока не доказывают IEEE underflow. Baseline/fused CUDA Graph проверяют 8 битовых FTZ/DAZ/tie-even/midpoint/sign/min-normal случаев и pin input/output/evidence bytes. Отдельная `numeric_subnormal.lean` доказывает частные half-tick свойства и несовпадение gradual underflow с FTZ для целочисленной модели. Подтверждены 73/73 Rust-теста, check/Clippy, baseline/fused 8/8 gradual-underflow значений, Lean 14 файлов и SHA-256 offline verify 63 артефактов. Подтверждённый manifest: `4af0ead91a2c210f1dff6c19399beba42e2e189eda8fe3305f06e5e1ca65dce1`. Общая FP32-семантика, JIT/SASS и полнота PTX аудита остаются открытыми.

**Exceptional-FP32 finite/special-value slice (локально подтверждён после Lean fix `c6b8d40`):** добавлен ограниченный независимый binary32 bit-oracle `special_mul_bias_relu_boundary` для конкретных NaN/Infinity/signed-zero, min-subnormal identity и maxFinite×2 overflow; неподдерживаемые случаи и signaling NaN явно отвергаются. При qNaN проверяется только quiet-NaN-класс, не payload/sign; для остальных выходов требуется точное совпадение битов. Два CUDA Graph профиля `baseline_special/fused_special` проверяют 13 строк на одном DAG `Mul→BiasAdd→ReLU`; snapshot обеих ветвей, входы, LLVM/PTX и отчёт об expected/observed входят в SHA-256, offline verify выполняет тот же проверяющий bit-oracle без GPU. Отдельный Lean `numeric_special.lean` показывает классификационные факты о конкретных кодировках binary32, без теорем о машинной арифметике. Пользователь подтвердил `task numeric-stage8` (14/14 GPU-профилей, включая по 13/13 special FP32 values в baseline/fused, все 15 Lean-файлов), затем `task verify-numeric-stage8` подтвердил сохранённый 72-файловый bundle. Актуальный SHA-256 manifest: `431dc70b2eadb5147b94c82b0567dd91f3680eb9c80cdda01bed1d5f9d7fe727`. В этом логе не представлен повторный полный `task test` на финальном HEAD; результат 75/75 Rust-тестов не подтверждается. LLVM/PTX/SASS/JIT, signaling NaN, произвольные overflow и полный IEEE-754 закрыты не были.

**Numeric Translation Validation: крупный срез локально подтверждён на `c63cd1f`, кроме Clippy.** Из Semantic Neural IR, guarded fusion и `GpuMemoryContract` строится `TranslationPlan` с ordered kernel roots, входами, shapes, broadcast и MatMul reduction; `GpuEngine::lower` выполняет структурную проверку LLVM/PTX до CUDA JIT, с fail-closed при нарушении RNE/FMA/FTZ, адресных шаблонов, ReLU guard или биекции entry points. Все 14 Stage 8 baseline/fused GPU-профилей прошли; каждый сохранил SHA-pinned `translation.plan`/`translation.report`, общий `translation_summary.txt` перепроверен командой `task verify-numeric-stage8`. Пользователь подтвердил `task fmt-check` (коммит `c63cd1f`), `task check`, **80/80 Rust-тестов**, CUDA+Lean, **101 evidence-файл** и успешный offline verify; manifest SHA-256 `7f4d473333b7360a5fbc75496b7bdbabc441c007fd862d29190719ddf03afa54`. `task clippy` на `c63cd1f` остановился на единственном `collapsible_if` в `translation.rs`; изменение синтаксиса подготовлено отдельно и его повторный локальный результат ещё ожидается. Проверка ограничена известными LLVM/PTX-шаблонами и не является доказательством полной семантической эквивалентности LLVM, PTX, JIT, SASS либо железа. Assurances: Lean ideal/dyadic conditional; per-kernel structural validation; concrete GPU oracle; JIT/SASS=TCB.

**Оставшиеся обязательства этапа 8:** моделировать устойчивость пороговых/условных вычислений с явными error margins; расширять содержательные invariants на другие операции по мере формализации; расширить диапазонную модель за текущие `i64`/60-bit ограничения там, где это нужно; формализовать FMA/contract, FTZ/subnormal, NaN/Infinity, signed zero, overflow и условные ветви; привязать численные теоремы к конкретным LLVM/PTX/JIT режимам либо честно оставить соответствующую часть в TCB. Предыдущий Stage 7 сохраняет незакрытые memory/race/translation-validation обязательства.

**Работы:** формальная семантика IEEE 754 binary32 (rounding, FMA, FTZ/subnormal, NaN/Infinity/signed zero, порядок вычислений/редукций), Exact FP32 или доказанные границы Numerical Error. Применять численные контракты к **любой** пользовательской функции, её вложенным блокам и их глобальной композиции; для условных решений учитывать пороги стабильности/переходы между ветвями. Никакие конкретные LLM-функции не обязательны для покрытия базовой арифметики.

**Готово, когда:** для поддерживаемого нестандартного составного вычисления Lean доказывает конкретную FP32-гарантию с областью входов и ε либо точное поведение, а межблочная композиция математически выводит применимый итоговый численный контракт; одного совпадения GPU и CPU недостаточно.

## 9. Собственная крупная архитектура, состояние и глобальные доказательства

**Работы:** Rust-проект-потребитель строит **новую архитектуру без шаблона Transformer** из иерархических проверяемых блоков. Стандартная организация нескольких исследований в одном workspace: `labs/<architecture-name>-<version>/`, где каждая лаборатория является отдельным Rust consumer package VeriNeural и владеет собственным artifact root. Предоставить средства для параметров и зависимостей, повторного использования блоков и, если требуется по выбранному примеру, состояния/переходов с явными инвариантами. Проверять зависимость сертификата от фактических значений или контрактов параметров. Сгенерировать не только отдельные теоремы для узлов, но и **содержательную теорему о заявленном свойстве всей композиции**. Добавить канонический экспорт полного Semantic Neural IR в один детерминированный Graphviz `architecture.dot`: nested module clusters без сокрытия внутренних узлов, stable instance paths, все tensor/parameter/state/control edges, shapes/dtypes/ports, domains, numerical contracts, proof/obligation statuses и GPU/lowering metadata. DOT должен быть самодостаточным для архитектурного анализа человеком или ИИ-агентом и содержать fingerprint исходного IR. После записи DOT pinned Graphviz из `flake.nix` обязан автоматически выполнить эквивалент `dot -Tpng architecture.dot -o architecture.png`; PNG выпускается вместе с DOT и `architecture.lean`, но остаётся производным визуальным артефактом. Пользовательские функции обновления параметров могут быть выражены общими вычислительными операциями; готовый Backward/optimizer для типовой LLM не требуется.

**Готово, когда:** как минимум одна нетиповая архитектура, состоящая из нескольких вложенных уровней блоков с собственными контрактами, компилируется в NVIDIA PTX, исполняется и проходит Lean-проверку локальных и **глобальной** теоремы. Для той же архитектуры автоматически создаётся один полный `architecture.dot`, по которому без обращения к другим файлам видны topology, hierarchy, shapes, component instances, parameter/state dependencies и статусы численных/формальных/GPU свойств; его IR fingerprint соответствует сертифицируемому Semantic Neural IR. Рядом автоматически создаётся `architecture.png` из этого DOT, а `architecture.lean` остаётся формальным sibling-артефактом в корне той же лаборатории/artifact root. Изменение параметров корректно проверяется на применимость гарантии. Не нужно демонстрировать Transformer, KV Cache, качество генерации текста или сервис инференса.

**Stage 9 первая крупная реализация (новая, локально не проверена):** самостоятельный Cargo workspace consumer `labs/coupled-field-0.1.0/` с глубиной 3 (`CoupledField → CoupledUnit → BoundaryCell`), 7 модулями, общими параметрами и парным fork/fan-in, без готовой LLM. Lean генерирует отдельную `architecture_global_nonnegative` для всего вложенного графа при явных исходных nonnegative гипотезах; GPU baseline/fused CUDA Graph сравниваются с независимым checked `i64`-oracle (`[4,10,8,12]`). Общий `snapshot_architecture` экспортирует детерминированный полный Semantic IR и самоописательный `architecture.dot` с module clusters, instance paths, shapes/domains/contracts, параметрами, provenance, input edges, GPU lowering metadata и SHA-256 fingerprint исходного IR; pinned Graphviz генерирует рядом `architecture.png`. Все 17 evidence-файлов, включая sibling `architecture.lean`, PNG, PTX/LLVM/translation и parameters.bits, включаются в SHA-256 manifest выбранного artifact root (по умолчанию lab package root). `--verify` без CUDA проверяет актуальность Rust IR/DOT, Lean и сохранённых GPU evidence; несовместимая id/version блокирует перезапись. Новый Rust API входит в reusable `verineural`, lab остаётся consumer. Выпуск новой лаборатории **не проверен** командами Cargo, Lean, Graphviz или CUDA; full Stage9 требования state/control/parameters proof/update, универсальный exporter и формальная GPU machine equivalence не объявлены закрытыми. Две новые Taskfile-команды `lab-stage9` и `verify-lab-stage9` локально выполняет пользователь. Stage10 `cargo verineural build --certified` намеренно не добавлен.


**P1 после review (исходный код, выполнение ещё не подтверждено):** экспорт DOT v3/auto-v3 больше не фиксирует все входы на одном ранге, различает ослабленные layout-ограничения для параметров и добавляет явные source/destination/module-port атрибуты рёбер. `ArchitectureSnapshot::verify` и auto-build используют одинаковое повторное рендеринг-сравнение PNG с сохранённым DOT. В `CoupledField` добавлена независимая `composition.lean` цель: первый результат ветви не превосходит итоговый `ReLU(Add(left, right))`, поскольку `right` имеет доказанную ReLU-неотрицательность; она не выводится исключительно из свойства последнего ReLU. Для новой сборки требуется 18 файлов в Stage 9 manifest. В репозитории пока остаются ранее сохранённые лабораторные DOT/PNG/manifest, созданные старой версией экспортёра: они должны быть **перегенерированы из Rust через `task lab-stage9` и `task lab-xor` в pinned devShell**, затем проверены offline и закоммичены только после успешного выполнения. Не считать старый bundle сертифицированным текущим генератором и не подменять новое GPU/Lean evidence ручным хешированием. Регрессионные Rust/Lean тесты и лабораторные CUDA прогоны должны быть подтверждены пользователем перед отметкой реализации как проверенной.



**P2 / VN-P13–VN-P14 (исходники подготовлены, локальная приёмка ожидается):** библиотечный `NeuralGraph::component("path")` вводит лёгкий путь для вызовов существующих модулей с позиционными или именованными портами, контекстными `BindingError` и атомарным rollback. `ProofGoal::nonnegative/bounded_above/generate_source/certify` позволяют объявить проверяемые цели без ручной сборки Lean-файла; `ObligationError` сообщает имя неподтверждённой теоремы, но не превращает генерацию исходника в proof. Независимый workspace consumer `labs/consumer-acceptance-0.1.0/` содержит три непохожих сценария: вложенный fork/fan-in DAG; двухшаговый state-transition со строгим invariant claim; tensor-heavy композицию matmul + пользовательский `PointwiseSpec` + mask/select + axis reduction (это **не** готовый Attention и не softmax). Для каждого сценария предусмотрены позитивные/негативные проверки, стабильный Semantic IR, явные theorem ids и GPU capability statuses. Собственная pointwise-операция в этой лаборатории не требует правок core, но остаётся **GPU unsupported** и не получает фиктивной FP32/SASS гарантии. Команды `task p2-consumers`, `task verify-p2-consumers` зависят от `prepare` (Rust autoformat). Наличие исходников **не является** доказательством прохождения тестов: для завершения P2 требуется локальное выполнение обеих задач, `task check`, `task clippy`, а также проверка, что ошибочные Lean-цели действительно отвергнуты.


**P2 frontend (исходники; локальная приёмка ожидается):** добавлены именованный `ModelBuilder`, атомарный `compose_inferred`, вывод структурных контрактов, привязка Tensor по имени с валидацией, выходы и независимые ProofGoal через `ArchitectureAutoCompiler::plan`. Четыре регрессионных теста покрывают рабочий блок, скрытый capture/rollback, незарегистрированные источники и некорректные binding. Пользователь локально подтвердил `task check`, `task test`, `task clippy` на `e829ff9` (151 Rust-тест пройден, 4 проигнорированы); это не Lean/GPU-сертификация. Для завершения выполнить локально `task check`, `task test`, `task clippy`. Параметры пока остаются IR input с метаданными роли, state/update и доказательство GPU hardware не реализованы.

**P2 consumer migration (исходники; новая локальная приёмка ожидается):** лаборатория XOR использует публичные ModelBuilder input/parameter/block/output/proof_goal и Model::bind/metadata/plan; consumer-специфичный 2→2→1 граф, четыре вложенных модуля и независимая Boolean XOR проверка остаются внутри lab/src/lib.rs. Убраны ручные MathContract, NeuralModule и input_roles, сохранён end-to-end sibling artifact workflow через исходный Semantic IR, Lean и CUDA Graph. Автоматическое доказательство всех именованных целей не следует из model planning: example как прежде kernel-check'ит только architecture_global_nonnegative, а proof_goal генерируется и проверяется на пригодность в Rust-тесте. Изменённый канонический IR делает прежние DOT/PNG/Lean/PTX/manifest устаревшими. Для нового среза выполнить task check, task test, task clippy, task test-lab-xor, task lab-xor, task verify-lab-xor; артефакты перегенерировать и проверить только из исходников, а не редактировать hashes вручную.

**P2 CoupledField migration (исходники; локальная проверка ожидается):** лаборатория Stage 9 объявляет три входа и два общих параметра через ModelBuilder, строит 7 вложенных модулей через ModelBuilder::block / NeuralGraph::compose_inferred и называет три выхода (final, left, right). Независимая реляционная цель coupled_branch_lower_bound теперь живёт в Model::proof_goals, а пример проверяет её Lean-кернелом через исходный composition.lean. Сохранены глобальная архитектурная nonnegative теорема для final, исходный i64-оракул, CUDA Graph baseline/fused и 18-файловая схема evidence. Тест Model::plan(AllOutputs) проверяет структурное планирование трёх выходов и общего multi-output NVPTX; исполняющий Stage9-пример по-прежнему запускает только final. Убраны ручные MathContract/NeuralModule и дублирование input_roles в лаборатории, но это не изменяет доказательную границу Lean/FP32/SASS. Перегенерация старых Stage9 Semantic IR, DOT/PNG, Lean/PTX и SHA-256 обязательна, ручное исправление манифеста недопустимо. Выполнение task check, task test, task clippy, task lab-stage9 и task verify-lab-stage9 ещё требуется.

## 10. Certified Build конструктора и проверка конечных критериев

**Работы:** целевой `cargo verineural build --certified` и интерфейс Taskfile; выбор architecture package через `--manifest-path`/эквивалент и настраиваемый `--artifact-root <path>` с Rust API-эквивалентом. По умолчанию artifact root равен корню выбранного Cargo package, поэтому стандартная лаборатория `labs/<architecture-name>-<version>/` хранит собственные `semantic.ir`, `architecture.dot`, `architecture.png`, `architecture.lean` и `SHA256SUMS` прямо в своём корне, а GPU/proof/certificate/report детали — в подпапках этого же root. Certified build проверяет deterministic DOT generation, соответствие `vn_ir_sha256` каноническому IR, автоматически рендерит PNG через pinned Graphviz, включает DOT/PNG в SHA-256 manifest и гарантирует, что все writers используют один выбранный artifact root. Относительный custom root разрешается относительно architecture package; абсолютный host path не влияет на semantic fingerprints; несовместимый bundle другой architecture id/version нельзя молча перезаписывать. Дополнительные SVG/PDF не являются источником истины и опциональны. End-to-end набор независимых архитектурно-нейтральных примеров; проверки критериев `SPEC.md`. Никакого serving/CI/деплоя в рамках этого этапа без отдельного разрешения.

**Готово, когда:** произвольная **поддерживаемая** пользовательская архитектура из Rust получает NVIDIA GPU-код, проверенные локальные/глобальные математические и заявленные численные гарантии, один полный самодостаточный `architecture.dot`, автоматически созданный из него `architecture.png`, явные границы TCB и воспроизводимый сертификат; недоказанная обязательная цель блокирует сертификацию. DOT байтово детерминирован для одинакового IR/config, DOT/PNG включены в manifest и однозначно связаны с Semantic Neural IR; PNG получен pinned Graphviz-рендером именно из сохранённого DOT. Две лаборатории с разными architecture id/version не конфликтуют артефактами при стандартных настройках, а custom artifact root не может молча перезаписать несовместимый bundle. Готовность определяется критериями архитектурно-нейтрального конструктора из `SPEC.md`, а не реализацией существующего LLM-стека.

**Stage 10, первый API-срез (исходники, локальная приёмка ещё не выполнена):** `Model::certify(CertifiedBuildConfig)` вызывает уже существующий `ArchitectureAutoCompiler::build`, обязательно проверяет все именованные цели Lean kernel + axiom audit и требует NVPTX lowering каждого именованного выхода. `Model::verify_certified` повторно проверяет current Semantic IR, DOT ↔ SHA-256, PNG, Lean, LLVM/PTX translation и SHA256SUMS без GPU. `identity.txt` связывает id/version, имена и роли источников, выходы и цели; сериализация декларации не зависит от Rust `Debug` или process-local `NodeId.graph`. Старый v1 Debug-shaped identity принимается при миграции только когда после удаления исключительно allocation-specific graph numbers вся декларация совпадает. Файл находится под SHA256SUMS, чужая id/version или изменённая декларация не может молча заместить bundle. Для XOR и CoupledField добавлены независимые thin-entrypoints `--verify`, `--artifact-root`, `--pinned-sha256` и Taskfile-команды `certified-xor`, `verify-certified-xor`, `certified-coupled`, `verify-certified-coupled`. Никакой новый semantic IR и собственный лабораторный Lean/CUDA backend не создаётся.

**Stage 10, продолжение (исходники, без локальной приёмки):** добавлен Cargo external subcommand `cargo-verineural`: `build --certified` / `verify --certified`, `--manifest-path`, `--artifact-root`, опциональный внешний SHA-256 pin при replay. Сборку выбранного consumer он делегирует в единственный `examples/<name>_certified.rs` пакета; определение самой архитектуры остаётся в `src/lib.rs`, а все доказательства/компиляция в едином библиотечном `Model::certify`. Для запуска без установки в PATH используются `task certified-build -- --manifest-path labs/xor-0.1.0/Cargo.toml` и `task verify-certified-build -- --manifest-path ...`. После успешной сертификации корневые `semantic.ir`, `architecture.dot`, `architecture.png`, `architecture.lean`, `SHA256SUMS`, `report.txt`, `identity.txt` являются относительными Unix symlink на единый `auto-build/`; старые файлы лаборатории с той же id/version переносятся в `.verineural-pre-stage10/`, неизвестные файлы не перезаписываются. Это важное ограничение: legacy migration нескольких ссылок не является единой атомарной операцией, публикация внутреннего `auto-build/` использует staging с возможным промежутком между directory renames. Настоящий one-switch root publish, полная поддержка произвольных consumer без certified example, формальная FP32/SASS гарантия и локальные Lean/LLVM/Graphviz/GPU тесты ещё не готовы; Stage 10 нельзя считать завершённым до подтверждённой приёмки. Старые Stage9 генераторы пишут в те же корневые имена, поэтому после certified-build их запуск может сделать bundle недействительным: выполнять Stage9 генерацию перед Stage10, повторно сертифицировать после изменений.

**Stage 10, повторная сертификация и pinned replay локально подтверждены (2026-10-23, `4871740`):** после обновления `dev` пользователь выполнил `task check` (успешно; `cargo fmt` автоматически создал и отправил `4871740`), `task verify-certified-stage10` (**4/4** tool-dependent теста, включая повторную публикацию существующего каталога через Linux atomic exchange, стабильный манифест, неверный внешний pin, чужую identity, ложную Lean-цель и tampering). Повторный `task certified-build` для XOR в существующий `/tmp/verineural-stage10-xor-v2` прошёл, `SHA256SUMS` сохранил хеш `6bc265ca583d2e4c6803fecf74c5b7df0f39f3817d7902d425ccc6deeda77797`, `task verify-certified-build --pinned-sha256 <этот-хеш>` прошёл с Lean 4.29.1. Предыдущий прогон на `84980ea` подтвердил `task test`, `task clippy` без предупреждений и первый build/replay обеих лабораторий (CoupledField manifest `0788971dd228bc51de0105780817d45d9141c832f1fc18d279cf386d83c47f44`). **Принят рабочий Stage 10 certified-build срез**: Rust API → Semantic IR → Lean kernel+axiom audit → LLVM/PTX structural validation → Graphviz DOT/PNG → SHA-256 → offline replay → deterministic reissue. **Полный Stage 10 по целевой SPEC.md остаётся открытым**: первая миграция семи root-alias не одна атомарная операция; потребителю нужен `examples/<name>_certified.rs`; проверки исполнения CUDA Graph на GPU и формальной LLVM/PTX/JIT/SASS/FP32 эквивалентности этот срез не предоставляет. Ни эти, ни численные гарантии сверх указанного не объявлять доказанными.

**Rotating-matrices P2 (код в dev; приёмка совместно с P1 ожидается):** общие конструкторы `Tensor::from_vec/from_slice/zeros`; `NeuralGraph::sub/div/exp/reduce_max/broadcast_to` и одноимённые CPU references; расширение `PointwiseExpr` и canonical IR, DOT, тесты форм, row-major семантики, NaN и честных unsupported GPU/Lean профилей. Лаборатория не объявляет собственный tensor helper и SHA256SUMS/copy-publish Stage10; structural/index routing GPU lowering уже закрыт отдельным P0 срезом. Никаких ready-made softmax/stochastic-choice примитивов или фиктивных FP32 доказательств.

**P1 state/execution — ФЛАГМАНСКИЙ CONSUMER ПОДКЛЮЧЁН, локальная приёмка ожидается:** `SourceRole::State`, `ModelBuilder::state/next_state/block_multi`, identity с transition links и owned `ModelExecution`. Runtime metadata копируется при prepare, поэтому execution не держит Rust lifetime на `Model`. Deterministic rotating-matrices использует один captured architecture CUDA Graph; parameters и dictionary обновляются один раз на forward, `token_bits` по шагам, state feedback выполняется D2D в `step()` без H2D rebinding. Stochastic/fixed-route policy тоже больше не использует raw `GpuExecution` и host-derived rotated matrices: host выбирает action, затем один из восьми route-specific `ModelExecution` исполняет rotation + transition из Semantic IR. Выполнить локально общий acceptance; аппаратный replay отдельно требует NVIDIA.

**P1 numeric branch stability (код; локальная Lean-приёмка ожидается):** добавлены generic `RneConditionEvidence` и `RneSelectPlan` для канонического `PositiveMask → SelectPositive`. Condition и обе ветви имеют независимые RNE contracts; stable positive/nonpositive decision наследует contract только выбранной ветви, а zero-crossing interval fail-closed через `UnstableBranch`. Lean `Numeric.lean` содержит отдельные conditional composition lemmas, а generated theorem сохраняет concrete q/bounds/ε и margin. Проверка: `task verify-numeric-branch`. Это conditional source-level RNE theorem, не GPU/LLVM/PTX/SASS proof и не общий comparison framework.

**P1 recurrent numeric rollout (код; локальная Lean-приёмка ожидается):** `RneRangePlan::compile_with_source_contracts` принимает уже доказанный source `ApproxContract`; `RneStateRolloutPlan` на fixed horizon переносит `next_state` contract в state source следующего шага, поэтому ε действительно накапливается, а не обнуляется на temporal boundary. Fresh non-state inputs остаются exact на каждом шаге. Feedback разрешён только при invariant q-scale; Mul/MatMul-подобный переход, меняющий q состояния, отклоняется. `task verify-numeric-state` проверяет Rust contract chain и kernel-check каждого step source. Это ещё не произвольная индукция по времени и не machine FP32 proof.

**P1 arbitrary-horizon state invariant (код; локальная Lean-приёмка ожидается):** `generate_inductive_state_nonnegative` берёт один reusable consumer transition `state,input → next_state,output`, требует прямые nonnegative F32 sources, state-preserving shape/dtype и явный `Nonnegative` claim на `next_state`. Lean определяет рекурсивное состояние по `Nat` и доказывает `∀ t` индукцией, повторно используя локальную theorem transition-а, а не генерируя N копий шага. Любой extra source или недостающий claim отклоняется. `task verify-inductive-invariant` проверяет positive/negative cases и real Lean+axiom audit. Это ideal-Int invariant, не FP32/GPU temporal proof.

**P1 reusable module-definition proof (код; локальная Lean-приёмка ожидается):** `generate_reusable_module_nonnegative` нормализует consumer-defined single-output module body в generic Lean definition, доказывает один definition-level `Nonnegative` theorem и создаёт отдельные application theorems для concrete instances. Reuse требует совпадения name/version/`MathContract` и normalized body semantics; одинаковая identity с отличающимся телом fail-closed. Каждый instance заново закрывает declared `NonNegativeFinite requires`, даже если generic property не использует их в proof term. Gate: `task verify-reusable-module-proof`. Persistent cache, multi-output definition reuse и общий theorem dependency graph остаются следующими срезами.

**P1 global reusable-module composition (код; локальная Lean-приёмка ожидается):** architecture-level `generate_global_nonnegative` теперь проходит общий application-order DAG `ModuleInstanceRef` и композирует как single-, так и multi-output modules по конкретным output ports. Совместимые invocations/ports группируются по kind/identity/full contract/normalized selected-output body, получают один shared definition theorem и отдельные concrete application theorem; global theorem использует именно applications и fail-closed при `ModuleDefinitionMismatch` либо недостающей premise конкретного вызова. Existing single-output theorem names сохранены; multi-output applications получают явный `_port_N`. Ограничение остаётся по свойству ideal-Int `Nonnegative`; общий typed theorem dependency graph для произвольных predicates ещё открыт.

**P1 multi-output reusable definition proof (код; локальная Lean-приёмка ожидается):** `generate_reusable_multi_module_nonnegative` связывает все выходные F32 `Nonnegative` свойства одного `MultiOutputModule` в одну generic conjunction theorem от общей нормализованной tuple-definition; каждый порт каждого concrete instance получает отдельную application theorem и закрывает declared premises. Reuse требует совпадения полной named contract/identity/version и всех output semantics; любое несовпадение fail-closed. Тот же тип module теперь участвует и в `generate_global_nonnegative` как producer конкретного output port; два compatible multi instances переиспользуют definition theorem выбранного порта, а mismatched selected-port semantics отклоняются. Проверка `task verify-reusable-module-proof` включает реальный Lean kernel/axiom audit. Не заявляется универсальная multi-output композиция произвольных предикатов или FP32/GPU машинных доказательств.

**P1 static tensor slice / attention-like consumer (код; локальная Lean-приёмка ожидается):** `Operation::SliceAxis { axis,start,len }`, `NeuralGraph::slice_axis` и `Tensor::slice_axis` реализуют статический contiguous interval с сохранением ранга, явными bounds и row-major семантикой. Canonical IR фиксирует axis/start/len; Lean `sliceAxis` и `sliceAxis_nonnegative` покрывают ideal-Int semantics/property. GPU и RNE backend пока честно отвергают slice. Независимый consumer режет packed [2,6] на Q/K/V-подобные [2,2], делает transpose/matmul и доказывает идеальную nonnegative композицию без готового Attention API. Проверки: `task verify-tensor-slice`, `task p2-consumers`, `task verify-p2-consumers`.

**P1/P12 explicit GPU capability plan (код; локальная проверка ожидается):** `GpuProgram::capability_plan` проверяет каждый запускаемый kernel против соответствующих memory и translation evidence и отклоняет необъяснённый удалённый fusion node. Fusion evidence повторно фиксирует single-use и сохранение selected outputs. `GpuAssuranceBoundary` разделяет generated memory Lean obligations, structural LLVM/PTX validation, отдельный numeric proof requirement, CUDA runtime acceptance и неустановленную JIT/SASS correspondence. `capability.plan` входит в auto-build bundle и offline replay. Gate: `task verify-gpu-capabilities`.

**P06/P12 certified memory obligations (код; локальная Lean-приёмка ожидается):** memory contracts теперь экспортируются как named `LeanProgram` theorems. Auto-build проверяет и axiom-audit'ит `gpu_N/memory.lean` и `gpu_shared/memory.lean`, добавляет зависимости в общий audit и SHA manifest; offline replay повторяет source comparison + kernel check + audit. Общий GPU compiler только генерирует obligations, поэтому это усиливает certified path без ложного утверждения об LLVM/PTX pointer semantics. Gate: `task verify-gpu-capabilities`.

**P1/P10 explicit storage semantics (код; локальная проверка ожидается):** текущий layout-профиль формализован как contiguous row-major strides. `NodeStorage` различает owned allocation и zero-copy `Reshape` alias, а `storage_root/aliases` разрешают alias chain. Canonical IR поднят до v4 и связывает storage metadata с fingerprint; DOT экспортирует тот же storage contract. GPU allocation и kernel planning читают `NodeStorage`, а не повторяют собственное знание о Reshape. Reordering ops вроде Permute/Slice остаются owned contiguous outputs. Gate: `task verify-storage-semantics`. Arbitrary strides/noncontiguous views остаются открыты.

**P1 hardware acceptance fixture (код, локальный CUDA-прогон ожидается):** `task gpu-state-replay` вызывает `examples/stateful_model_gpu.rs`, собирая один consumer-defined 3-output transition с двумя публичными readouts и приватной state link. Один GPU session/compiled CUDA Graph, `run` плюс пять `step`, точные FP32 bit fixtures для состояния и выходов, смена input и параметра без recapture, явный reset и отказы на неверные роли, формы, значения. CPU unit tests и ранее подтверждённый `task gpu-multi-output` не заменяют проверку этого D2D replay; статус GPU acceptance отмечать завершённым только после фактического успешного запуска на NVIDIA. Не выдавать это за доказательство CUDA/JIT/SASS/FP32 или произвольной рекуррентной сходимости.

**P1 P0 reliability fixes (код; локальная приёмка ожидается):** GPU prepare расширяет roots всеми State-источниками, не изменяя pure Semantic IR и состав proof/certificate roots. D2D feedback проверяется до первого replay. Публичные выходы, алиасные перезаписываемым state через Input или Reshape, получают GPU-only snapshots до feedback в `run/step`; чтение `output(name)` адресное, без копирования прочих output/next_state на CPU. Добавлен Rust preflight regression для полностью заменяемого unread state и GPU acceptance `task gpu-state-replay` с двумя state (включая полностью недостижимое из обычного DAG), прямым alias и reshape-view. Эти исправления не расширяют GPU операторы и не доказывают FP32/JIT/SASS или атомарность при CUDA-ошибках.

**P1 GPU feedback reliability / P2 IR transaction patch (код, совместная локальная приёмка ожидается):** feedback/snapshot D2D enqueue или sync error устанавливает `ExecutionPoisoned`; остальные операции `GpuExecution` fail-closed до пересоздания. Неверная схема state-link отклоняется ещё до копирования и не poison. CPU tests имитируют частичный DMA failure и synchronize failure, без заявления об аппаратной fault injection. Пять clone-based builders (`apply_module_with`, `compose_inferred`, `compose_multi_inferred`, `scan_fixed`, `apply_multi_module_with`) переведены на вложенные RAII append transactions с откатом `nodes`, `modules`, `multi_modules`, `module_order` на Err и unwind. Добавлены regressions nested rollback и catch_unwind. Проверить совместно с предыдущими P0: `task check`, `task test`, `task clippy`, `task gpu-state-replay`, затем `task gpu-multi-output` и зависимые потребительские проверки. CUDA/FP32/SASS формально не доказаны, после DMA failure автоматического восстановления нет.

**P1/P2 review fixes (код, локальная проверка ожидается):** `GpuExecution` реализует единый fail-closed lifecycle для CUDA H2D/D2D/D2H/Graph/synchronize/benchmark: `ExecutionPoisoned` после CUDA-ошибки, `OutputNotReady` до первого replay и после валидного source update; ошибки preflight не poison. Аппаратная приёмка `task gpu-state-replay` также проверяет запрет чтения неготовых результатов. `NeuralGraph::dependency_scope` больше не сканирует каждый модуль для каждого посещённого узла: строит локальный индекс module outputs и сохраняет declared unused inputs и duplicate output/module bindings. Совместно прогнать `task check`, `task test`, `task clippy`, `task gpu-state-replay`, `task gpu-multi-output`, `task p2-consumers`, `task verify-p2-consumers`. CUDA/FP32/SASS не доказаны.


**P0 structural/indexing GPU lowering — КОД ЗАКРЫТ, локальная приёмка ожидается:** GPU backend принимает только профиль, реально требуемый rotating-matrices: rank-2 transpose (`Permute [1,0]`), rank-2 `ReverseAxis(0/1)`, contiguous binary `Concat(0)`, rank-2 `ArgMax(axis=1, First)` и scalar `Take(axis=0)`. Для новых kernels добавлены memory contracts, deterministic translation-plan kinds и structural LLVM/PTX validation; неподдерживаемые axes/ranks/tie policies остаются fail-closed. Runtime output ABI теперь сохраняет `DType`: `TensorValue::{F32,Index}` проходит через `read_value/read_values`, `execute_value/execute_values` и `ModelExecution::output_value`; device ArgMax `i32` читается как Index, а старые F32-only методы отклоняют Index вместо bit reinterpretation. Добавлен отдельный ignored NVIDIA acceptance для публичного ArgMax output. Lean memory-address source покрывает row-major адреса transpose/reverse/concat/argmax/take на конкретных размерностях; `Take` формулирует предпосылку `selected < choices`, а reverse-row translation plan фиксирует `rows`. Evidence model rotating-matrices объявляет `recurrent_state` через `ModelBuilder::state` и связывает `next_state`; deterministic runtime теперь компилирует и replay-ит тот же единый graph `router → ArgMax → Take → transition → readout`, без host route/rotation. Stochastic training остаётся явно внешней sampling policy. Новый typed-output hardware regression агентом не запускался. Это арифметический proof слоя индексов и structural translation evidence, не доказательство CUDA pointers, LLVM/PTX/JIT/SASS/FP32.

**P0 VN-P01..VN-P06 acceptance — КОД ЗАКРЫТ, локальная проверка ожидается:** supported typed relational predicates, external versioned Pointwise primitive semantics, module capture/ownership rollback, independent multi-goal fork/fan-in composition and certified Lean axiom/fingerprint/SHA replay are wired into one developer gate: `task p0-acceptance`. Deliberately false predicates/claims and unsupported semantics remain fail-closed. This closes the P0 acceptance criteria only for the documented supported DSL; arbitrary user Lean text, arbitrary atomic math, IEEE-754 machine correspondence and signed provenance are not smuggled into the claim and remain outside P0.

## Общие ограничения для всех этапов

- **Никаких** CI/CD, GitHub Actions, hooks, контейнеризации и прочей инфраструктуры без явного согласия пользователя.
- **Не создавать pull request**. Изменения в `main` интегрировать только fast-forward, без merge-коммитов и принудительной перезаписи истории; порядок указан в корневом `AGENTS.md`.
- Команды сборки/тестирования описывать в Taskfile; по умолчанию их **выполняет локально пользователь**, а не агент.
- Все новые Rust-компоненты принадлежат **единому Cargo workspace**; необходимые инструменты включать в корневой `flake.nix`.
- Реализовывать крупные работающие срезы, не создавать бессвязную модуляризацию или абстракции «на будущее».
- Целевая спецификация описывает конечное состояние. В промежуточных версиях нельзя выдавать наличие контрактов, Lean-файлов или GPU-тестов за доказательство реального машинного исполнения.

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

# Автоматическая сборка VeriNeural: Rust IR → Lean → NVIDIA NVPTX

Первый библиотечный вертикальный срез: `ArchitectureAutoCompiler`. Пользователь определяет собственные `NeuralModule`, `MultiOutputModule`, `PrimitiveSpec` или `ParametricModule`, собирает `NeuralGraph`, перечисляет архитектурные выходы и **явные** математические цели. Готовые Transformer/RNN/LLM классы для этого не требуются.

## Публичный API

```rust
let compiler = ArchitectureAutoCompiler::default();
let tasks = vec![ArchitectureProofTask::Goal(ProofGoal {
    name: "root_property".into(),
    output: final_output,
    kind: ProofGoalKind::Nonnegative { assumptions: vec![] },
})];
let report = compiler.build(
    &graph,
    &[intermediate_output, final_output],
    &tasks,
    GpuRequirement::AllOutputs,
    std::path::Path::new("target/my-architecture"),
)?;
```

Для фиксированного рекуррентного горизонта: `ArchitectureProofTask::FixedScanNonnegative { name, scan }`. Программное создание `NeuralGraph` и объявление контрактов остаются на Rust, а не на отдельном несинхронизированном GPU/Lean DSL.

- `plan()`: повторно проверяет DAG-зависимости, арность, shape/domain контрактов, захваты через границы модуля и привязки output-портов; запрещает цели вне зависимостей выбранных выходов; генерирует одну Lean-модель для **всех выбранных выходов**, индивидуальные Lean-цели и GPU-планы для каждого выхода. Без вызовов Lean/LLVM или записи файлов. Ошибочная форма/порт и неподтверждаемая известными правилами цель отклоняются. Произвольное ложное равенство остаётся Lean-обязательством и отклоняется при `build`, а не объявляется доказанным при `plan`.
- `build()`: перезаписывает **только зарезервированный подкаталог `<root>/auto-build/`**, запускает Lean kernel и `#print axioms` для общего файла и каждой независимой цели, выполняет `GpuEngine::lower` и его проверку LLVM/PTX для каждого поддерживаемого выхода. `report.txt` появляется **только после успеха всех обязательных шагов**. При неуспехе прежняя успешная запись не сохраняется.
- `GpuRequirement::AllOutputs`: отсутствие GPU lowering **хотя бы одного** выбранного выхода даёт ошибку.
- `GpuRequirement::AllowUnsupported`: поддерживаемые выходы компилируются, неподдерживаемые явно маркируются `unsupported` без ложного PTX/certificate.
- В `report.txt`: входные гипотезы Lean, доказанные имена и число теорем для отдельных целей, транзитивные аксиомы каждого проверенного theorem, версия Lean, число структурно проверенных связей и портов, статусы PTX по выходам.

- `auto-build/semantic.ir` является канонической graph-only сериализацией `NeuralGraph::canonical_ir()`: узлы, операции, формы, домены, версии и интерфейсы модулей входят в неё; process-local graph ID, выбранные outputs, proof goals и GPU policy не входят. Выборы конкретной сборки связываются реконструируемым `AutoBuildPlan`, производными Lean/GPU/DOT артефактами и полным `SHA256SUMS`.
- Канонический формат имеет явную версию (`verineural-semantic-ir-v3`); dtype узлов и портов входит в canonical semantics. Bundle старого формата не мигрируется неявно: его нужно пересобрать из текущего Rust-графа, чтобы новый fingerprint, DOT и `SHA256SUMS` снова описывали одну сборку.
- По успешной сборке `auto-build/SHA256SUMS` охватывает `semantic.ir`, `architecture.dot`, `architecture.png`, каждый Lean исходник, `lean/audit.txt`, `report.txt`, все сохранённые `graph.ll`, `graph.ptx`, `translation.plan` и `translation.report`. `AutoBuildReport::manifest_sha256` возвращает хеш самого манифеста для внешнего закрепления; старые данные удаляются до сборки.
- `ArchitectureAutoCompiler::build` сохраняет полный детерминированный `architecture.dot` из того же `semantic.ir`, что участвует в Lean/NVPTX сборке. В DOT: все узлы и реальные зависимости (ровно одно ребро на IR-вход), вложенные кластеры обычных и многовыходных consumer-модулей, именованные входные/выходные порты, контракты/домены/shape/dtype, reachability и отдельные статусы математической проверки/GPU. DOT содержит `vn_ir_sha256`, равный SHA-256 сохранённого `semantic.ir`; Graphviz из Nix devShell сразу создаёт `architecture.png` из сохранённого DOT. Никаких relay-рёбер для портов, не требующих IR-связи, не добавляется.
- `ArchitectureAutoCompiler::verify_bundle(&graph, &outputs, &tasks, gpu_policy, root, pinned_sha256)` проверяет наличие **ровно ожидаемого набора** манифестных записей, содержимое файлов, каноническую связь с **текущим** Rust IR, байтовое совпадение DOT с повторной генерацией и PNG с повторным Graphviz-рендером, заново запускает Lean и аудит разрешённых аксиом, а также повторно проверяет сохранённый PTX против сформированного по текущему графу LLVM/translation plan. **CUDA драйвер не требуется**, но для воспроизведения PNG нужен Graphviz из devShell. `pinned_sha256: Some(...)` сравнивает digest с внешним эталоном; `None` означает replay без проверки происхождения.


Рабочий пример: `task auto-architecture` либо `cargo run -p verineural --example auto_architecture`. Повторная проверка без LLVM-компиляции и CUDA: `task verify-auto-architecture`; для проверки отдельно сохранённого эталона digest: `cargo run -p verineural --example auto_architecture -- --verify --pinned-sha256 <SHA256SUMS digest>`. Создаёт два пользовательских блока ReLU и модуль объединения `Add`, проверяет `Nonnegative` всей fork/fan-in композиции и компилирует три выбранных выхода в NVPTX.

## Многовыходный NVIDIA CUDA Graph

- `GpuEngine::compile_outputs(&graph, &[...])` собирает единый DAG зависимостей выбранных узлов. Один kernel принадлежит одному вычисляемому IR-узлу: общий producer исполняется один раз; `Reshape` остаётся zero-copy alias; источник, явно выданный как output, запрещено устранять через experimental fusion.
- `GpuProgram::outputs()`, `GpuProgram::kernel_count()` и `GpuExecution::read_outputs()` дают упорядоченный список всех результатов одного captured CUDA Graph. `GpuExecution::update_inputs` меняет общий входной набор без повторного захвата. Старые `compile`, `prepare`, `read_output`, `execute` работают через один выход и сохраняются.
- Если все выбранные архитектурные выходы поддерживаются и число выходов больше одного, `ArchitectureAutoCompiler` теперь в дополнение к совместимым `gpu_{i}` создаёт `gpu_shared/graph.ll`, `graph.ptx`, `translation.plan`, `translation.report`. Эти артефакты входят в `SHA256SUMS` и заново валидируются при `verify_bundle`. При `AllowUnsupported` и хотя бы одном неподдерживаемом выходе **общий GPU план не выпускается**, иначе возможно ложное заявление о поддержке всей архитектуры.
- `task gpu-multi-output` запускает consumer-пример с четырьмя выходами, объединённой Lean-моделью и двумя воспроизведениями **одного** CUDA Graph (два набора f32 входов), сверенными с независимыми точными fixtures. Команда требует NVIDIA CUDA GPU и Nix devShell, в отличие от offline `task verify-auto-architecture`.

Существование общего PTX и проверка конкретных GPU-результатов не доказывают LLVM/PTX/JIT/SASS или полное соответствие FP32 математической модели. Аудит Lean-аксиом остаётся независимым от CUDA-выполнения.

## Границы гарантии

**Lean proven** относится к идеальной `Int`-математике и исключительно к сформулированным свойствам для выбранного подграфа с явными предпосылками. **Связанность/типы модулей** повторно проверяются структурно на Rust IR; такая проверка не является доказательством соответствия GPU-программы Lean-модели. Проверка `TranslationPlan` обнаруживает известные несовместимости LLVM/PTX, но не доказывает битовую IEEE-754 или корректность PTX/SASS/железа. GPU-код *сгенерирован*, не *запущен*. Модели в которых есть неподдерживаемые операции, не получат GPU `proven`.

Пока нет глобального пользовательского языка произвольных математических предикатов, автоматического запуска общей CUDA Graph execution непосредственно из `ArchitectureAutoCompiler::build` и индукционного доказательства произвольного горизонта. Общий многовыходный API GPU и его отдельный CUDA-пример теперь существуют. DOT/PNG являются производными визуальными артефактами, не формальным доказательством LLVM/PTX/JIT. Манифест является защитой целостности и привязкой к актуальному графу, а не криптографической подписью издателя. `snapshot_architecture` и auto-build имеют разные presentation/build metadata, но используют одну canonical graph serialization и одну нормализованную модель иерархии module instances. Эти пункты остаются открытыми.

# polytropos-alpha

Исследовательский проект **VeriNeural**: импортируемая Rust-библиотека, компилятор и система формальной верификации нейросетевых вычислений и архитектур.

## Состояние

**Этап 2/10: Rust Neural API и Semantic Neural IR реализован в исходном коде; локальная сборка и тесты не подтверждены.** Этап 1 (инициализация) выполнен. Следующий этап по [ROADMAP.md](docs/verineural/ROADMAP.md) — генерация проверяемых Lean 4-моделей.

- `crates/verineural/` — импортируемая Rust-библиотека, не приложение.
- `Tensor<T, Shape>` — владеющий данными тензор, проверяющий размеры и длину буфера.
- `NeuralGraph` — append-only DAG из типизированных узлов; проверяет арность, формы, принадлежность узлов графу и ограничения доменов.
- `NeuralModule` и `MathContract` — компонуемые модули с проверкой формы и консервативно выводимого домена; ошибочная композиция отклоняется без изменения графа.
- `FP32Policy` — требования к FP32 округлению, FMA, subnormal и порядку редукции; пока **не доказанная семантика** выполнения.
- `OperationSemantics` — идеальная операция, требуемая FP32 policy и `GpuSemantics::NotLowered`; GPU-код не создаётся.
- CPU reference для Add, Mul, MatMul, ReLU, Sqrt и Reshape. Это расчёты Rust `f32`, не доказанные exact-FP32 kernels.

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

**Граница гарантии:** `graph.input(shape, Domain::Finite)` регистрирует *предположение пользователя*, а не автоматически проверяет реальные данные. Для конкретного `Tensor<f32>` используйте `validate_domain` до исполнения. Гарантии модулей проверяются по консервативно выводимой информации графа; арифметическое переполнение означает, что два конечных входа не дают автоматически гарантированно конечный выход. Узлы `sqrt` без установленного неотрицательного конечного домена отклоняются. Редукции и GPU пока не реализованы. Наличие IR и контрактов не заменяет доказательство Lean.

## Локальная разработка

```sh
nix develop
task --list
task check
task test
task clippy
task fmt-check
task example-stage2
```

Согласно [agents.md](agents.md), сборка, тесты и другие тяжёлые команды выполняются **пользователем локально**, а не ИИ-агентом без отдельного разрешения. Для будущего GPU-исполнения необходим совместимый NVIDIA GPU и системный NVIDIA-драйвер.

[SPEC.md](docs/verineural/SPEC.md) описывает всю конечную систему; [ROADMAP.md](docs/verineural/ROADMAP.md) — последовательность 10 этапов. Никаких сертификатов исполнения, LLVM/PTX и Lean proof engine этап 2 не содержит.

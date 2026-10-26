# Rotating Matrices 0.1.0

Исследовательская лаборатория **двух матриц 64×64 с 4 пространственными поворотами каждая**.
Архитектура собирается потребителем публичного API VeriNeural, а не переносится в `crates/verineural`.

## Схема

```text
train x: UTF-8 bytes
       │
       ▼
Dictionary.fit(train only) -> byte-id tokens -> 9-bit base2 (in 1×64 tensor)
       │                      │
       └-> histogram 1×64    │
                  │          │        previous h_t (1×64)
                  └──────────┼──────────┤
                             │          ▼
                             │    router logits 1×8
                             │          │
                             │    sample / argmax
                             │       (core_id, rotate)
                             ▼          │
                       core transition <- select one of R0/R90/R180/R270(W0 or W1)
                             │
                             ▼
                          h_(t+1)
                             │ (next byte: feed back as previous h)
                             ▼
                  after last byte: head 1×1
                             │
                      round(256 * scalar)
                             ▼
                       one integer y
```

`W0` и `W1` имеют по **4096 независимых параметров**, всего 8192. Повороты
`R0/R90/R180/R270` строятся из базовых весов для каждого шага: **это 8
действий маршрутизатора, а не 8 независимых матриц**. Поворот означает
пространственный поворот массива весов, а не перестановку токенов.

Переход: `h_next = ReLU((h + bits(token_id + 1)) × rotated_weight + bias)`.
Router: `logits = (dictionary_histogram + h) × router_weights + router_bias`.
Readout: `scalar = h_final × head_weights + head_bias`; итог один **целочисленный**
ответ в `[0,255]` (не ограничен одной десятичной цифрой).

## Данные

40 именованных операций: 8 простых (`+`, `*`, `XOR`, `AND`, `OR`, `SUB=abs(a-b)`,
`SHL`, `SHR`) и 32 композиции из них с `+`, `XOR`, `AND`, `OR`.
Аргументы `a,b ∈ [0,15]`. Составные ответы берутся по модулю 256.
Примеры: `"10 + 9" -> 19`, `"1 AND 0" -> 0`.
Всего 10 240 пар `(операция, a, b)`; 20% комбинаций операндов (одни и
те же для всех операций) полностью исключаются из обучения.
Словарь строится только на train. Валидация не обновляет параметры.

## Запуск (из корня workspace)

```sh
task lab-rotating
# Inspect + train 128 updates; training требует GPU. Сертификация запускается отдельно.

task inspect-lab-rotating
# Только inspect: схема, формы, словарь и diagnostic inspect.semantic.ir без GPU/Lean

task test-lab-rotating
# Разработчик локально запускает CPU-only тесты

task train-lab-rotating -- 128
# Требует NVPTX llc и NVIDIA CUDA Graph; нет тихого CPU fallback

task certify-lab-rotating
# Требует Lean + LLVM NVPTX; structural/indexing routing уже имеет GPU lowering

# Сквозная приёмка (только в pinned Nix devShell):
task verify-api-acceptance
# P0/P1/P2 контракты, Lean-аудит и независимые consumer-сценарии

task gpu-state-replay
# Named state feedback, persistent CUDA Graph, проверка битов FP32

task verify-lab-rotating-gpu
# GPU ArgMax/Index и восемь маршрутов против независимого CPU-оракула
```

Потребитель теперь использует `Tensor::from_slice` вместо собственного helper.

`artifacts/inspect.txt`, `artifacts/inspect.semantic.ir` и `artifacts/training-report.txt`
остаются диагностикой/результатами лаборатории. Ранее опубликованные
артефакты и результаты обучения сохранены в Git как исследовательские
свидетельства. Только устаревшие симлинки `artifacts/certified/*` на
ещё не созданный `auto-build/` удалены из индекса: нужный bundle
создаётся командой сертификации и игнорируется Git. Stage 10 публикует единый
сертифицируемый bundle в `artifacts/certified/auto-build/` с собственными
`SHA256SUMS`, `identity.txt`, Lean, PTX, DOT/PNG и manifest. Лаборатория
не копирует файлы из bundle вручную и не пересчитывает SHA-256 сама.
Structural/indexing P0 lowering уже входит в общий architecture graph.
Успех `--certify-architecture` всё ещё зависит от локально доступных Lean/LLVM
и проходит fail-closed через translation/memory evidence; это не сертификат
IEEE-754/JIT/SASS или качества обучения.

`architecture.png` рендерится VeriNeural/Graphviz именно из сохранённого
`architecture.dot`. Общий Lean генерируется из объединённого Semantic Neural IR,
в котором присутствуют router logits, recurrent transition и terminal readout.
Deterministic runtime исполняет единый Semantic IR в одном CUDA Graph;
для стохастической политики отдельные route-specific графы остаются вне certified forward.

P0-граница `selected_rotated_weight` из общего IR удалена. Evidence-модель
строит восемь вариантов двух параметров только универсальными
`permute + reverse_axis + concat`, получает индекс через `argmax` с явной
политикой `First` и выбирает матрицу через `take`.

GPU lowering для `permute/reverse_axis/concat/argmax/take` реализован в узком
профиле rotating-matrices вместе с translation audit и Lean bounds для адресов.
Обычный deterministic forward исполняет этот же объединённый graph:
`router → ArgMax → Take → transition → readout`. Стохастическое обучение
остаётся отдельной host sampling policy и не объявляется CUDA/FP32-сертифицированным.

Для публикации реальных результатов после запуска:

```sh
git add labs/rotating-matrices-0.1.0/artifacts/
git commit -m "lab(rotating-matrices): publish local run artifacts"
git push origin dev
```

После обучения `artifacts/` также содержит `training-report.txt`
с FP32 bit-level параметрами, accuracy/MAE/RMSE, распределением действий и
количеством верных ответов по операциям. До/после обучения оцениваются одни
и те же validation-примеры. Сэмплер распределяет проверочные пары по всем
40 операциям, не выбирая 40 нулевых ответов в первом окне проверки.
Выводится также baseline постоянного ответа, равного медиане **только train**
ответов, измеренный на том же наборе validation-примеров.
После обучения дополнительно сравниваются восемь
фиксированных маршрутов (по 40 отложенных примеров каждый). Декодирование использует целое число, например 19.

## Обучение и честная граница верификации

* Ядра выполняются через `GpuEngine::prepare` + `GpuExecution::update_inputs` /
  `replay` / `read_output`, то есть VeriNeural CUDA Graph. Нет собственного
  CUDA kernel и второго CPU forward.
* Deterministic runtime использует один owned `ModelExecution` и shared CUDA
  Graph из `architecture_model()`; parameters/dictionary обновляются один раз
  на forward, token по шагам, state feedback выполняется D2D. Stochastic/fixed
  routing тоже использует named `ModelExecution`: host выбирает только action,
  а каждый из восьми route-specific графов строит поворот базовой матрицы
  универсальными `permute/reverse_axis` внутри Semantic IR.
* `architecture.lean` kernel-check'ит определения, формы и локальную семантику
  объединённого поддерживаемого графа. Глобальная теорема корректности
  маршрутизации и произвольного recurrent loop пока не заявлена.
* Evidence path и deterministic execution больше не скрывают route/rotation на host:
  `argmax(First) → take` и `permute/reverse_axis/concat` исполняются в общем
  CUDA Graph. Training runtime всё ещё использует stochastic host sampling как
  внешнюю policy. Даже для stochastic/fixed-route path host больше не
  materialize-ит rotated weight: выбранный route-specific `ModelExecution`
  вычисляет spatial rotation внутри IR/GPU. В deterministic path состояние
  переходит через `ModelExecution::step` D2D; host-копия `next_state` нужна
  только для диагностического `Trace`, не для следующего replay.
  CUDA/IEEE-754/SASS не заверены.
* Обучение действительно меняет router weights (REINFORCE с baseline),
  scalar head (онлайн SGD) и обе базовые матрицы (SPSA с фиксированным
  маршрутом). Это исследовательский стохастический метод, **не обещание
  сходимости**. Результаты нужно интерпретировать как эмпирические.
* Из-за маленького набора обучающих шагов по умолчанию точность может быть
  низкой. Цель лаборатории: выявить, полезна ли маршрутизация и повороты,
  а не объявить, будто 40 арифметических алгоритмов выучены автоматически.

## Требующие отдельного развития

Structural/indexing GPU lowering и translation/memory evidence для `permute`,
`reverse_axis`, `argmax`, `take` и `concat` закрыты на узком профиле
лаборатории. Stateful deterministic и stochastic/fixed-route execution уже
используют named `ModelExecution`; производный rotated tensor больше не
подсовывается с host. Открыты P1: более общий tensor-примитивный каталог,
численная композиция routing/branching и инвариант рекуррентного цикла.
Отдельно стоит добавить ablation:
фиксированный маршрут против обучаемого, один core против двух, без вращений.

## Новая проверка маршрутизации (код добавлен, NVIDIA-приёмка не выполнена)

`tests/gpu_parity.rs` независимо вычисляет индексы R0/R90/R180/R270,
router logits, recurrent step и scalar readout через простые CPU-циклы.
Восемь forced-action сценариев, повторное использование одного GPU execution,
смена длины последовательности, сброс state и восьмикратный tie `ArgMax First`
сопоставляются с реальным единым CUDA Graph; CPU-only smoke-test проверяет,
что oracle различает все восемь маршрутов. Численная проверка использует
ограниченную погрешность FP32, **не** доказательство machine equivalence.
Для полного пункта GPU-приёмки отдельно выполнить `task lab-rotating`;
он включает обучение 128 обновлений, но **не заменяет** parity test.

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
# Fail-closed probe, пока structural/indexing P0 не имеет GPU lowering
```

`artifacts/` — **единственная папка выходных данных лаборатории; не игнорируется Git**.
Evidence строится от одного `architecture_model()` и содержит один набор
архитектурных файлов, а не отдельные сертификаты router/core/head:

```text
artifacts/
├── semantic.ir
├── architecture.lean
├── architecture.dot
├── architecture.png
├── architecture.audit.txt
├── architecture.graph.ll
├── architecture.graph.ptx
├── architecture.translation.plan
├── architecture.translation.report
├── certification.txt
├── identity.txt
├── SHA256SUMS
├── inspect.txt
└── training-report.txt
```

`inspect.semantic.ir` — диагностический экспорт и не входит в certified SHA bundle.

`architecture.png` рендерится VeriNeural/Graphviz именно из сохранённого
`architecture.dot`. Общий Lean генерируется из объединённого Semantic Neural IR,
в котором присутствуют router logits, recurrent transition и terminal readout.
Runtime может исполнять части раздельно для последовательного CUDA Graph replay,
но это не превращает их в три разные архитектуры.

P0-граница `selected_rotated_weight` из общего IR удалена. Evidence-модель
строит восемь вариантов двух параметров только универсальными
`permute + reverse_axis + concat`, получает индекс через `argmax` с явной
политикой `First` и выбирает матрицу через `take`.

GPU/numeric lowering для `permute/reverse_axis/concat/argmax/take` пока
намеренно `unsupported`, поэтому `Model::certify` на этой архитектуре обязан
fail-closed. Это не повод возвращать host-only параметр в Semantic IR. Исполняемый
training runtime пока сохраняет host sampling/materialization до появления
backend lowering; он не объявляется CUDA/FP32-сертифицированным.

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
* Runtime использует три переиспользуемых исполнения, но evidence-модель одна:
  `architecture_model()` содержит `routing_logits`,
  `routed_recurrent_transition` и `terminal_scalar_readout` в одном IR.
* `architecture.lean` kernel-check'ит определения, формы и локальную семантику
  объединённого поддерживаемого графа. Глобальная теорема корректности
  маршрутизации и произвольного recurrent loop пока не заявлена.
* Evidence path больше не скрывает deterministic route/rotation на host:
  `argmax(First) → take` и `permute/reverse_axis/concat` находятся в общем IR.
  Training runtime всё ещё использует stochastic host sampling и materialization
  выбранного rotated binding, потому что structural/indexing GPU lowering пока
  отсутствует. Последовательное состояние передаётся между запусками CUDA Graph;
  полный цикл и его свойство не доказаны. CUDA/IEEE-754/SASS не заверены.
* Обучение действительно меняет router weights (REINFORCE с baseline),
  scalar head (онлайн SGD) и обе базовые матрицы (SPSA с фиксированным
  маршрутом). Это исследовательский стохастический метод, **не обещание
  сходимости**. Результаты нужно интерпретировать как эмпирические.
* Из-за маленького набора обучающих шагов по умолчанию точность может быть
  низкой. Цель лаборатории: выявить, полезна ли маршрутизация и повороты,
  а не объявить, будто 40 арифметических алгоритмов выучены автоматически.

## Требующие отдельного развития

Полная сертификация `rotate` и selection теперь упирается только в GPU/numeric
lowering уже используемых evidence-моделью `permute`, `reverse_axis`, `argmax`,
`take` и `concat`. GPU lowering и его translation evidence остаются отдельной
обязательной работой. Верификация рекуррентного цикла требует инварианта состояния и
доказанной композиции по времени. Отдельно стоит добавить ablation:
фиксированный маршрут против обучаемого, один core против двух, без вращений.

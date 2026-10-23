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
# Полный запуск: inspect, train 128 updates, certify; нужен GPU + Lean

task inspect-lab-rotating
# Только inspect: схема, формы, словарь и три semantic IR без GPU/Lean

task test-lab-rotating
# Разработчик локально запускает CPU-only тесты

task train-lab-rotating -- 128
# Требует NVPTX llc и NVIDIA CUDA Graph; нет тихого CPU fallback

task certify-lab-rotating
# Lean + NVPTX: сертификаты трёх *локальных* вычислительных графов
```

`artifacts/` — **единственная папка выходных данных лаборатории; не игнорируется Git**.
Она создаётся при запуске и должна попадать в коммит вместе с кодом.
Файлы лежат непосредственно в ней, без `gpu/core`, `certified-core` и других
подкаталогов: `inspect.txt`, `core.ir`, `router.ir`, `head.ir`,
`core.graph.ptx`, `router.graph.ptx`, `head.graph.ptx`,
`core.certified.tar`, `router.certified.tar`, `head.certified.tar`,
`core.dot`, `router.dot`, `head.dot`, соответствующие PNG/Lean и отчёты.
Три tar-файла сохраняют неизменённую внутреннюю структуру сертифицирующего
VeriNeural bundle и его SHA256SUMS для последующего аудита; временные
рабочие каталоги автоматически удаляются. Внешний `tar` определён в Nix.

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
* `ModelBuilder` описывает три универсальных вычислительных графа:
  `recurrent_transition`, `routing_logits`, `scalar_readout`.
* Только `h_next ≥ 0` идеальной `Int` модели задан как Lean `ProofGoal`.
  `certify-primitives` может подтвердить локальные доказательства **после
  фактического запуска Lean checker** и LLVM/PTX lowering.
* Дискретный `argmax`/sampling маршрутизатора и поворот базовой матрицы
  выполняются на host как **неверифицированные** преобразования привязанных
  данных. Последовательное состояние передаётся между запусками CUDA Graph;
  полный цикл и его свойство не доказаны. CUDA/IEEE-754/SASS не заверены.
* Обучение действительно меняет router weights (REINFORCE с baseline),
  scalar head (онлайн SGD) и обе базовые матрицы (SPSA с фиксированным
  маршрутом). Это исследовательский стохастический метод, **не обещание
  сходимости**. Результаты нужно интерпретировать как эмпирические.
* Из-за маленького набора обучающих шагов по умолчанию точность может быть
  низкой. Цель лаборатории: выявить, полезна ли маршрутизация и повороты,
  а не объявить, будто 40 арифметических алгоритмов выучены автоматически.

## Требующие отдельного развития

Полная сертификация `rotate` и selection должна опираться на универсальные
IR-примитивы permutation / transpose / gather и selection со строгим GPU + Lean
lowering. Верификация рекуррентного цикла требует инварианта состояния и
доказанной композиции по времени. Отдельно стоит добавить ablation:
фиксированный маршрут против обучаемого, один core против двух, без вращений.

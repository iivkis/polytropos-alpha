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

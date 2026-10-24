# Torus16-Mix 0.1.0

Лаборатория-потребитель публичного API VeriNeural. Источник математики:
исследовательский handoff от 2026-10-25, файл
projects/virtual16_recurrent_operator/operator16.py, режим torus16_mix.
Эта версия переносит **ядро оператора**; полный четырёхслойный EMA4-LM
(Conv9, RMSNorm, SwiGLU, embedding, head), обучение и оценку BPC пока не переносит.
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
проверку named bindings и негативную проверку неподдерживаемого GPU.

~~~sh
task certify-lab-torus16-mix
~~~

Эта команда **сейчас обязана завершаться ошибкой**, а не выпускать фиктивный
сертификат: Pointwise Exp/Div по-прежнему не имеют строгого GPU lowering.
Для BroadcastTo и SliceAxis теперь есть узкий структурный GPU-профиль
[1,1] -> [1,N] и [1,N] SliceAxis(1), а также Concat(1) на [1,N].
Он проходит через общий Semantic IR / CUDA Graph / translation validation;
поддержка других форм остаётся fail-closed. Генерация Lean memory obligations
не означает, что пользователь уже проверил их Lean или GPU. Нет отдельного CUDA, своего IR и скрытого CPU fallback.
Проверка форм и связей в Semantic IR не равна верификации CUDA FP32.
Ни GPU-ускорение, ни реальное обучение, ни сходимость, ни GPU/Lean
сертификация Torus16-Mix здесь не заявлены. Для завершения лаборатории
нужны универсальные backend-поддержки этих операций, далее EMA4-LM и
реальный CUDA Graph training/evaluation, без архитектурных операций в core.

PyTorch-паритет **намеренно не реализован**, как и требовалось.

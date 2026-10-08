> 在 experiment 分支，本文描述兼容的 action-producing DecisionEngine；旧 `poker.agent.policy.Policy` 为其别名。以模型循环为核心的新 Policy 契约见 [ReAct Agent](react/README.md)。

# poker.agent：Policy 接口参考

本文件说明已经实现的 `poker.agent` 包 API。项目需求、模块架构、安装和验收统一以[仓库 README](../../../README.md)为准；研究背景见[Agent 研究笔记](../../../docs/AGENT_RESEARCH.md)。

## 1. 核心入口只有一个

继承 [Policy](policy.py) 后，唯一必须实现的方法是：

```python
def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
    ...
```

`request` 给出本次决策的玩家观察、标识和截止时间；`context` 提供分析工具、随机源和取消控制；返回值是一个行动提议。Agent 运行时负责校验、选择具体动作、经 CLI 提交，以及处理真实确认。

Policy 对象自行持有状态、模型权重、提示词、求解器、缓存或网络客户端。这些内部对象不要求可序列化为 JSON。规则程序可以完全不用工具；网络策略可直接调用自己的模型客户端；自然语言策略可以使用下面的 `PromptPolicy` 适配层。

| 方法 / 属性 | 是否必需 | 已实现的契约 |
|---|---|---|
| `decide(request, context)` | 必需 | 返回 `Decision`；不直接操作 CLI 或 Arena |
| `identity` | 可选覆盖 | 默认名称为类的限定名，版本为 `unspecified`；用于日志 |
| `open(session)` | 可选 | 初始化，接收 `agent_id` 和 `player_name`；成功后运行时才请求 CLI 入座 |
| `observe(event)` | 可选 | 维护状态，接收观察、动作确认或已观察到的完成结果 |
| `close(reason)` | 可选 | 释放资源；在此前运行中的回调返回后串行执行，也可能在初始化失败后执行 |

`open`、`observe`、`decide`、`close` 在同一策略工作线程串行调用，不会同时进入同一个 Policy。构造对象和读取 `identity` 发生在运行时启动侧；有线程归属要求的资源适合在 `open` 中初始化。每个 `AgentRunner` 及其 Policy 只运行一次。

## 2. 为什么选择这个边界

| 备选接口 | 当前选择的理由 |
|---|---|
| 只接受 Markdown policy | 文本适合提示词，但程序、网络和求解器需要保留自己的数据结构；文本属于具体 Policy 的输入 |
| 把 LLM messages 作为核心输入输出 | 会要求非语言模型策略模拟聊天；消息转换集中在 `LanguageModel` 适配层 |
| 强制所有 Policy 返回完整动作分布 | 确定策略也有用途，无限注金额空间很大；允许直接动作，混合策略按需返回有限候选分布 |
| 所有策略实现异步 `step → tool → resume` 协议 | 当前一个同步 `decide` 可以内部多次调用工具；运行时另有 CLI 线程继续收信息，策略作者无需实现外部状态机 |

同步入口有明确代价：取消需要策略配合，Python 线程不能硬杀正在执行的模型调用或本地计算。需要强隔离时，应保持每个 Agent 独立进程，并让外部进程管理器管理退出。

## 3. 观察、输出和金额

[models.py](models.py) 中的 `DecisionRequest` 包含：

- `decision_id`：本次请求标识，用于关联提交与确认。
- `observation.view`：不可变 `PlayerView`，含本人底牌、公共牌、公开玩家、合法动作、规则配置和当前手公开行动历史。
- `observation.received_at`：运行时接收该观察时的单调时钟时间。
- `deadline`：单调时钟截止时间；外部 I/O 应使用 `context.remaining_seconds()` 作为剩余预算。

所有 Arena 信息都来自 CLI 的玩家安全视图。Policy 和分析工具没有收到 `Table`、SQLite、牌堆、烧牌或对手未公开底牌。当前手的 `action_history` 来自服务端记录，包括盲注及公开动作；`history_complete=False` 表示历史不能当作完整行动线。旧协议或旧快照缺失字段时保留这种不完整状态。历史只覆盖当前手，观察也不保证捕获每个中间快照。

`Decision.choice` 有两种形式：

```python
Decision(PlayerAction(ActionKind.CHECK))

Decision(ActionDistribution((
    WeightedAction(PlayerAction(ActionKind.CALL), 0.7),
    WeightedAction(PlayerAction(ActionKind.RAISE_TO, 100), 0.3),
)))
```

第二个例子只有在当前服务器同时提供这些动作、且允许加至 100 时才有效。`bet_to` / `raise_to` 的 `to` 是**本街累计投入总额**，不表示新增支付额；其他动作不带金额。

运行时会验证每个候选，包括概率为零的候选，再采样。概率必须有限、非负且总和为 1，允许 `1e-9` 的求和绝对误差；重复的 `(kind, to)`、空分布、NaN、bool 概率、非整数或越界金额会被拒绝。框架不会归一化、截断金额或改成另一动作。Policy 随机源和运行时动作采样使用从配置种子派生的两个独立 RNG。

`Decision.annotations` 可附带 JSON 诊断信息，默认不写日志，也不送 Arena；它不限制 Policy 内部状态的类型。

## 4. 观察事件与执行顺序

`observe` 接收以下 `PolicyEvent`：

| 事件 | 含义 |
|---|---|
| `ObservationChanged` | CLI 交付了新的玩家视图，可更新对手模型或场景缓存 |
| `ActionFeedback` | 对应 `decision_id` 的动作确认或拒绝；`confirmed` 与 `error_code` 明确结果 |
| `HandCompleted` | CLI 实际交付了新的完成结果；内容为真实分池与退款，框架不推测收益 |

初始化完成后，运行时先交付积累的事件，再对当前轮到本人的视图调用 `decide`。同一个视图 token 不重复启动决策。首次观察中已存在的旧 `result` 作为背景信息保留，不计入本会话新观察的完成手数，也不额外生成 `HandCompleted`。

决策结束后有两层过期保护：CLI 先主动刷新并核对牌桌、玩家、手号和 revision；提交时再携带 `expected_revision`，服务端在单消费者内原子检查后才执行动作。过期动作不会执行；新的观察可以触发重新决策。

动作排入队列不代表成功。只有对应服务端成功响应才递增 `confirmed_actions`。连接断开或确认超时可能意味着执行结果未知；框架不会重发这个动作。

## 5. 分析工具

`context.tools.describe()` 返回允许调用的 `ToolSpec`；`context.tools.call(name, arguments)` 返回 `ToolResult`。默认工具是：

| 名称 | 输入 | 返回 |
|---|---|---|
| `observation` | `{}` | 本次决策绑定的玩家安全视图 |
| `legal_actions` | `{}` | 服务器给出的动作及支付额 / 本街金额范围 |
| `public_history` | `{}` | 当前手号、`complete` 标记和公开动作序列 |

工具始终绑定本次决策的不可变观察，不会在调用中偷偷换成更新的局面。更新局面由 CLI 交付，并使旧决策取消。工具用于分析和查询，最终下注由 `Decision` 统一交给运行时。

自定义工具继承 `Tool`，实现 `spec` 和 `invoke(arguments, context)`。输入为 `JsonObject`，输出为 `JsonValue`，注册时检查 JSON Schema。工具可以在内部使用任意模型或数值对象，再将边界结果转换为 JSON。

```python
from poker.agent.tools import Tool, ToolContext, ToolSpec
from shared_logging import JsonObject, JsonValue

class PotTool(Tool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("pot_chips", "当前观察的底池筹码数。",
                        {"type": "object", "additionalProperties": False},
                        {"type": "integer"})

    def invoke(self, arguments: JsonObject, context: ToolContext) -> JsonValue:
        context.control.check()
        return context.observation.view.pot_total

def make_tools(config: JsonObject) -> list[Tool]:
    return [PotTool()]
```

工具工厂用 `--tool-factory your_module:make_tools` 加载，其结果追加到标准工具；名称必须唯一。运行时按每次决策计数，未知工具和错误参数也消耗一次尝试预算。`ToolResult.ok=False` 携带 `unknown_tool`、`invalid_arguments`、`invalid_output` 或 `tool_failed`；取消、deadline 和预算耗尽通过对应 `PolicyError` 异常传播。普通工具异常不会将原始异常文本、参数或结果写入工具日志。

## 6. 最小策略工厂

下面是可放进可导入模块 `my_policy.py` 的完整示例，仅示范接入，不声明竞技水平。工具调用是可选的，此处用它演示失败处理。

```python
from poker.agent.context import DecisionContext
from poker.agent.models import Decision, DecisionRequest, PolicyError
from poker.agent.policy import Policy
from poker.domain.models import PlayerAction
from poker.domain.types import ActionKind
from shared_logging import JsonObject

class ExamplePolicy(Policy):
    def decide(self, request: DecisionRequest, context: DecisionContext) -> Decision:
        context.check_cancelled()
        if not context.tools.call("public_history", {}).ok:
            raise PolicyError("Public history unavailable")
        option = request.observation.view.me.legal_actions[0]
        sized = option.kind in (ActionKind.BET_TO, ActionKind.RAISE_TO)
        return Decision(PlayerAction(option.kind, option.min_to if sized else None))

def make_policy(config: JsonObject) -> Policy:
    return ExamplePolicy()
```

`--policy my_policy:make_policy` 调用工厂并检查其返回 `Policy` 实例。`--config file.json` 将一个 JSON 对象交给策略工厂和所有额外工具工厂；工厂可据此读取 Markdown、加载网络或配置远程服务。`module:callable` 加载的是受信任的 Python 代码，没有 OS sandbox；进程和接口隔离不等于限制插件访问文件或网络。

## 7. 自然语言策略的组合方式

[language.py](language.py) 提供 `PromptPolicy(instructions, model, max_rounds=8)` 和抽象 `LanguageModel.complete(ModelRequest) -> ModelTurn`。下面的组合函数可以与具体模型适配器一起放入策略模块：

```python
from pathlib import Path
from poker.agent.language import LanguageModel, PromptPolicy
from poker.agent.policy import Policy

def with_markdown(path: Path, model: LanguageModel) -> Policy:
    return PromptPolicy(path.read_text(encoding="utf-8"), model, max_rounds=6)
```

策略工厂负责构造自己的 `LanguageModel` 实现，再调用此函数。`ModelRequest` 提供原始 `instructions`、观察 JSON、可用工具 schema、此前工具交换和 `timeout_seconds`。一次模型结果须为以下两种之一：

```python
ModelTurn(tool_calls=(ModelToolCall("c1", "legal_actions", {}),))
ModelTurn(decision=Decision(PlayerAction(ActionKind.CHECK)))
```

`PromptPolicy` 执行分析工具，把 `ModelToolExchange(call, result)` 交回下一次模型调用，直到收到决策或耗尽 round budget。call ID 在一次决策内必须非空且唯一；下一次决策重置这组交换记录。模型适配器如使用“提交动作”的函数调用格式，应将其转换为 `ModelTurn(decision=...)`，不能直接发送 Arena 命令。

初始 A01 验收使用 fake `LanguageModel`。后续六种 Policy 实验增加了 [CodexLanguageModel](naive_language.py)，通过本机已登录的 Codex CLI 进行真实模型推理，已在六人牌桌完成自然语言到工具再到动作的链路。它复用 Codex 自己的认证，不读取或复制认证令牌。`PromptPolicy.identity.version` 是提示文本的短哈希；具体模型名、后端、usage 和每次调用另有记录，费用未知时为 null。

## 8. 启动与退出

在仓库根目录安装后，先启动服务端，再分别在两个终端启动 Agent。只有一个参与者使用 `--auto-start`；它负责在人数满足时请求新手，这项职责独立于 Policy。

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\poker-server.exe --port 8765 --db .data/agent-table.sqlite3
```

```powershell
.\.venv\Scripts\python.exe -m poker.agent --port 8765 --name Agent-A --policy poker.agent.examples:first_legal --auto-start --min-players 2 --max-hands 3 --seed 1
```

```powershell
.\.venv\Scripts\python.exe -m poker.agent --port 8765 --name Agent-B --policy poker.agent.examples:tool_first --max-hands 3 --seed 2
```

安装后的 `poker-agent.exe` 与 `python -m poker.agent` 使用同一入口。已有 `first_legal`、`uniform`、`tool_first` 三个示例工厂，分别展示直接动作、有限分布和工具调用。示例策略的行为只用于工程接入检查。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--table` | 服务端默认桌 | 在同一游戏端口选择启动时配置的桌号；其他运行 / Policy 契约不变 |
| `--decision-timeout` | 30 秒 | 单次 `decide` 截止时间 |
| `--callback-timeout` | 5 秒 | `open` / `observe` 回调 watchdog |
| `--submission-timeout` | 10 秒 | 等待动作确认的上限 |
| `--shutdown-timeout` | 1 秒 | 等待策略 worker 关闭的上限 |
| `--max-tool-calls` | 8 | 每次决策允许的工具调用尝试数 |
| `--max-actions` / `--max-hands` | 不限制 | 已确认动作数 / 本会话新观察到的完成手数 |
| `--session-timeout` | 不限制 | 整个运行时间上限，包含初始化 |

达到正常的动作数或手数上限时，运行时停止新增决策，处理已排队的 hooks 并等待已有动作确认；后续新观察不会继续追加事件。显式停止、会话时间上限和 watchdog 属于有界中断路径，不能保证排空全部回调。它们会取消当前决策并拒绝迟到结果；阻塞 I/O 仍需实现方使用剩余时间作为 timeout。

`close` 不会与尚未返回的 Policy 回调并发执行。若策略或 CLI 没有在关闭期限内停完，`RunResult` 如实返回 `policy_closed` / `cli_closed` 和错误，不能将其视为成功。未确认动作会明确标记执行可能未知。独立进程是推荐运行形态，但框架自身没有提供任意线程的强制终止。

进程退出时输出 JSON，包括停止原因、确认动作数、观察手数、决策数、工具调用数、关闭状态和 `ok`。`ok` 要求正常停止、两侧均完成关闭且没有错误；失败返回非零退出码。日志沿用项目统一的 `shared_logging`。

## 9. 六种最小实现和两层适配

[naive_policies.py](naive_policies.py) 与 [naive_language.py](naive_language.py) 提供六个可实际运行的工厂：

| 工厂 | 决策逻辑放在哪里 | 实际验证的机制 |
|---|---|---|
| `poker.agent.naive_policies:rules` | RulesPolicy 的条件分支 | 根据街道、跟注价格和筹码选择动作 |
| `poker.agent.naive_policies:mixed` | MixedPolicy 的候选权重 | 返回真实有限分布，由运行时采样 |
| `poker.agent.naive_policies:lookup` | LookupPolicy 的场景表与缓存 | 表命中、缓存复用、可替换表项 |
| `poker.agent.naive_policies:stateful` | StatefulPolicy 的实例状态 | 收到确认与结果后更新记忆，记忆参与后续决策 |
| `poker.agent.naive_policies:solver` | SolverPolicy 的采样和候选计算 | 从未知牌空间采样，比较候选动作的近似 EV |
| `poker.agent.naive_language:natural_language` | Python 中的自然语言指令、PromptPolicy 和模型适配器 | 模型先请求分析工具，收到结果后给出最终动作 |

`decide` 只有一个入口，并不把上述决策逻辑搬进 AgentRunner。Policy 可以在一次调用里做多步计算、查询或模型交互。AgentRunner 对这些内部机制保持一致的调度和确认契约。

有两个各自独立的适配边界。`CodexLanguageModel` 把模型的结构化输出转换为 `ModelTurn`；`PokerCliAgentAdapter` 把运行时提交转换为原 CLI 能处理的操作。模型适配器没有 Arena 连接，CLI 适配器也不判断牌局策略。

六人驱动 [verify_six_agents.py](../../../scripts/verify_six_agents.py) 为每个策略启动独立进程，并按决策 ID 将机制日志、提交、服务端实际动作和确认核对起来。它同时核对玩家视图、公开历史、6000 筹码和 52 张牌守恒、七个真实运行 PID、最终 SQLite 和关闭状态。失败也保留回执，不会自动换用另一种策略。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_six_agents.py `
  --output-dir .data/verification/six-policy-new `
  --hands 2 --language-backend codex --language-max-calls 32
```

`--language-backend stub` 是明确标注的离线协议桩，用于快速回归，不计真实自然语言推理。旧版 `codex` 后端的后续实验默认显式使用 `gpt-6-luna / low`，不读取全局模型配置；可以通过 `--language-model` 覆盖。每次调用在空临时目录运行，禁用环境工具，并要求完整的模型完成事件和合法结构化答案。调用失败不切换到 stub，也不回退到更贵的模型。CLI 接入依据 [Codex 非交互模式官方文档](https://learn.chatgpt.com/docs/non-interactive-mode)；API 回执记录真实 token，未取得实际费用时不推算账单。

默认单轮最多 32 次模型尝试，失败也计入预算；夜间监督器默认总上限 64 次。每个自然语言决策最多 3 轮模型交互，六人驱动默认决策期限 180 秒。这些约束不等同于 token 或金额硬上限：旧版尚无单独的输入上下文 / token 硬限制，订阅额度扣减和费用未知时继续记录 `null`。默认模型与预算由测试覆盖，公开验收摘要见 [VALIDATION.json](../../../docs/VALIDATION.json)。2026-10-08 已结束的夜间批次仍按原记录使用 `gpt-6.1-sol / low`，没有被追溯修改。

NaiveLanguagePolicy 将当前 DecisionControl 限定在模型调用作用域内；CodexLanguageModel 以短间隔检查取消和剩余时间。停止决策时会终止并回收它创建的模型进程，子进程启动后的证据写入异常也走同一清理路径。一次调用的轮询不增加模型尝试数；尝试记录在进程启动前落盘，失败和未完成的尝试都占预算。

六人实验使用公开的 `verification_continuation=true` 限制候选为过牌、跟注和小额下注，以便保持六人参与；各策略仍按自己的机制选动作。求解器只做少量均匀未知牌采样和局部收益近似，未实现 CFR/GTO。牌序只由服务端控制，不能作为策略的输入。固定牌序与随机牌序分别记录，均用于工程验证。

[run_agent_overnight.py](../../../scripts/run_agent_overnight.py) 可以在固定截止时间和调用预算内重复运行六人驱动，轮换座位，并分开统计真实模型与 stub 场景。每轮用新桌和新进程，保留 manifest、stdout、成功或失败回执以及模型尝试。夜间任务是否完成，应读取实际报告和证据，不能由计划轮数推算。

每轮会比较运行前后的相关源码哈希；期间有变化时保留实际对局与调用统计，但判验收失败。监督器本身被替换后，仍在执行旧版本的进程会停止启动后续轮次。正常失败、超时和停止路径清理受控子进程；若 driver 被外部硬杀并跳过自身清理，当前实现没有 Windows Job Object，不能保证回收已经脱离父进程的所有子孙进程。

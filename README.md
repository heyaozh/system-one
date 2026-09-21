<p align="center">
  <img src="docs/logo.svg" alt="jev" width="120"/>
</p>

# jev

**Typed, calibrated decisions for Rust.**
A backend-agnostic client for *System One* models — models that answer a fixed set of questions with calibrated probabilities instead of generating text. The first backend is [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev); the same interface serves a mock, a recording, or **your own fine-tuned model**.

[English](#english) · [中文](#中文)

---

## English

### Why

Every "smart if-statement" has the same shape:

```
state (anything serialisable) + questions (bounded answer spaces) → answers with probabilities
```

Ticket routing, a pre-trade risk gate, an NPC's reflex layer, labelling a million headlines — same shape. `jev` fixes that shape once and stacks the tools you need on top:

| Layer | What you get |
|---|---|
| **Types** | `Noul` (P(yes)), `Choice<E>` (distribution over an enum), `Score` (distribution over an ordered scale). Full distributions, not just arg-max. |
| **Derive** | `#[derive(JevChoice)]` on an enum, `#[derive(JevQuestions)]` on a struct → schema generated, answers parsed back. Compile-time checks (≤255 options, 2–10 levels, field types). |
| **Decision helpers** | `decide(cost)` minimises expected loss with a cost matrix; `entropy()` for active learning; `sample()` for stochastic agents. |
| **Backends** | `JevHttp` (official API), `Mock`, `Replay`, `LocalLogprob` (your model via vLLM / llama.cpp), `Shadow` (A/B two backends). Or implement `DecisionBackend` yourself. |
| **Engine** | Content-hash cache, concurrent `ask_many`, JSONL recording (Parquet with `--features parquet`). |
| **Calibration** | Attach real outcomes to recordings; get Brier, log-loss, ECE and a reliability table per question. |

The point is the **loop**, not the wrapper: *decide → record → attach outcomes → measure calibration → train a small model → swap the backend*. Business code never changes.

![architecture](docs/architecture.svg)

### Quick start

```toml
[dependencies]
jev = { path = "jev" }        # or from crates.io once published
tokio = { version = "1", features = ["full"] }
```

```rust
use jev::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Department {
    /// Invoices, payouts, charges, refunds.
    Billing,
    /// Bugs, integrations, API errors.
    Technical,
    Sales,
    Spam,
}

#[derive(Debug, JevQuestions)]
struct Triage {
    #[jev("Which department should handle this ticket?")]
    department: Choice<Department>,
    #[jev("Is the customer explicitly asking for money back?")]
    wants_refund: Noul,
    #[jev("How urgent is this?", levels = ["Can wait a week", "Within a day", "Right now"])]
    urgency: Score,
}

#[tokio::main]
async fn main() -> jev::Result<()> {
    // JEV_API_KEY=... in the environment; see examples for a Mock fallback.
    let engine = Engine::new(JevHttp::from_env().expect("JEV_API_KEY"))
        .with_recorder(Recorder::open("runs/triage.jsonl")?);

    let t: Triage = engine.ask("Hi, my payouts have failed for 3 days, I want the fees back.").await?;

    // Cost-aware: misrouting a real customer to Spam costs 5, other mistakes 1.
    let route = t.department.decide(|truth, action| match (truth, action) {
        (a, b) if a == b => 0.0,
        (_, Department::Spam) => 5.0,
        _ => 1.0,
    });
    // A missed refund request costs 4, a false flag costs 1 → threshold 0.2.
    let flag = t.wants_refund.decide(1.0, 4.0);

    println!("{route:?} refund={flag} urgency={}", t.urgency.argmax_label());
    Ok(())
}
```

Three questions, one call, ~100 ms, and the routing rule is an explicit cost matrix rather than a threshold someone guessed.

#### Try one question first

Before writing any struct, try a question from the shell — `examples/ask` builds the schema at runtime:

```bash
export JEV_API_KEY=...   # without it you get a uniform Mock answer, and a note saying so

cargo run --example ask -- "Is this customer asking for a refund?" \
  "my payouts failed for 3 days, I want the fees back"

cargo run --example ask -- --choice billing,technical,sales,spam \
  "Which department should handle this?" "getting a 500 from /v1/orders"

cargo run --example ask -- --levels "Can wait a week,Within a day,Right now" \
  "How urgent is this?" "my payouts have failed for 3 days"
```

It prints the full distribution, not just the arg-max — which is the whole point.

### The three primitives

| Rust type | API type | Answer |
|---|---|---|
| `Noul` | `noul` | `p: f64` — probability of *yes* |
| `Choice<E>` | `choice` | `probabilities: Vec<(E, f64)>`, `chosen`, `confidence` |
| `Score` | `score` | `probabilities: Vec<f64>` over levels, `value` (probability-weighted index), `legend`, `confidence` |

`#[jev(...)]` field attributes: positional `"instructions"` (or a `///` doc comment), `levels = [...]` for `Score`, optional `yes = "…", no = "…"` criteria for `Noul`, `name = "…"` to override the wire name. Enum variants take `#[jev(key = "…", desc = "…")]` or a doc comment.

### Backends

```rust
// Official API (reads JEV_API_KEY, JEV_ENDPOINT, JEV_MODEL)
let b = JevHttp::from_env().unwrap();

// Deterministic mock for tests — a rule, or uniform
let b = Mock::with_rule(|state, question, spec| /* Option<RawAnswer> */ None);

// Replay a recording: no network, no cost, reproducible
let b = Replay::from_jsonl("runs/2026-09.jsonl")?.with_fallback(JevHttp::from_env().unwrap());

// Your own model behind an OpenAI-compatible server (vLLM, llama.cpp, LM Studio, Ollama)
let b = LocalLogprob::new("http://localhost:8000/v1", "my-finetune-v3");

// A/B: answer from primary, record both
let b = Shadow::new(JevHttp::from_env().unwrap(), LocalLogprob::new(url, model))
    .with_recorder(Recorder::open("runs/shadow.jsonl")?);
```

All of them implement one trait:

```rust
#[async_trait]
pub trait DecisionBackend: Send + Sync {
    fn id(&self) -> String;
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers>;
}
```

so `Engine<Box<dyn DecisionBackend>>` lets you pick the backend at runtime and keep the rest of the code identical.

**`LocalLogprob` is how your own model plugs in.** It renders each question as a labelled multiple-choice prompt, asks for one token, and reads the log-probabilities of the option labels — no generation, no parsing, a distribution every time. Fine-tune any open model with a proper scoring rule (cross-entropy on real outcomes) and it becomes a domain-specific System One model behind the same interface.

### The data loop

```rust
// 1. label a stream, 8 in flight, everything recorded
let engine = Engine::new(backend).with_recorder(Recorder::open("runs/news.jsonl")?).with_concurrency(8);
let results = engine.ask_many::<NewsFeatures, _, _>(headlines).await;

// 2. later, when reality has happened: join outcomes by state hash
let mut outcomes = HashMap::new();
outcomes.insert(jev::hash::hash_json(&serde_json::to_value(&h)?), json!({ "is_surprise": true }));
jev::record::attach_outcomes("runs/news.jsonl", &outcomes)?;

// 3. did 0.85 mean 85 %?
let records = jev::record::read_records("runs/news.jsonl")?;
println!("{}", CalibrationReport::from_records(&records, "is_surprise", 10).render());
```

```
n=300  brier=0.1228  log_loss=0.4111  ece=0.0890  acc@0.5=0.857  base_rate=0.180
  bin          count   mean_p   frac_pos   |gap|
  [0.10,0.20)    249   0.150    0.092   0.058  ##
  [0.80,0.90)     51   0.850    0.608   0.242  ############
```

Those rows, with outcomes attached, are also a labelled training set. `--features parquet` exports them for pandas / polars.

### Examples

| Example | Shows |
|---|---|
| `ask` | One question, from the command line — no structs, schema built at runtime. |
| `ticket_routing` | The canonical smart if-statement; cost-matrix routing. |
| `trade_risk_gate` | Structured state (a proposed order + mandate) → pre-trade compliance gate with hard overrides. Generic trading, no exchange specifics. |
| `chat_reflex` | Reflex layer for a conversational character: response tier, animation clip (sampled, not arg-max), memory need, jailbreak risk. |
| `batch_and_calibrate` | `ask_many` → JSONL → `attach_outcomes` → calibration report (→ Parquet). |
| `shadow_local_model` | Official API as primary, your local model as shadow, both recorded. |

```bash
cargo run --example ask -- "Is this a refund request?" "my payouts failed, I want the fees back"
cargo run --example ticket_routing               # Mock backend, runs offline
JEV_API_KEY=... cargo run --example ticket_routing  # real API
cargo run --example batch_and_calibrate --features parquet
cargo test
```

Every example falls back to a rule-based `Mock` when `JEV_API_KEY` is unset, so they run anywhere.

### Feature flags

* `http` (default) — `JevHttp` and `LocalLogprob` (pulls in `reqwest` with rustls).
* `derive` (default) — the derive macros.
* `parquet` — `jev::record::export_parquet`.

### Limits worth knowing

* The official API allows ≤255 `choice` options and 2–10 `score` levels; the derive macro enforces this at compile time.
* `LocalLogprob` supports ≤26 options per `choice` (one letter per option) in this version.
* Jev returns probabilities but no rationale; a `Shadow` with an LLM judge, sampled occasionally, is the usual audit pattern.
* Calibration is a property of a distribution: a backend calibrated on its training data can be confidently wrong on yours. Measure before you trust — that is what the recorder is for.

### Status

`0.1.0` — API shapes follow the public TypeSafe docs as of September 2026. Not affiliated with TypeSafe AI. Licensed MIT OR Apache-2.0.

---

## 中文

### 为什么做这个

所有「聪明的 if 语句」都是同一个形状：

```
state（任何可序列化的东西）+ questions（有界的答案空间）→ 带概率的答案
```

工单分流、交易前风控闸门、NPC 的反射层、给一百万条新闻标题打标——都是这个形状。`jev` 把这个形状固定下来，然后在上面堆你需要的工具：

| 层 | 提供什么 |
|---|---|
| **类型** | `Noul`（P(是)）、`Choice<E>`（枚举上的分布）、`Score`（有序等级上的分布）。保留完整分布，不只是 argmax。 |
| **派生宏** | 枚举上 `#[derive(JevChoice)]`，结构体上 `#[derive(JevQuestions)]` → schema 自动生成，答案自动解析。编译期检查（≤255 个选项、2–10 个等级、字段类型）。 |
| **决策工具** | `decide(cost)` 用成本矩阵最小化期望损失；`entropy()` 用于主动学习；`sample()` 用于随机 agent。 |
| **Backend** | `JevHttp`（官方 API）、`Mock`、`Replay`、`LocalLogprob`（你自己的模型，经 vLLM / llama.cpp）、`Shadow`（两个 backend 做 A/B）。也可以自己实现 `DecisionBackend`。 |
| **Engine** | 内容哈希缓存、并发 `ask_many`、JSONL 记录（`--features parquet` 导出 Parquet）。 |
| **校准** | 把真实结果 join 回记录；对每个问题给出 Brier、log-loss、ECE 和 reliability 表。 |

重点是**闭环**而不是 wrapper：*决策 → 记录 → 挂上真实结果 → 量校准 → 训一个小模型 → 换 backend*。业务代码始终不变。

![architecture](docs/architecture.svg)

### 快速开始

```toml
[dependencies]
jev = { path = "jev" }        # 发布到 crates.io 后改为版本号
tokio = { version = "1", features = ["full"] }
```

```rust
use jev::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug, JevChoice)]
enum Department {
    /// 账单、打款、扣费、退款。
    Billing,
    /// Bug、集成、API 报错。
    Technical,
    Sales,
    Spam,
}

#[derive(Debug, JevQuestions)]
struct Triage {
    #[jev("这张工单应该由哪个部门处理？")]
    department: Choice<Department>,
    #[jev("客户是否明确要求退钱？")]
    wants_refund: Noul,
    #[jev("有多紧急？", levels = ["可以等一周", "一天内", "马上"])]
    urgency: Score,
}

#[tokio::main]
async fn main() -> jev::Result<()> {
    // 环境变量 JEV_API_KEY=...；没有 key 时见 examples 里的 Mock 回退
    let engine = Engine::new(JevHttp::from_env().expect("JEV_API_KEY"))
        .with_recorder(Recorder::open("runs/triage.jsonl")?);

    let t: Triage = engine.ask("你好，我的打款失败三天了，手续费要退给我。").await?;

    // 成本感知：把真实客户错分到 Spam 代价 5，其它错分代价 1
    let route = t.department.decide(|truth, action| match (truth, action) {
        (a, b) if a == b => 0.0,
        (_, Department::Spam) => 5.0,
        _ => 1.0,
    });
    // 漏掉退款请求代价 4，误报代价 1 → 阈值 0.2
    let flag = t.wants_refund.decide(1.0, 4.0);

    println!("{route:?} refund={flag} urgency={}", t.urgency.argmax_label());
    Ok(())
}
```

三个问题、一次调用、约 100 毫秒；路由规则是显式的成本矩阵，而不是拍脑袋的阈值。

#### 先用一个问题试试

写结构体之前，先在命令行上试一个问题——`examples/ask` 在运行时拼 schema：

```bash
export JEV_API_KEY=...   # 没设就用 uniform Mock 回答，并会打印一行提示

cargo run --example ask -- "这位客户是在要求退款吗？" \
  "我的打款三天没到，手续费也想退"

cargo run --example ask -- --choice billing,technical,sales,spam \
  "应该由哪个部门处理？" "调 /v1/orders 一直 500"

cargo run --example ask -- --levels "可以等一周,一天之内,马上" \
  "这有多紧急？" "我的打款已经三天没到了"
```

打印的是完整分布，不只是 argmax——这正是重点。

### 三个原语

| Rust 类型 | API 类型 | 答案 |
|---|---|---|
| `Noul` | `noul` | `p: f64`——「是」的概率 |
| `Choice<E>` | `choice` | `probabilities: Vec<(E, f64)>`、`chosen`、`confidence` |
| `Score` | `score` | 各等级的 `probabilities: Vec<f64>`、`value`（概率加权的等级索引）、`legend`、`confidence` |

`#[jev(...)]` 字段属性：位置参数 `"问题文本"`（或 `///` 文档注释）；`Score` 需要 `levels = [...]`；`Noul` 可选 `yes = "…", no = "…"` 判据；`name = "…"` 覆盖线上字段名。枚举变体用 `#[jev(key = "…", desc = "…")]` 或文档注释。

### Backend

```rust
// 官方 API（读取 JEV_API_KEY、JEV_ENDPOINT、JEV_MODEL）
let b = JevHttp::from_env().unwrap();

// 测试用的确定性 mock——规则或均匀分布
let b = Mock::with_rule(|state, question, spec| /* Option<RawAnswer> */ None);

// 回放记录：不走网络、零成本、可复现
let b = Replay::from_jsonl("runs/2026-09.jsonl")?.with_fallback(JevHttp::from_env().unwrap());

// 你自己的模型，放在 OpenAI 兼容的服务后面（vLLM、llama.cpp、LM Studio、Ollama）
let b = LocalLogprob::new("http://localhost:8000/v1", "my-finetune-v3");

// A/B：主 backend 出答案，两者都记录
let b = Shadow::new(JevHttp::from_env().unwrap(), LocalLogprob::new(url, model))
    .with_recorder(Recorder::open("runs/shadow.jsonl")?);
```

它们都实现同一个 trait：

```rust
#[async_trait]
pub trait DecisionBackend: Send + Sync {
    fn id(&self) -> String;
    async fn decide(&self, state: &serde_json::Value, schema: &QuestionSchema) -> Result<RawAnswers>;
}
```

因此 `Engine<Box<dyn DecisionBackend>>` 可以在运行时选择 backend，其余代码完全一样。

**`LocalLogprob` 就是你自己模型的接入口。** 它把每个问题渲染成带字母标签的多选题，只要模型输出一个 token，然后读取各选项标签的 log-probability——不生成文本、不解析、每次都是一个分布。用 proper scoring rule（对真实结果做交叉熵）微调任何开源模型，它就成了同一接口后面的领域专属 System One 模型。

### 数据闭环

```rust
// 1. 给一个流打标，8 个并发，全部记录
let engine = Engine::new(backend).with_recorder(Recorder::open("runs/news.jsonl")?).with_concurrency(8);
let results = engine.ask_many::<NewsFeatures, _, _>(headlines).await;

// 2. 之后，现实发生了：按状态哈希把结果 join 回去
let mut outcomes = HashMap::new();
outcomes.insert(jev::hash::hash_json(&serde_json::to_value(&h)?), json!({ "is_surprise": true }));
jev::record::attach_outcomes("runs/news.jsonl", &outcomes)?;

// 3. 0.85 真的是 85% 吗？
let records = jev::record::read_records("runs/news.jsonl")?;
println!("{}", CalibrationReport::from_records(&records, "is_surprise", 10).render());
```

带上结果的这些行，同时就是一份有标签的训练集。`--features parquet` 可导出给 pandas / polars。

### 示例

| 示例 | 展示 |
|---|---|
| `ask` | 命令行上问一个问题——不写结构体，schema 运行时拼。 |
| `ticket_routing` | 最典型的「聪明 if」；成本矩阵路由。 |
| `trade_risk_gate` | 结构化状态（拟下单 + 授权说明）→ 交易前合规闸门 + 硬性覆盖。通用交易场景，不涉及具体交易所。 |
| `chat_reflex` | 对话角色的反射层：回应层级、动画片段（采样而非 argmax）、是否需要记忆、越狱风险。 |
| `batch_and_calibrate` | `ask_many` → JSONL → `attach_outcomes` → 校准报告（→ Parquet）。 |
| `shadow_local_model` | 官方 API 做主、本地模型做影子，两者都记录。 |

```bash
cargo run --example ask -- "这是在要求退款吗？" "我的打款三天没到，手续费也想退"
cargo run --example ticket_routing                 # Mock backend，离线可跑
JEV_API_KEY=... cargo run --example ticket_routing  # 真实 API
cargo run --example batch_and_calibrate --features parquet
cargo test
```

没有 `JEV_API_KEY` 时所有示例都回退到基于规则的 `Mock`，在任何机器上都能跑。

### Feature flags

* `http`（默认）——`JevHttp` 和 `LocalLogprob`（引入 rustls 版 `reqwest`）。
* `derive`（默认）——派生宏。
* `parquet`——`jev::record::export_parquet`。

### 值得知道的限制

* 官方 API：`choice` ≤255 个选项，`score` 2–10 个等级；派生宏在编译期强制。
* 当前版本 `LocalLogprob` 每个 `choice` ≤26 个选项（一个字母一个选项）。
* Jev 只给概率不给理由；常用的审计模式是挂一个 LLM 裁判做 `Shadow`，偶尔抽样。
* 校准是相对某个分布而言的：在自己训练数据上校准好的 backend，到了你的数据上可能自信地错。先量再信——recorder 就是干这个的。

### 状态

`0.1.0`——API 形状依据 2026 年 9 月的 TypeSafe 公开文档。与 TypeSafe AI 无关联。许可证 MIT OR Apache-2.0。

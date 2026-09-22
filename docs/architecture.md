# Architecture / 架构

![architecture](architecture.svg)

## Flow (Mermaid)

```mermaid
flowchart TB
    subgraph YOU["1 · Your code (per project)"]
        T["#[derive(AsChoice)] enum<br/>#[derive(AsQuestions)] struct"]
        S["state: any Serialize<br/>(string · struct · chat history · book snapshot)"]
        D["typed answers<br/>Noul · Choice&lt;E&gt; · Score<br/>decide(cost) · entropy() · sample()"]
    end

    subgraph ENGINE["2 · Engine (shared)"]
        SCH["schema()<br/>types → API JSON"]
        CACHE["cache<br/>(state, schema) hash"]
        MANY["ask_many · ask_many_raw<br/>N in flight, ordered"]
        RAW["from_raw()<br/>JSON → typed"]
        REC["Recorder<br/>runs/*.jsonl · Parquet"]
    end

    subgraph BE["3 · DecisionBackend — decide(state, schema) → RawAnswers"]
        HTTP["JevHttp<br/>api.typesafe.ai"]
        LOCAL["LocalLogprob<br/>your model via vLLM / llama.cpp"]
        REPLAY["Replay<br/>from a recording"]
        MOCK["Mock<br/>rule / uniform"]
        SHADOW["Shadow(primary, shadow)<br/>return primary, record both"]
    end

    subgraph LOOP["4 · The loop"]
        OUT["attach_outcomes()<br/>what actually happened"]
        CAL["CalibrationReport<br/>Brier · log-loss · ECE"]
        TRAIN["Training set<br/>rows + outcomes"]
        SWAP["Swap the backend<br/>business code unchanged"]
    end

    T --> SCH --> CACHE --> MANY --> HTTP & LOCAL & REPLAY & MOCK & SHADOW
    S --> CACHE
    HTTP & LOCAL & REPLAY & MOCK & SHADOW --> RAW --> D
    RAW --> REC --> OUT --> CAL --> TRAIN --> SWAP
    TRAIN -. "fine-tuned model becomes a backend" .-> LOCAL
```

## Layers / 分层

| Layer | What it owns | 你写的 / 共享的 |
|---|---|---|
| Types | `Noul`, `Choice<E>`, `Score`; derive macros | 每个项目只写这一层 |
| Engine | typing, cache, batching, recording | 共享 |
| Backends | `DecisionBackend` trait + 5 implementations | 共享；自己的模型也是一个 backend |
| Loop | outcomes → calibration → training set → swap | 共享 |
| CLI | `so-ask` (one question), `so-rank` (rank a folder of markdown) — runtime schemas on top of the Engine | 共享；不写 Rust 也能用 |

## Design notes / 设计说明

* **`QuestionSchema` serialises 1:1 to the official API's `questions` object.** The derive macro builds it; the HTTP backend sends it verbatim; a custom backend receives the same structure. There is no second schema language.
* **Answers keep the full distribution.** `Choice` exposes every option's probability, not just the arg-max, so `decide(cost)` can minimise expected loss and `sample()` can drive stochastic agents.
* **Hashes, not ids.** Records, cache and replay are keyed by the SHA-256 of the canonical state JSON plus the schema hash. Two processes labelling the same data produce joinable rows without coordination.
* **Shadow never fails the primary.** A shadow error is recorded as a `shadow-error` row; the caller sees the primary answer.
* **The CLI tools are thin.** `so-ask` and `so-rank` build a `QuestionSchema` at runtime and go through the same `Engine` (`ask_raw`, `ask_many_raw`), cache and recorder as library code. `so-rank` judges every document on its own with no keyword pre-filter, and knows nothing about any particular notes collection: which fields, sections and strip patterns to use all come from its arguments. It never drops a document silently; unreadable files, failed requests and documents left without body text are listed.
* **`LocalLogprob` is the read-out half of a System One model.** It turns any instruction-tuned LLM into a schema-constrained classifier by reading label log-probs. Calibration is the training half — that is what the recorder and the report are for.

* **`QuestionSchema` 与官方 API 的 `questions` 对象一一对应。** 宏生成它，HTTP backend 原样发送，自定义 backend 收到同样的结构，没有第二套 schema 语言。
* **答案保留完整分布。** `Choice` 暴露每个选项的概率而不只是 argmax，所以 `decide(cost)` 可以最小化期望损失，`sample()` 可以驱动随机 agent。
* **用哈希不用 id。** 记录、缓存、回放都以「状态 JSON 的 SHA-256 + schema 哈希」为键，两个进程各自打标的结果无需协调就能 join。
* **Shadow 永远不会拖垮主 backend。** 影子出错记一行 `shadow-error`，调用方照常拿到主答案。
* **命令行工具很薄。** `so-ask` 和 `so-rank` 在运行时拼 `QuestionSchema`，走的是和库代码同一个 `Engine`（`ask_raw`、`ask_many_raw`）、缓存和 recorder。`so-rank` 对每篇文档单独判断，不做关键词预筛；它不认识任何特定的笔记库，用哪些字段、哪些章节、剥掉什么，全部来自参数。它从不悄悄丢文档：读不了的文件、失败的请求、剥完没有正文的文档都会列出来。
* **`LocalLogprob` 是 System One 模型的「读出」那一半。** 它把任何指令微调的 LLM 变成受 schema 约束的分类器（读取选项标签的 logprob）；「校准」那一半靠训练——这正是 recorder 和 calibration report 存在的意义。

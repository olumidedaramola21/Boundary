# Boundary

An autonomous coding agent that runs model-proposed shell commands inside an isolated, resource-capped Docker sandbox, measures how well it works, and deliberately attacks its own safety boundary to see what holds.

<p align="center">
  <a href="https://www.nga.gov/artworks/72328-roses" target="_blank" rel="noopener noreferrer">
    <img src="docs/assets/roses.jpg" alt="Van Gogh's Roses: white roses overflowing a tan jug, a few blooms fallen onto the table" width="720">
  </a>
</p>
<p align="center"><sub>
Vincent van Gogh, <a href="https://www.nga.gov/artworks/72328-roses"><i>Roses</i></a>, 1890. National Gallery of Art, Washington. Gift of Pamela Harriman in memory of W. Averell Harriman. Public domain.<br>
Most of the bouquet stays in the jug; a few blooms end up outside it. The roses were painted pink, and the color has since faded. What you see today isn't what was there, and we only know because a record exists.
</sub></p>

## Thesis
> An agent is only as safe as its boundaries, and a Boundary is only as trustworthy as the evidence that it holds.


Boundary makes three claims:

  1. **Anything that consumes agent output is a Boundary**. The sandbox that runs its commands, the verifier that grades its work, the summary that compresses its history, and the console that displays its traces can all be attacked through what the agent writes. The sandbox and the verifier are the same kind of object under adversarial pressure. The central question is what an agent trained to beat the verifier then does to the sandbox.

  2. **A guarantee isn't real until it has been attacked**. A Docker flag or an instruction to the model states an intention. Only an attack on the running system shows what holds, so every guarantee gets one, with its pass criteria committed before the attack runs.

  3. **The record outranks
   the agent's word**. A task passes when its check passes, not when the agent says it's done. A history summary that drops evidence of what the agent did is a failure, even when the agent carries on without noticing.

The painting makes the same argument. The jug holds most of the bouquet, but a few blooms have ended up on the table outside it. Containment is rarely all or nothing.

> [!NOTE]
> Boundary is under active development. The claims above are what it's built to test, not results yet. Some components described below are still being built, and the results will be published here once the evaluations that produce them have run.

## Why this exists
The agent loop itself (model proposes an action, the system executes it, the model observes the result, repeat) is common knowledge now. The questions Boundary is exploring include:

- Which parts of its safety are from the sandbox, and which is just hope?
- What happens when the environment turns hostile: a file containing injected instructions, a command that forks endlessly?
- How much does the agent forget when its history is trimmed or summarized?
- Does it actually complete tasks, and how often? What does a completed task cost?

Boundary attempts to answer these questions through three unified systems:

| Component | Description |
| :--- | :--- |
| An agent | A ReAct-style loop where the agent owns every iteration: explicit tool schemas, a model-agnostic core, hard limits, and an explicit stop reason for every run. |
| An evidence engine | An evaluation harness, a red-team suite, an amnesia evaluation, and a cross-model study, all producing data committed to the repo. |
| A console | A static website that turns that data into interactive replays, dashboards, and attack reports. |

## Research questions

### Core (in progress)
Builds the agent, the sandbox it runs in, and the tools that measure both: an evaluation harness, a red-team suite, an amnesia evaluation, and a cross-model study, with every result published in the console.

Testing how an agent that is honestly trying to do its job behaves. Does it complete real tasks, what does each success cost, what does it forget when is its history is trimmed, how much does it change beyond what it was asked to and which sandbox settings actually contain it when the environment turns hostile?

|  | Question| Prior work|
| :--- | :--- | :--- |
| RQ1 | Which sandbox hardening measures stop which classes of attack, and what attacks does the default configuration already contain? | Specific to Boundary's setup |
| RQ2 | How resistant is the agent to indirect prompt injection, meaning instructions hidden in files and command outputs? |  Replication of [IPI Arena](https://arxiv.org/abs/2603.15714) and [IssueTrojanBench](https://github.com/software-artifacts/IssueTrojanBench) |
| RQ3 | How much does the agent change beyond what the task requires (files outside scope, deletions, irreversible commands), and do models with similar pass rates differ?  | New; Closest: [CircumEval](https://www.lesswrong.com/posts/GHrqBKr8GLpbce6mN/door-s-locked-try-the-window)  |
| RQ4 | How much task-critical information does each context-management strategy lose? | Specific to Boundary's setup  |
| RQ5 | How does native tool-calling compare to regex-parsed text actions in reliability and failure modes, across models?  | Replication  |
| RQ6 | What is thhe cost per resolved task for each model, and what is the cost/quality frontier? |  Replication; standard cost-aware evaluation |

### Environment and training
Builds Boundary as an RL environment, with verification moved out of the agent's reach, a task factory that generates thousands of validated tasks from a small seed set, and a small open-weight model post-trained against it.

Testing what changes when the model stops trying to do the task and starts trying to make the check pass. An optimized policy will find any gap between the two, so this stage measures how the verifier can be gamed and which designs close each route, then what the training improves and what it breaks.

|  | Question| Prior work|
| :--- | :--- | :--- |
| RQ7 | When the verifier can be gamed, how often do agents game it, by which methods, and which verification designs close each route? | Replication of [ImpossibleBench](https://arxiv.org/abs/2510.20270)  |
| RQ8 | Does optimizing for task success degrade injection resistance, honesty or containment? Does reward hacking emerge during training, and does penalizing detected hacks remove it or only hide it? | Hack-penalty arm follows [Baker et al.](https://arxiv.org/abs/2503.11926) and [Drori et al.](https://arxiv.org/abs/2511.11584)  |
| RQ9 | Does post-training on generated tasks raise Pass@1 on held-out, hand-written tasks, and which method buys the most improvement per accelerator-hour? |  Replication; standard post-training methods |
| RQ10 | Is Boundary a usable training signal: solvable tasks, an accurate and deterministic verifier, a difficulty spread that actually produces learning signal? |  Replication of [SWE-smith](https://arxiv.org/abs/2504.21798) with a self-written seed corpus |

### Generalization and auditing
Builds a generalization experiment run on the models trained in the previous stage, an auditing tool that points the reward-integrity suite at other people's RL environments, and the statistics and monitoring needed for the results to hold up under review.

Testing whether teaching an agent to beat its verifier also makes it probe a sandbox Boundary it was never told about. Honeypots (planted credentials, a fake metadata service, a fake reward endpoint) turn containment into a measure of what the agent tries, not just what the sandbox stops. This also tests whether the same exploits work on third party environments, which monitors catch them, and whether history summaries quietly drop the evidence.

|  | Question| Prior work|
| :--- | :--- | :--- |
| RQ11 | Does optimizing against a gameable verifier increase unprompted probing of the sandbox boundary (opening planted credentials, reaching a fake metadata service, calling a fake reward endpoint) on tasks that never mention any of them? | New; Headline question   |
| RQ12 | When the reward-integrity suite is pointed at third-party RL environments, which exploit routes succeed, and do the fixes transfer back? | Builds on [Hodoscope](https://arxiv.org/abs/2604.11072); replication plus disclosure |
| RQ13 | On labeled multi-turn shell exploits, what recall does each class of monitor reach at a fixed false-positive rate, and does an activation probe trained on the base model survive RL?  | Probe test replicates the [Obfuscation Atlas](https://arxiv.org/abs/2602.15515) in a shell setting   |
| RQ14 | Do context-compaction summaries preserve evidence of the agent's flagged actions, and does training against a gameable verifier make them omit more of it? | New |


### Deployment
Builds Boundary's environment as a remote service, first on a cloud VM and then on Kubernetes, behind the same executor interface the local sandbox uses.

Testing whether the sandbox's guarantees survive a move to new infrastructure. Kubernetes can express most of the hardened profile, but some of its defaults quietly undo it; every pod gets a service-account token mounted unless told otherwise, which breaks "secrets never enter the container". This reruns the red-team suite on each platform to find which guarantees carry over, and measures what each platform costs.

|  | Question| Prior work|
| :--- | :--- | :--- |
| RQ15 | When the hardened profile moves from Docker to Kubernetes, which guarantees carry over, which weaken silently, and what does each platform-specific fix? | Specific to Boundary's setup   |
| RQ16 | How do reset latency, throughput, and cost per 1,000 episodes change from a local machine to a cloud VM to a Kubernetes cluster, and what does a stronger runtime cost in speed and task compatibility? |  Specific to Boundary's setup  |


Negative or suprising answers count as findings. Where a question overlaps published work, the result will be framed as a replication and cited.

> The prior work column comes from literature review. A replication repeats a published study; its value here is an open sandboxed version with every run's data committed. New means no direct precedent was found during review. Specific to Boundary's setup means the question measures this project's own sandbox, context, strategies, or infrastructure


## Roadmap
Boundary will be built in stages, all in this repository. The core release stands on its own; each stage builds on it.

| Stage | What it adds | Research Questions|
| :--- | :--- | :--- |
| Core (in progress) | Agent, eval harness, red-team suite, amnesia eval, model study | RQ1 - RQ6  |
| Environment and training | Boundary as an RL environment with protected verification, a task factory, and post-training a small open model against it  |  RQ7 - RQ10  |
| Generalization and auditing | Whether learning to game a verifier makes an agent probe its sandbox, boundary audit for checking other RL environment's verifiers, and a comparion of monitors for catching hacks |  RQ11 - RQ14  |
|Deployment | A cloud env server and a Kubernetes sandbox backend, attacked the same way |  RQ15 - RQ16  |


## Getting started
> [!WARNING] 
>Boundary executes shell commands written by a language model. They run inside a Docker container, never directly on your host, with no host folders mounted and networking off by default. Don't loosen those settings casually.


### Requirements
- Linux, or Windows with WSL2 (developed on WSL2 Ubuntu with Docker Engine running natively, not Docker Desktop)
- Python >=3.12,<3.13
- A Gemini API Key


### Installation and Running
1. Clone the repository and navigate into the folder:
```bash
git clone https://github.com/olumidedaramola21/Boundary.git 
cd Boundary

```

2. Install uv (if not already ) Follow the official instructions at [Astral uv Installation](https://docs.astral.sh/uv/getting-started/installation/).

3. Create and sync the project environment 
```bash
uv sync
```

4. Configure your API key (stored safely on the host, never passed into the sandbox)
```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

5. Run Boundary
```bash
  uv run python -m boundary.cli
```


## License

MIT. See [LICENSE](./LICENSE.txt)

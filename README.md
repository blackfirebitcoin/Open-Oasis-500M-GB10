# Open Oasis 500M on GB10 — single-GPU live port (negative result)

This is my fork of [`etched-ai/open-oasis`](https://github.com/etched-ai/open-oasis), ported to run live on a single NVIDIA DGX Spark (GB10, sm_121, aarch64). It exposes the public Oasis-500M weights through a FastAPI / WebSocket server you can play in a browser the same way Etched's hosted demo works.

I'm publishing this as a **negative-result writeup**: I tried to make the public Oasis-500M weights playable on this hardware and could not. The experiment files, the modified server, and the configurations I swept are all here so anyone landing on similar hardware doesn't have to redo the matrix.

## TL;DR

**Inside ~1–2 seconds of live input, the scene crystallizes into a blurry, unresponsive frame that stops reacting to keyboard or mouse.** I exhausted the practical inference-time levers — DDIM steps, context window, camera shaping, dynamic noising, multi-frame prompting, periodic VAE re-encode, input ablation — and none of them broke the ceiling. Multiple independent groups have reported the same behavior on the public 500M weights on other hardware, so I'm treating this as the architectural ceiling of the released checkpoint, not a port bug.

For a playable Minecraft world model on the same GPU, see my other fork: [Dreamerv4-MC-GB10](https://github.com/blackfirebitcoin/Dreamerv4-MC-GB10).

## Embedded video gallery

GitHub README markdown strips raw HTML5 `<video>` tags, so the directly viewable embedded gallery lives in a checked-in HTML page:

- **[Open the embedded video gallery](https://raw.githack.com/blackfirebitcoin/Open-Oasis-500M-GB10/dgx-spark-gb10/docs/video-gallery.html)** — RawGitHack-rendered, no setup required. Dark theme, hover the green `seed` badge on any clip to preview the start frame it was rendered from.
- [Gallery source in this repo](docs/video-gallery.html).
- [Direct MP4 directory](https://github.com/blackfirebitcoin/Open-Oasis-500M-GB10/tree/dgx-spark-gb10/benchmarks/renders) — browse and download individual clips.

The 13 clips trace: the headline starburst crystallization, the strongest live-run result, the strongest offline reference, the baseline live config (with and without input), the sampler / window / prompt / context-noise mitigation sweep, dynamic age-weighted noise, camera shaping, and input ablation.

## Best results I got

These are the most coherent frames I got out of my own live-play sessions. They look like real Minecraft for one or two seconds, then collapse.

| Savanna w/ water bucket | Dark portal scene |
| --- | --- |
| ![best-1](docs/results/best-savanna.png) | ![best-2](docs/results/best-portal.png) |

Both of these are the *opening* of a session. After roughly 30–60 frames everything becomes a soft blur and stops responding to input. Watch [`baseline-ddim4-w16.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/baseline-ddim4-w16.mp4) (with input) or [`baseline-noinput.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/baseline-noinput.mp4) (no input — proves crystallization is independent of input) for the live animation.

## What crystallization actually looks like

The single clearest demonstration I have of the architectural ceiling is what happens when you give the model a non-Minecraft seed image. A reasonable failure mode would be the model generating a chaotic block-soup that at least drifts under input — i.e. it interprets the input as *some* kind of scene and lets the player move through it. That isn't what happens.

This is a starburst optical-illusion image, fed to the live server with normal WASD + mouse input:

![starburst-crystallization](docs/results/starburst-seed.png)

**Watch the live animation:** [`starburst-crystallization.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/starburst-crystallization.mp4) (or [open it inside the gallery, with hover-to-preview seed](https://raw.githack.com/blackfirebitcoin/Open-Oasis-500M-GB10/dgx-spark-gb10/docs/video-gallery.html#starburst-crystallization.mp4)).

The model never resolves the starburst into anything chaotic-but-Minecraft-shaped. It just locks. Even a totally out-of-distribution seed cannot knock it off its fixed point — that's the cleanest way I can describe what "crystallization" means here.

## Why I'm calling this an architectural ceiling, not a port bug

Three independent confirmations on the same public weights:

- **LoopNav (May 2025)** evaluated public Oasis-500M and reported "model collapse, where generated images progressively degrade over time… models already fail at the simplest cases."
- **WorldPack (Dec 2025)** treats Oasis as a "known-weak baseline" newer architectures beat for spatial consistency.
- **Etched's own design page** lists fuzzy distance, temporal consistency of uncertain objects, and long-context difficulties as known limitations.

The "minutes of gameplay" claims in press coverage refer to Etched's larger **unreleased** models on H100s, not the public 500M release. The downloadable checkpoint is a downscaled reference. On any hardware, what you can actually pull from HuggingFace caps out at ~1.6 seconds of memory: `dit.py` defines `max_frames=32`, and the live server defaults to a 16-frame sliding window for latency reasons. I tested both 16 and 32; the ceiling moves by a fraction of a second and that's it.

## What I tried (and what didn't help)

Most rows below link to a representative clip in the [video gallery](https://raw.githack.com/blackfirebitcoin/Open-Oasis-500M-GB10/dgx-spark-gb10/docs/video-gallery.html).

| Lever | Configs swept | Effect on crystallization | Representative clip |
| --- | --- | --- | --- |
| DDIM steps | 1, 2, 4, 8 | None (slows FPS, ceiling unchanged) | [`mitigation-ddim8.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/mitigation-ddim8.mp4) |
| Context window | 16, 32 | None (32 is the architectural max for 500M) | [`mitigation-window32.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/mitigation-window32.mp4) |
| Stabilization level | 5, 15, 30 | None | [`mitigation-stabilization30.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/mitigation-stabilization30.mp4) |
| Pin first prompt frame | on / off | None | [`mitigation-pin-prompt.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/mitigation-pin-prompt.mp4) |
| Constant context-noise injection | on / off | None | [`mitigation-context-noise.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/mitigation-context-noise.mp4) |
| Dynamic age-weighted noise | `max ∈ {15, 50, 100, 200}` × `age_weight ∈ {0, 0.5, 0.7}` | None | [`dynamic-noising-best-move.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/dynamic-noising-best-move.mp4) |
| Camera input shaping ("safe-camera") | period=12, max=18px, pitch=0 | None — the visual feels smoother but the scene still crystallizes | [`safe-camera-fwd-sway-120.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/safe-camera-fwd-sway-120.mp4) |
| Multi-frame video prompt | n=4, 8 | None | (in `live_server.py` flags) |
| Real MineRL action prefill | n=4, 8 | None | (in `live_server.py` flags) |
| Periodic VAE re-encode | period ∈ {4, 8, 12, 16} | None | — |
| Input ablation | W-only, mouse-only, step-input, sparse-yaw | None | [`ablate-walk-only.mp4`](https://cdn.jsdelivr.net/gh/blackfirebitcoin/Open-Oasis-500M-GB10@dgx-spark-gb10/benchmarks/renders/ablate-walk-only.mp4) |

CLI flags for every one of these are still present in `live_server.py` so the experiments are reproducible.


## What is actually in this fork

| File | Status | Notes |
| --- | --- | --- |
| `live_server.py` | new | FastAPI / WebSocket server with all experimental CLI flags from the sweep. Defaults are the baseline. |
| `utils.py` | upstream bug fix | `torchvision.io.read_video` returns THWC; upstream `load_prompt` was treating it as TCHW and silently producing a one-frame prompt. |
| `static/index.html` | new | Minimal browser UI for live play and status. |
| `static/seeds/` | new | Seed images used by the live server and gallery (savanna, crickle, portal, boat, starburst, etc.). |
| `dit.py`, `vae.py`, `attention.py`, `rotary_embedding_torch.py`, `generate.py` | unmodified | Carried as-is from upstream. |
| `benchmarks/renders/` | new | 13 MP4 clips from the sweep. Inline-playable through `docs/video-gallery.html`. |
| `docs/video-gallery.html` | new | Embedded video gallery (dark theme, hover-to-preview seeds). |
| `docs/UPSTREAM-README.md` | preserved | Original Etched / Decart README. |
| `docs/results/` | new | Best-result still images shown in the README above. |

## Setup on GB10

Same as upstream, plus the live-server deps:

```bash
git clone https://github.com/blackfirebitcoin/Open-Oasis-500M-GB10.git
cd Open-Oasis-500M-GB10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops diffusers timm av fastapi uvicorn websockets
huggingface-cli login
huggingface-cli download Etched/oasis-500m oasis500m.safetensors
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors
python live_server.py --port 8766
# open http://localhost:8766
```

## What's next

Not pursuing this further on the inference side. The only realistic remaining lever I can see is training-side — LoRA finetune on long stable sequences, a custom inference stack with a DM-style memory module, or the unreleased larger Etched weights — and that's out of scope for a single-GPU port project.

If you want a playable Minecraft world model on GB10 today, use [Dreamerv4-MC-GB10](https://github.com/blackfirebitcoin/Dreamerv4-MC-GB10).

## Credits

Upstream model and reference implementation: [Etched](https://www.etched.com/) and [Decart](https://www.decart.ai/) — [`etched-ai/open-oasis`](https://github.com/etched-ai/open-oasis). This fork's GB10 port, live server scaffolding, sweep, and writeup by Page.

"""
Open Oasis 500M — live FastAPI/WebSocket server for GB10.

Phase 0 envelope, baked-in defaults:
  ddim_steps=4, max_window=16, camera_clamp=±0.3
  mouse_pixel_sensitivity=200, mouse_ema_alpha=0.5
  Decode-latest-only optimization vs stock generate.py.
"""
import argparse
import asyncio
import contextlib
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from safetensors.torch import load_model

sys.path.insert(0, str(Path(__file__).parent))
from dit import DiT_models  # noqa: E402
from vae import VAE_models  # noqa: E402
from utils import load_prompt, sigmoid_beta_schedule, ACTION_KEYS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("oasis-live")


class OasisLiveEngine:
    """Single-GPU, single-client interactive engine."""

    def __init__(self, dit_path, vae_path, device="cuda:0",
                 ddim_steps=4, max_window=16, camera_clamp=0.3,
                 mouse_pixel_sensitivity=200.0, mouse_ema_alpha=0.5,
                 max_noise_level=1000, noise_abs_max=20.0, stabilization_level=15,
                 pin_prompt=False, inject_context_noise=False,
                 dynamic_noising=False, dynamic_noise_max=50,
                 dynamic_noise_age_weight=0.0,
                 safe_camera=False, safe_camera_period=12,
                 safe_camera_max_px=18.0, safe_camera_deadzone_px=4.0,
                 safe_camera_pitch_scale=0.0,
                 vae_reencode_period=0):
        self.device = device
        self.ddim_steps = ddim_steps
        self.max_window = max_window
        self.camera_clamp = camera_clamp
        self.mouse_pixel_sensitivity = mouse_pixel_sensitivity
        self.mouse_ema_alpha = mouse_ema_alpha
        self.max_noise_level = max_noise_level
        self.noise_abs_max = noise_abs_max
        self.stabilization_level = stabilization_level
        self.pin_prompt = pin_prompt
        self.inject_context_noise = inject_context_noise
        self.dynamic_noising = dynamic_noising
        self.dynamic_noise_max = int(dynamic_noise_max)
        self.dynamic_noise_age_weight = float(dynamic_noise_age_weight)
        self.safe_camera = bool(safe_camera)
        self.safe_camera_period = max(1, int(safe_camera_period))
        self.safe_camera_max_px = float(safe_camera_max_px)
        self.safe_camera_deadzone_px = float(safe_camera_deadzone_px)
        self.safe_camera_pitch_scale = float(safe_camera_pitch_scale)
        self.safe_camera_pending_yaw_px = 0.0
        self.safe_camera_pending_pitch_px = 0.0
        self.vae_reencode_period = max(0, int(vae_reencode_period))
        self.cli_prompt_path = None
        self.cli_n_prompt_frames = 1
        self.cli_video_offset = None
        self.cli_prefill_actions = None

        log.info(f"loading DiT from {dit_path}...")
        self.dit = DiT_models["DiT-S/2"]()
        load_model(self.dit, dit_path)
        self.dit = self.dit.to(device).eval()

        log.info(f"loading VAE from {vae_path}...")
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(self.vae, vae_path)
        self.vae = self.vae.to(device).eval()

        self.scaling_factor = 0.07843137255
        betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
        self.idx = {k: i for i, k in enumerate(ACTION_KEYS)}

        self.x: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.mouse_smooth_yaw = 0.0
        self.mouse_smooth_pitch = 0.0
        self.safe_camera_pending_yaw_px = 0.0
        self.safe_camera_pending_pitch_px = 0.0
        self.frame_counter = 0
        log.info(f"engine ready: ddim={ddim_steps} window={max_window} clamp=±{camera_clamp} "
                 f"sens={mouse_pixel_sensitivity}px ema_alpha={mouse_ema_alpha} stab={stabilization_level} "
                 f"pin_prompt={pin_prompt} inject_ctx_noise={inject_context_noise} "
                 f"dyn_noise={dynamic_noising} dyn_max={dynamic_noise_max} dyn_age_w={dynamic_noise_age_weight} "
                 f"safe_camera={safe_camera} safe_period={safe_camera_period} safe_max_px={safe_camera_max_px} "
                 f"safe_deadzone_px={safe_camera_deadzone_px} safe_pitch_scale={safe_camera_pitch_scale} "
                 f"vae_reencode_period={vae_reencode_period}")

    def _encode_prompt(self, prompt_path, n_prompt_frames=1, video_offset=None):
        x = load_prompt(prompt_path, video_offset=video_offset, n_prompt_frames=n_prompt_frames).to(self.device)
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.half):
                latent = self.vae.encode(x_flat * 2 - 1).mean * self.scaling_factor
        latent = rearrange(latent, "(b t) (h w) c -> b t c h w",
                           t=T, h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        return latent

    def _decode_then_reencode(self, latent_BTCHW):
        """Round-trip the latest latent through pixel space and back through VAE.

        Discards drift that lives outside the VAE manifold.
        """
        B, T, C, H, W = latent_BTCHW.shape
        flat = rearrange(latent_BTCHW, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.half):
                pixels = (self.vae.decode(flat / self.scaling_factor) + 1) / 2
                pixels = torch.clamp(pixels, 0, 1)
                pixels = pixels * 2 - 1
                latent = self.vae.encode(pixels).mean * self.scaling_factor
        latent = rearrange(latent, "(b t) (h w) c -> b t c h w",
                           t=T, h=pixels.shape[-2] // self.vae.patch_size,
                           w=pixels.shape[-1] // self.vae.patch_size)
        return latent

    def reset(self, prompt_path, n_prompt_frames=None, video_offset=None,
              prefill_actions=None):
        if n_prompt_frames is None:
            n_prompt_frames = self.cli_n_prompt_frames
        if video_offset is None:
            video_offset = self.cli_video_offset
        if prefill_actions is None:
            prefill_actions = self.cli_prefill_actions
        log.info(f"reset: encoding prompt {prompt_path} n_prompt_frames={n_prompt_frames}")
        self.x = self._encode_prompt(prompt_path, n_prompt_frames=n_prompt_frames,
                                     video_offset=video_offset)
        T = self.x.shape[1]
        if prefill_actions is not None and prefill_actions.shape[0] >= T:
            self.actions = prefill_actions[:T].to(self.device).unsqueeze(0)
            log.info(f"reset: using prefill_actions for {T} frames")
        else:
            self.actions = torch.zeros(1, T, 25, device=self.device)
        self.mouse_smooth_yaw = 0.0
        self.mouse_smooth_pitch = 0.0
        self.safe_camera_pending_yaw_px = 0.0
        self.safe_camera_pending_pitch_px = 0.0
        self.frame_counter = 0

    def _build_action(self, snap):
        a = torch.zeros(25, device=self.device)
        keys = set(snap.get("keys", []))
        buttons = set(snap.get("buttons", []))
        if "KeyW" in keys: a[self.idx["forward"]] = 1.0
        if "KeyS" in keys: a[self.idx["back"]] = 1.0
        if "KeyA" in keys: a[self.idx["left"]] = 1.0
        if "KeyD" in keys: a[self.idx["right"]] = 1.0
        if "Space" in keys: a[self.idx["jump"]] = 1.0
        if "ShiftLeft" in keys: a[self.idx["sneak"]] = 1.0
        if "ControlLeft" in keys: a[self.idx["sprint"]] = 1.0
        if 0 in buttons: a[self.idx["attack"]] = 1.0
        if 2 in buttons: a[self.idx["use"]] = 1.0
        # VPT action layout: cameraX = pitch (vertical look), cameraY = yaw (horizontal look).
        # Browser dx is horizontal mouse motion -> yaw; dy is vertical -> pitch.
        # No EMA: raw per-tick deltas, just clamped. EMA was inherited from Dreamer
        # but caused persistent at-clamp camera magnitude that accelerated decoherence.
        raw_dx = float(snap.get("dx", 0.0))
        raw_dy = float(snap.get("dy", 0.0))
        if self.safe_camera:
            self.safe_camera_pending_yaw_px += raw_dx
            self.safe_camera_pending_pitch_px += raw_dy
            pulse_now = ((self.frame_counter + 1) % self.safe_camera_period) == 0
            if pulse_now and abs(self.safe_camera_pending_yaw_px) >= self.safe_camera_deadzone_px:
                yaw_px = max(-self.safe_camera_max_px,
                             min(self.safe_camera_max_px, self.safe_camera_pending_yaw_px))
            else:
                yaw_px = 0.0
            if pulse_now and abs(self.safe_camera_pending_pitch_px) >= self.safe_camera_deadzone_px:
                pitch_px = max(-self.safe_camera_max_px,
                               min(self.safe_camera_max_px, self.safe_camera_pending_pitch_px))
                pitch_px *= self.safe_camera_pitch_scale
            else:
                pitch_px = 0.0
            if pulse_now:
                # Drop remainder instead of draining it forever; the goal is controlled sparse
                # camera impulses, not replaying every raw browser mouse pixel.
                self.safe_camera_pending_yaw_px = 0.0
                self.safe_camera_pending_pitch_px = 0.0
        else:
            yaw_px = raw_dx
            pitch_px = raw_dy
        yaw_norm   = yaw_px / self.mouse_pixel_sensitivity
        pitch_norm = pitch_px / self.mouse_pixel_sensitivity
        yaw_clamped   = max(-self.camera_clamp, min(self.camera_clamp, yaw_norm))
        pitch_clamped = max(-self.camera_clamp, min(self.camera_clamp, pitch_norm))
        a[self.idx["cameraX"]] = pitch_clamped
        a[self.idx["cameraY"]] = yaw_clamped
        return a.unsqueeze(0).unsqueeze(0)

    def step(self, snap):
        timing = {}
        t_total = time.perf_counter()

        new_action = self._build_action(snap)
        self.actions = torch.cat([self.actions, new_action], dim=1)

        B = self.x.shape[0]
        chunk = torch.randn((B, 1, *self.x.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -self.noise_abs_max, +self.noise_abs_max)
        self.x = torch.cat([self.x, chunk], dim=1)
        i = self.x.shape[1] - 1
        use_pinned_window = self.pin_prompt and self.x.shape[1] > self.max_window
        start_frame = max(0, i + 1 - self.max_window)

        t_denoise = time.perf_counter()
        noise_range = torch.linspace(-1, self.max_noise_level - 1, self.ddim_steps + 1)
        ddim_iter = list(reversed(range(1, self.ddim_steps + 1)))
        for step_idx, noise_idx in enumerate(ddim_iter):
            t_ctx = torch.full((B, i), self.stabilization_level - 1, dtype=torch.long, device=self.device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=self.device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)
            if use_pinned_window:
                x_curr = torch.cat([self.x[:, :1], self.x[:, -(self.max_window - 1):]], dim=1).clone()
                actions_curr = torch.cat([self.actions[:, :1], self.actions[:, -(self.max_window - 1):]], dim=1)
                t = torch.cat([t[:, :1], t[:, -(self.max_window - 1):]], dim=1)
                t_next = torch.cat([t_next[:, :1], t_next[:, -(self.max_window - 1):]], dim=1)
            else:
                x_curr = self.x.clone()[:, start_frame:]
                actions_curr = self.actions[:, start_frame: i + 1]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]
            # Constant inject (legacy / negative-control path)
            if self.inject_context_noise and x_curr.shape[1] > 1:
                a_ctx = self.alphas_cumprod[self.stabilization_level - 1].sqrt()
                s_ctx = (1 - self.alphas_cumprod[self.stabilization_level - 1]).sqrt()
                ctx_noise = torch.randn_like(x_curr[:, :-1])
                x_curr = torch.cat([
                    a_ctx * x_curr[:, :-1] + s_ctx * ctx_noise,
                    x_curr[:, -1:],
                ], dim=1)

            # Scheduled dynamic noising: feed model a noised view of context
            # (per-pass schedule, optional age weighting). Does NOT modify self.x or x_curr;
            # builds a temporary x_for_model + matching t_for_model just for the DiT call.
            x_for_model = x_curr
            t_for_model = t
            if self.dynamic_noising and x_curr.shape[1] > 1 and self.dynamic_noise_max > 0:
                ctx_len = x_curr.shape[1] - 1
                ddim_steps = self.ddim_steps
                progress = step_idx / max(1, ddim_steps - 1)
                ctx_max_level = int(round(self.dynamic_noise_max * (1.0 - progress)))
                if ctx_max_level > 0:
                    # ages: oldest=1.0, newest=0.0; multiplier scales noise per-frame
                    ages = torch.linspace(1.0, 0.0, ctx_len, device=self.device)
                    aw = self.dynamic_noise_age_weight
                    multiplier = 1.0 - aw * (1.0 - ages)
                    levels = (ctx_max_level * multiplier).clamp(min=0.0, max=float(self.max_noise_level - 1)).long()
                    levels_b = levels.unsqueeze(0).expand(B, -1)  # (B, ctx_len)
                    a_ctx = self.alphas_cumprod[levels_b].sqrt()           # (B, ctx_len, 1, 1, 1)
                    s_ctx = (1.0 - self.alphas_cumprod[levels_b]).sqrt()
                    ctx_clean = x_curr[:, :-1]
                    ctx_noise = torch.randn_like(ctx_clean)
                    ctx_noised = a_ctx * ctx_clean + s_ctx * ctx_noise
                    x_for_model = torch.cat([ctx_noised, x_curr[:, -1:]], dim=1)
                    t_for_model = torch.cat([levels_b, t[:, -1:]], dim=1)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.half):
                    v = self.dit(x_for_model, t_for_model, actions_curr)
            x_start = self.alphas_cumprod[t].sqrt() * x_curr - (1 - self.alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / self.alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / self.alphas_cumprod[t] - 1).sqrt()
            alpha_next = self.alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            self.x[:, -1:] = x_pred[:, -1:]
        torch.cuda.synchronize()
        timing["denoise_ms"] = (time.perf_counter() - t_denoise) * 1000

        t_decode = time.perf_counter()
        latest = self.x[:, -1:]
        latest_flat = rearrange(latest, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.half):
                decoded = (self.vae.decode(latest_flat / self.scaling_factor) + 1) / 2
        decoded = rearrange(decoded, "(b t) c h w -> b t h w c", t=1)
        decoded = torch.clamp(decoded, 0, 1)
        decoded = (decoded * 255).byte()
        frame = decoded[0, 0].cpu().numpy()
        torch.cuda.synchronize()
        timing["decode_ms"] = (time.perf_counter() - t_decode) * 1000

        t_jpeg = time.perf_counter()
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpeg = buf.getvalue()
        timing["jpeg_ms"] = (time.perf_counter() - t_jpeg) * 1000

        if self.x.shape[1] > self.max_window:
            if self.pin_prompt:
                self.x = torch.cat([self.x[:, :1], self.x[:, -(self.max_window - 1):]], dim=1).contiguous()
                self.actions = torch.cat([self.actions[:, :1], self.actions[:, -(self.max_window - 1):]], dim=1).contiguous()
            else:
                self.x = self.x[:, -self.max_window:].contiguous()
                self.actions = self.actions[:, -self.max_window:].contiguous()

        # VAE re-encode: every N frames decode-then-encode the latest latent
        # to discard accumulated drift outside the VAE manifold.
        if self.vae_reencode_period > 0 and \
           ((self.frame_counter + 1) % self.vae_reencode_period) == 0:
            t_re = time.perf_counter()
            cleaned = self._decode_then_reencode(self.x[:, -1:])
            self.x = torch.cat([self.x[:, :-1], cleaned], dim=1).contiguous()
            timing["reencode_ms"] = (time.perf_counter() - t_re) * 1000

        self.frame_counter += 1
        timing["total_ms"] = (time.perf_counter() - t_total) * 1000
        timing["window"] = int(self.x.shape[1])
        timing["fps"] = round(1000.0 / timing["total_ms"], 2)
        timing["ddim_steps"] = self.ddim_steps
        timing["frame"] = self.frame_counter
        return jpeg, timing


class ClientState:
    def __init__(self):
        self.keys = set()
        self.buttons = set()
        self.dx_accum = 0.0
        self.dy_accum = 0.0
        self.want_reset = False
        self.reset_seed = None
        self.lock = asyncio.Lock()

    async def update(self, msg):
        async with self.lock:
            t = msg.get("type")
            if t == "key":
                code = msg.get("code", "")
                if msg.get("down"): self.keys.add(code)
                else: self.keys.discard(code)
            elif t == "button":
                btn = int(msg.get("button", 0))
                if msg.get("down"): self.buttons.add(btn)
                else: self.buttons.discard(btn)
            elif t == "mouse":
                self.dx_accum += float(msg.get("dx", 0.0))
                self.dy_accum += float(msg.get("dy", 0.0))
            elif t == "reset":
                self.want_reset = True
                self.reset_seed = msg.get("seed")
            elif t == "blur":
                self.keys.clear()
                self.buttons.clear()

    async def consume(self):
        async with self.lock:
            snap = {"keys": list(self.keys), "buttons": list(self.buttons),
                    "dx": self.dx_accum, "dy": self.dy_accum}
            self.dx_accum = 0.0
            self.dy_accum = 0.0
            return snap

    async def consume_reset(self):
        async with self.lock:
            if self.want_reset:
                seed = self.reset_seed
                self.want_reset = False
                self.reset_seed = None
                return seed or "DEFAULT"
            return None


def make_app(engine, seeds_dir, default_seed):
    app = FastAPI()
    app.state.client_active = False
    app.state.client_lock = asyncio.Lock()

    @app.get("/")
    async def index():
        with open(Path(__file__).parent / "static" / "index.html") as f:
            return HTMLResponse(f.read())

    @app.get("/seeds")
    async def list_seeds():
        files = sorted([p.name for p in seeds_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        return {"seeds": files, "default": default_seed}

    app.mount("/seeds_img", StaticFiles(directory=str(seeds_dir)), name="seeds_img")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        async with app.state.client_lock:
            if app.state.client_active:
                await socket.accept()
                await socket.send_json({"type": "error",
                                        "message": "Another client is connected. Close that tab first."})
                await socket.close()
                return
            app.state.client_active = True

        await socket.accept()
        log.info("client connected")
        client = ClientState()

        ds = default_seed
        if Path(ds).is_absolute() or Path(ds).exists():
            full_path = str(ds)
        else:
            full_path = str(seeds_dir / ds)
        await asyncio.to_thread(engine.reset, full_path)
        await socket.send_json({"type": "ready", "default_seed": default_seed,
                                "ddim_steps": engine.ddim_steps,
                                "window": engine.max_window,
                                "camera_clamp": engine.camera_clamp,
                                "safe_camera": engine.safe_camera,
                                "safe_camera_period": engine.safe_camera_period,
                                "safe_camera_max_px": engine.safe_camera_max_px})

        async def listener():
            try:
                while True:
                    msg = await socket.receive_json()
                    await client.update(msg)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                log.warning(f"listener error: {e}")

        listener_task = asyncio.create_task(listener())

        try:
            while True:
                seed = await client.consume_reset()
                if seed:
                    p = seeds_dir / seed
                    if not p.exists():
                        p = seeds_dir / default_seed
                    await asyncio.to_thread(engine.reset, str(p))
                    await socket.send_json({"type": "reset_done", "seed": p.name})
                    continue

                snap = await client.consume()
                jpeg, timing = await asyncio.to_thread(engine.step, snap)
                t_send = time.perf_counter()
                try:
                    await socket.send_bytes(jpeg)
                    timing["ws_ms"] = (time.perf_counter() - t_send) * 1000
                    await socket.send_json({"type": "timing", **timing})
                except (WebSocketDisconnect, RuntimeError) as e:
                    log.info(f"client disconnected during send: {e}")
                    break
                if engine.frame_counter % 10 == 0:
                    log.info(f"[TIMING] frame={engine.frame_counter} "
                             f"denoise={timing['denoise_ms']:.0f}ms "
                             f"decode={timing['decode_ms']:.0f}ms "
                             f"jpeg={timing['jpeg_ms']:.0f}ms "
                             f"total={timing['total_ms']:.0f}ms "
                             f"fps={timing['fps']}")
        except WebSocketDisconnect:
            log.info("client disconnected")
        except Exception as e:
            log.exception(f"render loop error: {e}")
        finally:
            listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener_task
            async with app.state.client_lock:
                app.state.client_active = False

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default="oasis500m.safetensors")
    p.add_argument("--vae", default="vit-l-20.safetensors")
    p.add_argument("--seeds-dir", default="static/seeds")
    p.add_argument("--default-seed", default="imgs_0.png")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--ddim-steps", type=int, default=4)
    p.add_argument("--max-window", type=int, default=16)
    p.add_argument("--camera-clamp", type=float, default=0.3)
    p.add_argument("--mouse-sensitivity", type=float, default=200.0)
    p.add_argument("--mouse-ema-alpha", type=float, default=0.5)
    p.add_argument("--stabilization-level", type=int, default=15)
    p.add_argument("--pin-prompt", action="store_true")
    p.add_argument("--inject-context-noise", action="store_true")
    p.add_argument("--dynamic-noising", action="store_true",
                   help="Etched-style scheduled context noise (front-loaded across DDIM passes)")
    p.add_argument("--dynamic-noise-max", type=int, default=50,
                   help="Peak context noise level on the first DDIM pass (0..max_noise_level-1)")
    p.add_argument("--dynamic-noise-age-weight", type=float, default=0.0,
                   help="0=uniform across context, 1=newest gets 0x noise oldest gets 1x")
    p.add_argument("--safe-camera", action="store_true",
                   help="Convert raw browser mouse into sparse low-frequency yaw pulses")
    p.add_argument("--safe-camera-period", type=int, default=12)
    p.add_argument("--safe-camera-max-px", type=float, default=18.0)
    p.add_argument("--safe-camera-deadzone-px", type=float, default=4.0)
    p.add_argument("--safe-camera-pitch-scale", type=float, default=0.0)
    p.add_argument("--prompt-path", type=str, default=None,
                   help="Override default seed; supports image (.png/.jpg) or video (.mp4)")
    p.add_argument("--n-prompt-frames", type=int, default=1,
                   help="Number of frames to use from a video prompt (>=1)")
    p.add_argument("--video-offset", type=int, default=None,
                   help="Start offset (frames) when prompt is a video")
    p.add_argument("--actions-path", type=str, default=None,
                   help="Optional .one_hot_actions.pt for action prefill matching prompt frames")
    p.add_argument("--vae-reencode-period", type=int, default=0,
                   help="Re-encode the latest latent through the VAE every N frames (0=off)")
    args = p.parse_args()

    seeds_dir = Path(args.seeds_dir).resolve()
    if not seeds_dir.exists():
        log.error(f"seeds dir does not exist: {seeds_dir}")
        sys.exit(2)
    if not (seeds_dir / args.default_seed).exists():
        log.error(f"default seed missing: {args.default_seed}")
        sys.exit(2)

    cli_actions = None
    if args.actions_path:
        from utils import load_actions as _load_actions
        loaded = _load_actions(args.actions_path)
        cli_actions = loaded.squeeze(0)
        log.info(f"loaded prefill actions {tuple(cli_actions.shape)} from {args.actions_path}")

    engine = OasisLiveEngine(
        dit_path=args.dit, vae_path=args.vae,
        ddim_steps=args.ddim_steps, max_window=args.max_window,
        camera_clamp=args.camera_clamp,
        mouse_pixel_sensitivity=args.mouse_sensitivity,
        mouse_ema_alpha=args.mouse_ema_alpha,
        stabilization_level=args.stabilization_level,
        pin_prompt=args.pin_prompt,
        inject_context_noise=args.inject_context_noise,
        dynamic_noising=args.dynamic_noising,
        dynamic_noise_max=args.dynamic_noise_max,
        dynamic_noise_age_weight=args.dynamic_noise_age_weight,
        safe_camera=args.safe_camera,
        safe_camera_period=args.safe_camera_period,
        safe_camera_max_px=args.safe_camera_max_px,
        safe_camera_deadzone_px=args.safe_camera_deadzone_px,
        safe_camera_pitch_scale=args.safe_camera_pitch_scale,
        vae_reencode_period=args.vae_reencode_period,
    )
    engine.cli_prompt_path = args.prompt_path
    engine.cli_n_prompt_frames = max(1, int(args.n_prompt_frames))
    engine.cli_video_offset = args.video_offset
    engine.cli_prefill_actions = cli_actions
    default_for_app = args.prompt_path if args.prompt_path else args.default_seed
    app = make_app(engine, seeds_dir, default_for_app)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

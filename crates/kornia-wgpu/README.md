# kornia-wgpu

Hardware-accelerated image and tensor operations for Kornia-RS via WebGPU.

This crate is a prototype and GSoC proposal for introducing a lightweight, portable GPU compute backend to the Kornia-RS ecosystem.

[Google Docs proposal](https://docs.google.com/document/d/1f9y_QCpjZI-XzioxuNEmyO0uMCO2iC9Z4pjMTrP8GLY/edit?usp=sharing)

#### For the maintainers: feel free to comment any suggestions and improvements, in case there are some inconsistencies.

---

## Synopsis

Kornia-RS currently relies on CPU-bound operations. `kornia-wgpu` implements `ops::image` and `ops::tensor` modules to enable high-performance spatial image processing and multidimensional tensor math.

Built entirely on safe Rust abstractions, it uses `wgpu` (v28.0.0) to provide a portable GPU compute backend across Vulkan, Metal, DX12, and WebGL : preserving Kornia-RS's lightweight philosophy by avoiding heavyweight dependencies like CUDA or BLAS.

---

## Why wgpu?

### wgpu over CubeCL

`wgpu` targets Vulkan, Metal, DX12, and WebGL from a single codebase, while CubeCL primarily targets CUDA and ROCm : introducing a hard CUDA dependency that contradicts kornia-rs's lightweight philosophy. CubeCL is also still in early development with a frequently changing API and sparse documentation, making it a poor foundation for a stable library. `wgpu` is mature, well-documented, and requires no external toolchain beyond the GPU drivers already present on the target platform.

Additionally, wgpu exposes texture bindings and hardware sampler units (TMUs), enabling operations like resize, perspective warp, and separable filters to delegate interpolation entirely to dedicated fixed-function hardware. CubeCL has no equivalent : all sampling must be implemented manually in shader arithmetic on storage buffers.

CubeCL kernels are also launched via `unsafe` functions with no compile-time verification of argument types or vectorisation factors : mismatches produce runtime panics or silent incorrect results. kornia-wgpu confines all `unsafe` to a single auditable call in `transfer.rs`, with kernel arguments enforced at compile time via `GpuImage<T, C>` and `GpuTensor<T, N>` newtypes.

### wgpu over raw CUDA

A raw CUDA backend via `cudarc` was considered and rejected for three reasons. First, it reintroduces a hard CUDA dependency and requires `nvcc` at build time : contradicting kornia-rs's zero-external-toolchain philosophy. Second, it only runs on NVIDIA hardware, breaking on AMD, Apple Silicon, Raspberry Pi, and in CI environments without a GPU. Third, wgpu's Vulkan backend on NVIDIA hardware with hand-written WGSL and hardware texture sampling achieves comparable throughput while remaining fully portable.

The same codebase that runs at 30 fps on a Jetson Orin (Vulkan) also runs unchanged on a developer's MacBook (Metal), in a browser (WebGL), and in CI without any GPU (llvm_pipe). A CUDA backend cannot do any of those.

---

## Key Features

**Cross-platform GPU compute.** Runs on Vulkan, Metal, DX12, and WebGL via WGSL shaders with no platform-specific code.

**Zero-copy GPU chaining.** Execute multiple operations sequentially in VRAM. The GPU reads its own outputs as the next operation's inputs without any PCIe transfers between ops.

**Real-time video processing.** Grab live frames from cameras or RTSP streams via `kornia-io`, process them on the GPU, and stream results : all without leaving Rust.

**Dynamic pipeline caching.** `WgpuSession` owns the device and queue and caches compiled WGSL pipelines. Each `(op, type)` pair is compiled exactly once, avoiding expensive shader compilations during hot loops.

**Pool-based memory management.** All GPU buffers : compute outputs and staging readback buffers : come from pre-allocated, size-class-keyed pools. After the first frame, zero allocations occur per frame for fixed-size workloads. Pool return is automatic on drop via `PooledBufferGuard` : no manual `acquire()`/`release()` calls, and no risk of pool exhaustion from buffers held across async cancel points.

**Compile-time GPU/CPU boundary enforcement.** `GpuImage<T, C>` and `GpuTensor<T, N>` are newtypes that expose no CPU slice methods. Calling `.as_slice()` on a GPU resource is a compiler error, not a runtime panic. The only way to read GPU data is `image_to_cpu` or `download_tensor`.

**Unconditionally safe data transfers.** Uses `bytemuck` and a `GpuPixel` supertrait to guarantee there are no padding bytes, making CPU ↔ GPU byte reinterpretations completely safe. The `Pod` bound is enforced at compile time : if a type has padding, it won't compile. There is exactly one `unsafe` block in the entire crate, in `transfer.rs`, which is locally auditable.

**Native u8 support.** The `cast_u8_to_f32_gpu` kernel uploads raw u8 frames directly and packs 4 bytes per u32 to work around WGSL's lack of native u8 storage, dividing by 255 in the shader : eliminating the CPU cast bottleneck entirely.

**Hardware texture sampling (planned).** Where operations involve spatially-local sampling : resize, perspective warp, and separable filters : the implementation will use wgpu texture bindings rather than storage buffers, delegating bilinear interpolation and boundary handling to dedicated hardware texture units (TMUs). This typically yields a 3–5× improvement over storage buffer shaders and is the primary planned optimisation beyond the current prototype. CubeCL has no equivalent capability.

**Runs in CI without a GPU.** wgpu's `llvm_pipe` software backend provides a production-grade CPU execution path : the same WGSL shaders run unchanged, with no mocking or test-only codepaths. Correctness tests, pool lifecycle tests, and buffer round-trip tests all pass on a standard GitHub Actions runner with no GPU present. Performance benchmarks require a real GPU; `llvm_pipe` is not representative of hardware throughput.

---

## Architecture

```mermaid
flowchart TB
    U(["User code"]):::user

    subgraph OPS["  Ops layer  "]
        OI["ops::image
        resize · cast_u8_to_f32 · grayscale · flip · normalize · filters · warp"]:::ops
        OT["ops::tensor
        elementwise · reductions · activations · matmul"]:::ops
    end

    subgraph SES["  Session layer  "]
        S["WgpuSession
        Device + Queue · PipelineKey cache · acquire_compute()"]:::session
    end

    subgraph MEM["  Memory layer  "]
        GT["GpuImage / GpuTensor
        Newtypes : no CPU slice methods"]:::memory
        MA["WgpuAllocator + PooledBufferGuard
        Arc<wgpu::Buffer> · pool return on drop"]:::memory
        MP["BufferPool + StagingPoolMap
        size-class keyed · zero alloc after warmup"]:::memory
        MT["transfer.rs
        image_to_gpu · image_to_cpu · wrap_gpu_buffer"]:::memory
    end

    W(["wgpu 28 : WGSL shaders : Vulkan · Metal · DX12 · WebGL"]):::wgpu

    U --> OI & OT
    OI & OT --> S
    S --> GT & MA & MP & MT
    GT & MA & MP & MT --> W

    classDef user    fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    classDef ops     fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef session fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef memory  fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef wgpu    fill:#F1EFE8,stroke:#5F5E5A,color:#444441
```

### Memory design

Every GPU buffer is acquired from a pool and automatically returned when the `GpuImage` or `GpuTensor` holding it is dropped. The return path is:

```
GpuImage drops
  → WgpuAllocator drops
  → Arc<PooledBufferGuard> hits zero
  → PooledBufferGuard::drop fires
  → BufferPool::release(buffer)      ← ready for the next op
```

`GpuImage` and `GpuTensor` are newtypes that expose only dimensional metadata. The inner `Image<T, C, WgpuAllocator>` and `Tensor<T, N, WgpuAllocator>` types are `pub(crate)` : external callers can never reach `.as_slice()` on a GPU-backed resource.

### Safety design

The entire crate is built on three guarantees:

1. **No accidental CPU access.** `GpuImage<T, C>` and `GpuTensor<T, N>` expose no `.as_slice()`. The compiler rejects it at the call site. There is no runtime check : it is structurally impossible.

2. **No unsafe byte reinterpretation.** All CPU↔GPU transfers go through `bytemuck::cast_slice` and `bytemuck::bytes_of`, which are fully safe functions. The `Pod` supertrait on `GpuPixel` is a compile-time proof that a type has no padding and every bit pattern is valid. If a type has implicit padding, deriving `Pod` fails at build time : the bug is caught before it can send garbage bytes to the shader.

3. **One auditable unsafe block.** `transfer.rs` contains a single `unsafe { from_raw_parts(...) }` call. Every other boundary crossing is safe Rust.

### Dispatch design

The 65535 per-dimension hardware limit on workgroup dispatch is handled automatically. For 2D image ops, a 2D workgroup grid is used. For 1D tensor ops over large flat buffers, the dispatch spills into Y when the X dimension would overflow. This is computed at runtime from the tensor size and workgroup geometry : the op author never thinks about it.

`var<immediate>` (push constants) is used to pass scalar parameters like image dimensions directly to shaders without allocating a uniform buffer. On platforms that don't support push constants, the implementation transparently falls back to small uniform buffers.

---

## Examples

### 1. Initialisation

All workflows begin by creating a shared `WgpuSession`.

```rust
use kornia_wgpu::session::WgpuSession;

let session = pollster::block_on(WgpuSession::new())?;
```

---

### 2. Image processing pipeline

Upload once, chain ops in VRAM, download once.

```rust
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_wgpu::{
    ops::image::resize::{resize_bilinear_f32, resize_nearest_f32},
    transfer::{image_to_cpu, image_to_gpu},
};

let gpu_src = image_to_gpu(&session, &cpu_src)?;

// Both ops run entirely in VRAM : zero PCIe transfers between them
let gpu_thumb    = resize_nearest_f32(&session, &gpu_src,   ImageSize { width: 4, height: 4 })?;
let gpu_model_in = resize_bilinear_f32(&session, &gpu_thumb, ImageSize { width: 3, height: 3 })?;

session.raw_device().poll(wgpu::PollType::wait_indefinitely());
let cpu_result = image_to_cpu(&session, &gpu_model_in)?;
```

`gpu_src`, `gpu_thumb`, and `gpu_model_in` are all `GpuImage<f32, 1>`. There is no `.as_slice()` on these types : calling it is a compile error.

---

### 3. Tensor math

Upload, chain ops in VRAM, download.

```rust
use kornia_tensor::{CpuAllocator, Tensor};
use kornia_wgpu::ops::tensor::elementwise::{add, relu};

let gpu_a = session.upload_tensor(&cpu_a)?;
let gpu_b = session.upload_tensor(&cpu_b)?;

// add result stays in VRAM : relu reads it directly
let gpu_sum  = add(&session, &gpu_a, &gpu_b)?;
let gpu_relu = relu(&session, &gpu_sum)?;

session.raw_device().poll(wgpu::PollType::wait_indefinitely());
let result = session.download_tensor(&gpu_relu)?;
```

---

### 4. Real-time video pipeline

Grab live 720p frames from an RTSP stream, upscale to 1080p on the GPU, and display them live via GStreamer `autovideosink`. The entire hot path stays on the GPU — three chained WGSL shaders with no CPU pixel math between them.

```
u8 frame from GStreamer
  → cast_u8_to_f32_gpu    packs bytes into u32, divides by 255 in shader   (PCIe upload)
  → resize_bilinear_f32   1280×720 → 1920×1080                              (VRAM only)
  → cast_f32_to_u8_gpu    pack4x8unorm: saturate → ×255 → round → pack      (VRAM only)
  → image_to_cpu          staged MAP_READ readback                           (PCIe download)
```

```rust
use gstreamer::prelude::*;
use kornia_image::ImageSize;
use kornia_io::{fps_counter::FpsCounter, gstreamer::StreamCapture};
use kornia_wgpu::{
    ops::image::cast::{cast_f32_to_u8_gpu, cast_u8_to_f32_gpu},
    ops::image::resize::resize_bilinear_f32,
    session::WgpuSession,
    transfer::image_to_cpu,
};
use std::sync::mpsc;
use std::thread;

const IN_W: usize = 1280;
const IN_H: usize = 720;
const OUT_W: usize = 1920;
const OUT_H: usize = 1080;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    gstreamer::init()?;

    let session = pollster::block_on(WgpuSession::new())?;

    let rtsp_url = std::env::args()
        .nth(1)
        .expect("Usage: video_pipeline <rtsp-url>");

    // --- Capture pipeline (RTSP → RGB appsink) ---
    let capture_desc = format!(
        "rtspsrc location={rtsp_url} protocols=tcp latency=200 ! decodebin ! videoconvert ! \
         video/x-raw,format=RGB ! appsink name=sink sync=false drop=true max-buffers=1"
    );
    let mut capture = StreamCapture::new(&capture_desc)?;
    capture.start()?;

    // --- Display pipeline (RGB appsrc → autovideosink) ---
    let display_desc = format!(
        "appsrc name=src format=time is-live=true block=true \
         caps=video/x-raw,format=RGB,width={OUT_W},height={OUT_H},framerate=30/1 ! \
         videoconvert ! autovideosink sync=false"
    );
    let display_pipeline = gstreamer::parse::launch(&display_desc)?
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast to Pipeline");

    let appsrc = display_pipeline
        .by_name("src")
        .expect("Failed to find appsrc")
        .dynamic_cast::<gstreamer_app::AppSrc>()
        .expect("Failed to cast to AppSrc");

    display_pipeline.set_state(gstreamer::State::Playing)?;

    // Dedicated thread: push downloaded frames to autovideosink.
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(5);
    let appsrc_clone = appsrc.clone();
    let display_thread = thread::spawn(move || {
        while let Ok((frame_idx, frame_data)) = rx.recv() {
            let mut buffer = gstreamer::Buffer::from_mut_slice(frame_data);
            let pts = gstreamer::ClockTime::from_mseconds(frame_idx as u64 * 33);
            buffer.get_mut().unwrap().set_pts(Some(pts));
            if appsrc_clone.push_buffer(buffer).is_err() {
                break;
            }
        }
    });

    let out_size = ImageSize { width: OUT_W, height: OUT_H };
    let mut fps_counter = FpsCounter::new();
    let mut frame_count = 0usize;

    loop {
        let Some(frame) = capture.grab_rgb8()? else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };

        fps_counter.update();

        // ── GPU stage 1: u8 → f32 ─────────────────────────────────────────────
        // Uploads raw u8 bytes as packed u32 words; WGSL shader unpacks and
        // divides by 255 — no CPU float math.
        let gpu_f32 = cast_u8_to_f32_gpu(&session, &frame)?;

        // ── GPU stage 2: bilinear resize 720p → 1080p ─────────────────────────
        // Reads gpu_f32 from VRAM, writes result back to VRAM.
        // Zero PCIe transfers between this op and its neighbours.
        let gpu_1080_f32 = resize_bilinear_f32(&session, &gpu_f32, out_size)?;

        // ── GPU stage 3: f32 → u8 ─────────────────────────────────────────────
        // Uses WGSL pack4x8unorm: saturate → ×255 → round → pack 4 bytes/u32.
        // Replaces the old CPU iterator entirely.
        let gpu_1080_u8 = cast_f32_to_u8_gpu(&session, &gpu_1080_f32)?;

        // ── Sync: wait for all three GPU stages to complete ───────────────────
        session.raw_device().poll(wgpu::PollType::wait_indefinitely());

        // ── PCIe download: GPU u8 → CPU Vec<u8> ──────────────────────────────
        // Maps a staging buffer and copies exactly OUT_W × OUT_H × 3 bytes.
        let cpu_out = image_to_cpu(&session, &gpu_1080_u8)?;

        // Hand off to display thread — no copy, no cast.
        if tx.send((frame_count, cpu_out.as_slice().to_vec())).is_err() {
            break;
        }

        frame_count += 1;
        if frame_count % 30 == 0 {
            println!("Frame {:4}  fps={:.1}", frame_count, fps_counter.fps());
        }

        // After the first frame, both the compute buffer (VRAM) and the staging
        // buffer (MAP_READ) are recycled from their pools :
        // zero GPU allocations per frame for fixed-size workloads.
    }

    drop(tx);
    let _ = display_thread.join();
    appsrc.end_of_stream()?;
    display_pipeline.set_state(gstreamer::State::Null)?;

    Ok(())
}
```

**Measured on an RTX 2050 (Vulkan backend):** the full round-trip — PCIe upload, three chained GPU ops (cast → resize → cast), PCIe download — completes in **9–10 ms per frame**, comfortably sustaining smooth 60 fps throughput after pool warmup.

A double-buffered async pipeline will be explored for the video node, overlapping PCIe upload of frame N+1 with GPU compute of frame N using `map_async` and a buffer ring drawn from the existing pool. This targets sustained throughput rather than single-frame latency.

```bash
cargo run --example video_pipeline --features gstreamer -- <rtsp-url>
```

---
Here is a clean, easy-to-follow "How to" section you can drop right into your `README.md`. It keeps the professional tone of your existing documentation while making the local webcam setup foolproof.

***

## Running the Examples

Most of the examples in this prototype are self-contained and can be run as-is using standard Cargo commands:

```bash
cargo run --example <example_name>
```

### Testing the Real-Time Video Pipeline

The `video_pipeline` example requires an active RTSP stream. For an easy local setup using your webcam, we recommend using [MediaMTX](https://github.com/bluenviron/mediamtx) to serve a low-latency H.264 stream.

**1. Set up MediaMTX**
Download the MediaMTX binary and place the provided `mediamtx.yml` configuration file in the same directory. This configuration automatically captures your local webcam (`/dev/video0`) using `ffmpeg` and serves it over RTSP. 

The relevant section in the provided `mediamtx.yml` is:
```yaml
paths:
  webcam:
    runOnInit: ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2M -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH
```
*[Note: You may need to change `/dev/video0` if your webcam is mounted elsewhere, or use a DirectShow/AVFoundation equivalent if you are on Windows/macOS.]*

**2. Start the stream**
Run the MediaMTX server with the config file:
```bash
./mediamtx mediamtx.yml
```
Your webcam stream will now be available at `rtsp://localhost:8554/webcam`.

**3. Run the pipeline**
Pass the local RTSP URL to the example. Make sure you have the `gstreamer` feature flag enabled:

```bash
cargo run --example video_pipeline --features gstreamer -- rtsp://localhost:8554/webcam
```
---

## Benchmarks

All measurements on RTX 4060 Laptop GPU. "GPU compute only" excludes PCIe transfer time. CPU: AMD Ryzen 7 8845HS (Zen 4, AVX2 : not a weak baseline).

| Operation | GPU compute only | CPU (kornia native) | Speedup |
|-----------|-----------------|---------------------|---------|
| bilinear resize 1ch 2160p→1080p | 272 µs | 6.46 ms | ~24× |
| bilinear resize 3ch 2160p→1080p | 632 µs | 9.74 ms | ~15× |
| bilinear resize 3ch 720p→1080p | 244 µs | 9.23 ms | ~38× |
| add [1024×1024] | 80 µs | 176 µs | ~2.2× |
| add 64× [1024×1024] sequential | 11.2 ms | 29.1 ms | ~2.6× |
| add flat [64×1024×1024] | 3.67 ms | 40.0 ms | ~10.9× |
| relu flat [64×1024×1024] | 2.55 ms | 29.8 ms | ~11.7× |
| add→relu chained flat [64×1024×1024] | 5.23 ms | 39.4 ms | ~7.5× |
| bilinear resize 3ch 2160p→1080p *(texture sampler, planned)* | ~100–200 µs est. | 9.74 ms | ~50–100× est. |

The 8845HS has a fast Zen 4 CPU with AVX2 SIMD : these are not weak baseline numbers. GPU advantage is most pronounced for large, memory-bandwidth-bound workloads and grows significantly with problem size. The sequential batched ops (~2.2–2.6×) expose kernel launch overhead from 64 separate dispatches with synchronisation between them. The flat batched ops (10–12×) show the real throughput advantage when the GPU's parallelism is fully utilised in a single dispatch.

> Texture sampler resize is a planned optimisation. Hardware bilinear units handle interpolation natively, typically yielding a 3–5× improvement over the current storage buffer shader for the same workload.

---

## vs CubeCL benchmarks

An equivalent bilinear resize was implemented in CubeCL (`WgpuRuntime`) using 16×16 workgroups and vec4 vectorisation : the maximum optimisation possible within CubeCL's API. Both implementations run on the same hardware via the same wgpu/Vulkan stack. This is a fair apples-to-apples comparison of hand-written WGSL vs CubeCL's generated WGSL : same GPU, same driver, same backend.

| Operation | kornia-wgpu | CubeCL (optimised) | speedup | tex. sampler (planned) |
|---|---|---|---|---|
| bilinear 1ch 2160p→1080p | **272 µs** | 703 µs | ~2.6× | ~100 µs est. |
| bilinear 3ch 2160p→1080p | **632 µs** | 1113 µs | ~1.8× | ~200 µs est. |
| bilinear 3ch 720p→1080p | **244 µs** | 801 µs | ~3.3× | ~100 µs est. |

> GPU compute only, no PCIe transfer. RTX 4060 Laptop GPU. The CubeCL implementation uses 16×16 workgroups and vec4 vectorisation : the maximum optimisation possible within CubeCL's API. The remaining 1.8–3.3× gap is structural: CubeCL's JIT IR layer adds dispatch overhead that cannot be eliminated, and its compute-only model has no access to hardware texture units (TMUs). Texture sampler resize (planned for kornia-wgpu) is expected to widen this gap further.

For elementwise ops like relu the gap narrows to ~1.3× (kornia-wgpu: 2.55 ms vs CubeCL: 3.40 ms at 64×1024×1024) since neither implementation has a spatial sampling advantage. The remaining difference is purely CubeCL's fixed JIT dispatch overhead. This confirms the gap is workload-dependent : largest where spatial sampling is involved, smallest for pure elementwise ops.

---
✅-already implemented

## Roadmap (GSoC deliverables)

### Core infrastructure (already implemented)
- `WgpuSession` : device, queue, pipeline cache
- `WgpuAllocator` + `PooledBufferGuard` : pool-backed GPU buffer lifecycle, automatic return on drop
- `BufferPool` + `StagingPoolMap` : size-class keyed buffer pools, zero alloc after warmup
- `GpuImage` / `GpuTensor` newtypes : compile-time GPU/CPU boundary enforcement
- `PipelineKey` cache : each shader compiled exactly once, O(1) lookup thereafter
- `bytemuck` transfers: Pod-guaranteed safe byte casts at every CPU↔GPU boundary

### Image operations (`ops::image`)
- Cast and scale: `cast_u8_to_f32_gpu`✅, `cast_f32_to_u8_gpu`✅ : GPU kernels, eliminate CPU cast bottleneck; `cast_u8_to_f32_gpu` bit-packs u8 into u32 to work around WGSL's lack of native u8 storage; `cast_f32_to_u8_gpu` uses WGSL `pack4x8unorm` to handle 4 pixels per thread
- Resize: nearest-neighbour✅, bilinear✅ (single and multi-channel)
- Grayscale
- Flip
- Normalize
- Filters: box, Gaussian, Sobel : separable filters will use shared memory tiling to avoid redundant global memory fetches
- `perspective_warp_gpu` : homography warp kernel contributed to `ops::image::warp`; key deliverable for the Bubbaloop bird's-eye view demo

### Tensor operations (`ops::tensor`)
- ✅ Elementwise math: `add`, `sub`, `mul`, `div` : vec4 vectorised
- ✅ Activations: `relu`, `exp`, `log`, `abs` : vec4 vectorised
- Reductions: `sum`, `mean`, `min`, `max` : two-pass parallel reduction for large tensors
- Stride-aware indexing for non-contiguous tensors (rank-4 NCHW) : fast-path dispatch when `is_contiguous()`, stride-aware WGSL variant otherwise
- Tiled matrix multiplication with `var<workgroup>` shared memory : improves arithmetic intensity, reduces global memory bandwidth

### Stretch goals
- Batched matrix multiplication (Z-dimension parallelism)
- Kernel fusion API: lazy `Expr` tree, WGSL codegen, `session.eval()` for elementwise chains : eliminates intermediate VRAM buffers for chained ops
- Texture-based image pipelines : hardware TMU access for resize, warp, and separable filters

---

## Project Timeline (12 Weeks)

### Community Bonding (May 1 – May 24)

- Align on API design, file structure, and Bubbaloop node integration model with Kornia maintainers and Bubbaloop team
- Finalise the REST API contract between the BEV node and Bubbaloop's pipeline manager
- Set up CI for GPU tests (Vulkan software renderer backend first, then coordinate Jetson Orin access with mentors)
- Review prototype code against kornia-rs conventions and incorporate mentor feedback on crate structure

Deliverable: architecture alignment document, CI pipeline, agreed file structure for kornia-wgpu.

---

### Weeks 1–2 : Bubbaloop Node Scaffold + Core Video Pipeline

The application is wired up first. Every op written in subsequent weeks lands directly into a running pipeline and can be tested end-to-end immediately.

Tasks:
- Implement `BevNode` struct with `WgpuSession` held for pipeline lifetime
- Wire GStreamer RTSP source to the node's frame handler using kornia-rs's existing capture utilities
- Integrate `cast_u8_to_f32_gpu` (already prototyped) and `resize_bilinear` (already prototyped) as the first two live pipeline stages
- End-to-end smoke test: RTSP stream → GPU cast + resize → CPU download (correctness, not yet 30 fps)
- Implement grayscale, flip, normalize image ops and wire normalize into the pipeline

Deliverables: functional Bubbaloop node (pipeline is live; BEV warp not yet present, but frames are flowing through GPU ops end-to-end); grayscale, flip, normalize contributed to kornia-imgproc.

---

### Weeks 3–4 : Perspective Warp + Full BEV Pipeline

Tasks:
- Implement `perspective_warp_gpu` kernel (`ops::image::warp`) with hardware TMU bilinear sampling via wgpu texture bindings
- Validate warp correctness against CPU reference implementation with numerical unit tests
- Wire `perspective_warp_gpu` into the BEV node: full upload → cast → resize → normalize → warp → download pipeline
- Test end-to-end BEV on development hardware (laptop/desktop GPU with Vulkan)
- Prototype the texture sampler pipeline for warp and resize (hardware TMU path)

Deliverables: `perspective_warp_gpu` contributed to kornia-wgpu under `ops::image::warp`; full BEV pipeline running on development hardware; texture pipeline prototype (TMU path for warp).

---

### Week 5 : Filter Kernels

Tasks:
- Implement box, Gaussian, Sobel filter kernels with WGSL compute shaders
- Optimise separable Gaussian with `var<workgroup>` shared memory (horizontal pass + vertical pass)
- Finalise and integrate texture-based sampling pipeline (hardware TMU path)
- Criterion benchmarks for all filter ops against kornia-imgproc CPU baseline

Deliverables: box, Gaussian, Sobel kernels contributed to kornia-imgproc; texture-based sampling pipeline integrated into kornia-wgpu; filter benchmark suite.

---

### Week 6 : Jetson Orin Integration + Bubbaloop Demo

Tasks:
- Deploy the full Bubbaloop BEV node on Jetson Orin
- Profile dispatch geometry on Jetson's Vulkan implementation; tune workgroup sizes if needed
- Record demo: single-camera bird's-eye view at 30 fps
- Run full-pipeline GPU vs CPU benchmark on Jetson hardware (upload → warp → download)
- Document Jetson-specific integration notes (unified memory behaviour, driver version, GStreamer pipeline string)

Deliverables: Bubbaloop BEV node running on Jetson Orin; demo video: 30 fps bird's-eye view; GPU vs CPU benchmark report (full pipeline latency).

Note: I do not personally own a Jetson Orin, but have access to similarly capable machines for all development and pre-deployment testing. I will coordinate with mentors for Jetson-side deployment and feedback. If the Jetson demo is not fully operational by midterm due to hardware access timing, I will complete it with carry-over time in Week 7 — but the full BEV pipeline will be running on development hardware (Vulkan/desktop GPU) by midterm without exception.

---

**Midterm Evaluation**

State at midterm: the Bubbaloop BEV node is live. `perspective_warp_gpu` is contributed to kornia-rs. The full BEV pipeline runs end-to-end with 2 PCIe crossings per frame. Filter kernels are contributed. The Jetson demo is either complete or in active deployment testing. The benchmark report exists.

---

### Weeks 7–8 : Tensor Reductions

Tasks:
- Implement `sum`, `mean`, `min`, `max` with parallel reduction strategy
- Two-pass reduction for large tensors (local reduce + global reduce)
- Handle edge cases: tensors not a power-of-two in size; non-contiguous inputs
- Unit tests and Criterion benchmarks against kornia-tensor CPU baseline

Deliverables: stable tensor reduction ops contributed to `ops::tensor`; GPU tensor reduction test suite.

---

### Weeks 9–10 : Non-Contiguous Tensor Support

Tasks:
- Implement `is_contiguous()` fast-path dispatch in session layer
- Stride-aware WGSL variant for non-contiguous tensors (up to rank-4 NCHW)
- Strides array passed via `var<immediate>` to avoid uniform buffer allocation per dispatch
- Tests covering transposed views, permuted channels, and sliced batches

Deliverables: contiguous fast-path and stride-aware non-contiguous path for all tensor ops; test suite covering transposed and permuted tensor views.

---

### Weeks 10–11 : Tiled Matrix Multiplication

Tasks:
- Implement tiled matmul kernel with `var<workgroup>` 16×16 shared memory tiles
- Collaborative tile load + `workgroupBarrier()` + partial dot product accumulation
- Criterion benchmarks at matrix sizes representative of kornia-rs vision workloads
- Batched matmul (Z-dimension parallelism) if time permits

Deliverables: tiled GPU matmul kernel contributed to `ops::tensor`; benchmark comparisons (GPU vs CPU, various matrix sizes).

---

### Week 12 : Finalisation

Tasks:
- Cross-platform testing: Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows)
- Final edge case audit: non-contiguous tensors, power-of-two buffer alignment, platform-specific dispatch limits
- Complete documentation for all contributed ops and the Bubbaloop node
- Final GSoC report

Deliverables: final documentation for kornia-wgpu and all contributed ops; Bubbaloop BEV demo video (public); final GSoC report.

Each phase includes continuous benchmarking against CPU implementations to ensure measurable performance improvements.

---

## Availability
- I happen to have summer break for almost the entirety of the coding period, so I can give **35 hours** per week of time to my project.

- I will be online on Discord daily, weekly meetings are also good.

- I can communicate on any platform preferred by the maintainers (Discord, Github, Mail, Slack, etc.)

- Timezone: **UTC+5:30**. But really you can reach me at almost any time of the day, I'll try to respond as quickly as possible.

---

## About Me

**Name:** Neelabhro Ghosh
**University:** Atal Bihari Vajpayee IIITM, Gwalior, India
**Degree:** B.Tech in Mathematics and Scientific Computing (2024–2028)
**Timezone:** Indian Standard Time (UTC+5:30)
**GitHub:** [@Nyx128](https://github.com/Nyx128)

I am an undergraduate student with a focus on high-performance computing, numerical analysis, and GPU programming. I have been actively contributing to the Kornia ecosystem and have a deep interest in bringing hardware-accelerated computer vision to Rust.

### Prior Contributions to Kornia-RS

I am already familiar with Kornia-RS's codebase, performance standards, and review process. I have successfully implemented several high-performance features using Rust, SIMD, and Rayon:

- [PR #654](https://github.com/kornia/kornia-rs/pull/654) : Pyramid Operations Optimization: implemented u8 pyramid operations using SIMD vectorization and Rayon, achieving 300–700% throughput improvement.
- [PR #685](https://github.com/kornia/kornia-rs/pull/685) : Draw Line Performance Improvements: implemented a SIMD-friendly Bresenham line algorithm, reducing algorithmic complexity from O(LT²) to O(LT).
- [PR #711](https://github.com/kornia/kornia-rs/pull/711) : AprilTag Algebra Refactor: refactored homography and Cholesky solvers using algebraic types, yielding a ~10% benchmark performance improvement.

### Relevant GPU and Graphics Experience

Beyond CPU-bound SIMD optimizations, my core technical expertise lies in GPU compute and rendering pipelines:

- **Yald (Rust & wgpu):** I authored a Rust GPU compute dispatch library built on wgpu, specifically targeting the reduction of GPU kernel dispatch boilerplate through a lightweight abstraction layer : directly mirroring the architectural goals of kornia-wgpu.
- **Graphics Programming:** I have built a Deferred Renderer in C++/OpenGL using custom GLSL shaders for real-time dynamic lighting, and I am proficient with low-level GPU APIs including Vulkan, CUDA, and WebGPU.

Between my background in Mathematics and Scientific Computing, my prior PRs optimizing Kornia-RS, and my domain expertise in wgpu compute pipelines, I look forward to delivering this project in the coming summer. I am actively discussing this proposal with Kornia maintainers to ensure alignment with project goals.

---

#### Skillsets I have that might come in handy here:
- RenderDoc and NVIDIA Nsight for GPU pipeline inspection, shader debugging, and dispatch-level bottleneck analysis

- Comfortable reading `x86 assembly` — useful for verifying SIMD codegen and compiler output rather than trusting it blindly

- Experience with Rust's `perf`, `cargo-flamegraph`, and `Criterion` for CPU-side profiling and regression tracking


Authored by Neelabhro Ghosh ([@Nyx128](https://github.com/Nyx128)) ;)
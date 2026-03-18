//! examples/video_pipeline.rs
//!
//! Live video pipeline: grab 720p frames from an RTSP stream,
//! upscale each frame to 1080p on the GPU using bilinear resize, and
//! report per-frame timing.
//!
//! Pipeline per frame:
//!   RTSP 720p RGB frame (u8)
//!     → CPU cast to f32 + normalize [0,1]   (one intermediate alloc — see note)
//!     → GPU upload  (PCIe crossing #1)
//!     → bilinear resize 1280x720 → 1920x1080  (GPU, stays in VRAM)
//!     → GPU download  (PCIe crossing #2)
//!
//! Run with:
//!   cargo run --example video_pipeline --features gstreamer -- rtsp://user:pass@ip:port/stream

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_io::{fps_counter::FpsCounter, stream::StreamCapture};
use kornia_wgpu::{
    ops::image::resize::resize_bilinear_f32,
    session::WgpuSession,
    transfer::{image_to_cpu, image_to_gpu},
};

const IN_W: usize = 1280;
const IN_H: usize = 720;
const OUT_W: usize = 1920;
const OUT_H: usize = 1080;
const FRAME_DEBUG: usize = 30;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── GPU session ───────────────────────────────────────────────────────────
    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.");
    println!("Pipeline: {IN_W}×{IN_H} 720p → {OUT_W}×{OUT_H} 1080p (bilinear, GPU)\n");

    // ── RTSP source ───────────────────────────────────────────────────────────
    let rtsp_url = std::env::args()
        .nth(1)
        .expect("Usage: video_upscale_720p_to_1080p <rtsp-url>\n  e.g. rtsp://user:pass@192.168.1.10:554/stream");

    let pipeline_desc = format!(
        "rtspsrc location={rtsp_url} latency=0 ! rtph265depay ! avdec_h265 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
    );

    let mut capture = StreamCapture::new(&pipeline_desc)?;
    capture.start()?;
    println!("RTSP H.265 stream started: {rtsp_url}\n");

    let out_size = ImageSize {
        width: OUT_W,
        height: OUT_H,
    };

    // ── Timing accumulators ───────────────────────────────────────────────────
    let mut fps_counter = FpsCounter::new();
    let mut frame_count = 0usize;
    let mut total_cast_us = 0u128;
    let mut total_upload_us = 0u128;
    let mut total_gpu_us = 0u128;
    let mut total_download_us = 0u128;

    // ── Frame loop ────────────────────────────────────────────────────────────
    loop {
        let Some(frame) = capture.grab_rgb8()? else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };

        frame_count += 1;
        fps_counter.update();

        // ── Step 1: CPU cast u8 → f32, normalize to [0, 1] ───────────────────
        //
        // GStreamer gives us Image<u8, 3, GstAllocator>.
        // resize_bilinear_f32 expects f32, so we cast on CPU for now.
        //
        // Future: a cast_and_scale GPU kernel uploads u8 directly and divides
        // by 255 in the shader, eliminating this intermediate allocation and
        // reducing the pipeline to a single PCIe crossing per frame.
        let t0 = std::time::Instant::now();
        let cpu_f32 = cast_u8_to_f32(&frame)?;
        total_cast_us += t0.elapsed().as_micros();

        // ── Step 2: Upload to GPU (PCIe crossing #1) ──────────────────────────
        let t1 = std::time::Instant::now();
        let gpu_in = image_to_gpu(&session, &cpu_f32)?;
        total_upload_us += t1.elapsed().as_micros();

        // ── Step 3: Bilinear upscale 720p → 1080p (stays in VRAM) ────────────
        let t2 = std::time::Instant::now();
        let gpu_out = resize_bilinear_f32(&session, &gpu_in, out_size)?;
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        total_gpu_us += t2.elapsed().as_micros();

        // ── Step 4: Download to CPU (PCIe crossing #2) ───────────────────────
        let t3 = std::time::Instant::now();
        let _cpu_out = image_to_cpu(&session, &gpu_out)?;
        total_download_us += t3.elapsed().as_micros();

        // ── Per-frame report every 10 frames ─────────────────────────────────
        if frame_count % FRAME_DEBUG == 0 {
            let n = FRAME_DEBUG as u128;
            println!(
                "Frame {:3}  fps={:5.1}  cast={:4}µs  upload={:4}µs  gpu={:3}µs  download={:4}µs  total={:4}µs",
                frame_count,
                fps_counter.fps(),
                total_cast_us / n,
                total_upload_us / n,
                total_gpu_us / n,
                total_download_us / n,
                (total_cast_us + total_upload_us + total_gpu_us + total_download_us) / n,
            );
            total_cast_us = 0;
            total_upload_us = 0;
            total_gpu_us = 0;
            total_download_us = 0;
        }
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn cast_u8_to_f32(
    frame: &kornia_image::Image<u8, 3, kornia_io::gstreamer::GstAllocator>,
) -> Result<Image<f32, 3, CpuAllocator>, Box<dyn std::error::Error>> {
    let data: Vec<f32> = frame.as_slice().iter().map(|&v| v as f32 / 255.0).collect();
    Ok(Image::new(frame.size(), data, CpuAllocator)?)
}

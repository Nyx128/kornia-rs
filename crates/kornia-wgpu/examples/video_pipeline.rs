//! examples/video_pipeline.rs
//!
//! Live video pipeline: grab 720p frames from an RTSP stream,
//! upscale each frame to 1080p on the GPU using bilinear resize,
//! and display them in a live window using GStreamer autovideosink.
//!
//! The entire hot path runs on the GPU:
//!   u8 frame  →  cast_u8_to_f32_gpu  →  resize_bilinear_f32  →  cast_f32_to_u8_gpu  →  download
//!
//! There is no CPU pixel math in the frame loop. The only CPU work is the
//! PCIe upload (write_buffer) and PCIe download (map_async), plus the
//! mpsc send to the display thread.

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
    // 1. Initialize GStreamer
    gstreamer::init()?;

    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.");
    println!("Pipeline: {IN_W}x{IN_H} 720p -> {OUT_W}x{OUT_H} 1080p\n");

    let rtsp_url = std::env::args()
        .nth(1)
        .expect("Usage: video_pipeline <rtsp-url>");

    // --- Capture Pipeline ---
    let capture_desc = format!(
        "rtspsrc location={rtsp_url} protocols=tcp latency=200 ! decodebin ! videoconvert ! \
         video/x-raw,format=RGB ! appsink name=sink sync=false drop=true max-buffers=1"
    );
    let mut capture = StreamCapture::new(&capture_desc)?;
    capture.start()?;
    println!("RTSP stream started: {rtsp_url}\n");

    // --- Display Pipeline ---
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
    println!("Live playback started. Press Ctrl+C to stop.");

    // --- Thread Setup ---
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(5);
    let appsrc_clone = appsrc.clone();

    // Dedicated thread for pushing buffers to GStreamer autovideosink.
    let display_thread = thread::spawn(move || {
        while let Ok((frame_idx, frame_data)) = rx.recv() {
            let mut buffer = gstreamer::Buffer::from_mut_slice(frame_data);

            let pts = gstreamer::ClockTime::from_mseconds(frame_idx as u64 * 33);
            buffer.get_mut().unwrap().set_pts(Some(pts));

            if appsrc_clone.push_buffer(buffer).is_err() {
                eprintln!(
                    "Failed to push buffer to GStreamer display. Window might have been closed."
                );
                break;
            }
        }
    });

    let out_size = ImageSize {
        width: OUT_W,
        height: OUT_H,
    };

    let mut fps_counter = FpsCounter::new();
    let mut frame_count = 0usize;

    // Timing accumulators (reset every 30 frames)
    let mut total_process_us = 0u128;
    let mut total_cast_up_us = 0u128;
    let mut total_resize_us = 0u128;
    let mut total_cast_down_us = 0u128;
    let mut total_download_us = 0u128;

    loop {
        let Some(frame) = capture.grab_rgb8()? else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };

        fps_counter.update();
        let t_process_start = std::time::Instant::now();

        // ── GPU stage 1: u8 → f32 (upload + unpack shader) ───────────────────
        //
        // cast_u8_to_f32_gpu uploads the raw u8 bytes as packed u32 words and
        // divides by 255 in the shader — no CPU float math.
        let t0 = std::time::Instant::now();
        let gpu_f32 = cast_u8_to_f32_gpu(&session, &frame)?;
        total_cast_up_us += t0.elapsed().as_micros();

        // ── GPU stage 2: bilinear resize 720p → 1080p ─────────────────────────
        //
        // Reads gpu_f32 from VRAM and writes the 1080p result back to VRAM.
        // Zero PCIe transfers between this op and the ones on either side.
        let t1 = std::time::Instant::now();
        let gpu_1080_f32 = resize_bilinear_f32(&session, &gpu_f32, out_size)?;
        total_resize_us += t1.elapsed().as_micros();

        // ── GPU stage 3: f32 → u8 (pack4x8unorm shader) ──────────────────────
        //
        // Each thread handles 4 pixels using the WGSL pack4x8unorm built-in:
        //   saturate to [0,1] → ×255 → round → pack 4 bytes into 1 u32
        // Replaces the old CPU iterator: `pixels.map(|v| (v*255).clamp() as u8)`
        let t2 = std::time::Instant::now();
        let gpu_1080_u8 = cast_f32_to_u8_gpu(&session, &gpu_1080_f32)?;
        total_cast_down_us += t2.elapsed().as_micros();

        // ── Sync: wait for all three GPU stages to finish ─────────────────────
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        // ── PCIe download: GPU u8 → CPU Vec<u8> ──────────────────────────────
        //
        // image_to_cpu maps a staging buffer and copies exactly
        // OUT_W * OUT_H * 3 bytes — no conversion, no arithmetic.
        let t3 = std::time::Instant::now();
        let cpu_out = image_to_cpu(&session, &gpu_1080_u8)?;
        total_download_us += t3.elapsed().as_micros();

        total_process_us += t_process_start.elapsed().as_micros();

        // Hand off the raw bytes directly — no copy, no cast.
        let u8_data = cpu_out.as_slice().to_vec();
        if tx.send((frame_count, u8_data)).is_err() {
            println!("Display window closed or thread disconnected. Exiting...");
            break;
        }

        frame_count += 1;

        if frame_count % 30 == 0 {
            let n = 30u128;
            println!(
                "Frame {:4}  fps={:5.1}  \
                 total={:5}µs | cast_up={:4}µs  resize={:4}µs  cast_down={:4}µs  download={:4}µs",
                frame_count,
                fps_counter.fps(),
                total_process_us / n,
                total_cast_up_us / n,
                total_resize_us / n,
                total_cast_down_us / n,
                total_download_us / n,
            );
            total_process_us = 0;
            total_cast_up_us = 0;
            total_resize_us = 0;
            total_cast_down_us = 0;
            total_download_us = 0;
        }
    }

    // --- Graceful Teardown ---
    drop(tx);
    let _ = display_thread.join();
    appsrc.end_of_stream()?;
    display_pipeline.set_state(gstreamer::State::Null)?;
    println!("Shutdown complete.");

    Ok(())
}
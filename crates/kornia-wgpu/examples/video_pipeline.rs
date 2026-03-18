//! examples/video_pipeline.rs
//!
//! Live video pipeline: grab 720p H.265 frames from an RTSP stream,
//! upscale each frame to 1080p on the GPU using bilinear resize.
//!
//! Pipeline per frame:
//!   RTSP H.265 720p RGB frame (u8, GstAllocator)
//!     → cast_u8_to_f32_gpu   (GPU kernel, no intermediate Vec<f32>)
//!     → resize_bilinear_f32  1280x720 → 1920x1080
//!     → image_to_cpu         (PCIe download)
//!
//! Only ONE PCIe upload crossing per frame — the u8 bytes go straight
//! into the cast kernel's input buffer and never touch the CPU as f32.
//!
//! Run with:
//!   cargo run --example video_pipeline --features gstreamer -- <rtsp-url>

use kornia_image::ImageSize;
use kornia_io::{fps_counter::FpsCounter, gstreamer::StreamCapture};
use kornia_wgpu::{
    ops::image::cast::cast_u8_to_f32_gpu, ops::image::resize::resize_bilinear_f32,
    session::WgpuSession, transfer::image_to_cpu,
};

const IN_W: usize = 1280;
const IN_H: usize = 720;
const OUT_W: usize = 1920;
const OUT_H: usize = 1080;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.");
    println!("Pipeline: {IN_W}x{IN_H} 720p -> {OUT_W}x{OUT_H} 1080p\n");

    let rtsp_url = std::env::args()
        .nth(1)
        .expect("Usage: video_upscale_720p_to_1080p <rtsp-url>");

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

    let mut fps_counter = FpsCounter::new();
    let mut frame_count = 0usize;
    let mut total_cast_us = 0u128;
    let mut total_resize_us = 0u128;
    let mut total_download_us = 0u128;

    loop {
        let Some(frame) = capture.grab_rgb8()? else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };

        frame_count += 1;
        fps_counter.update();

        // The u8 bytes from GstAllocator go straight into a u32 storage buffer
        // on the GPU. The shader unpacks each byte and divides by 255.
        // Zero intermediate Vec<f32> on the CPU.
        let t0 = std::time::Instant::now();
        let gpu_f32 = cast_u8_to_f32_gpu(&session, &frame)?;
        total_cast_us += t0.elapsed().as_micros();

        //Bilinear upscale 720p → 1080p (chained in VRAM)
        let t1 = std::time::Instant::now();
        let gpu_out = resize_bilinear_f32(&session, &gpu_f32, out_size)?;
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        total_resize_us += t1.elapsed().as_micros();

        //Download the final 1080p f32 image back to CPU
        let t2 = std::time::Instant::now();
        let _cpu_out = image_to_cpu(&session, &gpu_out)?;
        total_download_us += t2.elapsed().as_micros();

        if frame_count % 10 == 0 {
            let n = 10u128;
            println!(
                "Frame {:3}  fps={:5.1}  cast={:4}us  resize={:3}us  download={:4}us  total={:4}us",
                frame_count,
                fps_counter.fps(),
                total_cast_us / n,
                total_resize_us / n,
                total_download_us / n,
                (total_cast_us + total_resize_us + total_download_us) / n,
            );
            total_cast_us = 0;
            total_resize_us = 0;
            total_download_us = 0;
        }
    }
}

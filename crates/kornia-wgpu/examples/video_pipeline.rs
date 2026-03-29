//! examples/video_pipeline.rs
//!
//! Live video pipeline: grab 720p frames from an RTSP stream,
//! upscale each frame to 1080p on the GPU using bilinear resize,
//! and display them in a live window using GStreamer autovideosink.

use gstreamer::prelude::*;
use kornia_image::ImageSize;
use kornia_io::{fps_counter::FpsCounter, gstreamer::StreamCapture};
use kornia_wgpu::{
    ops::image::cast::cast_u8_to_f32_gpu, ops::image::resize::resize_bilinear_f32,
    session::WgpuSession, transfer::image_to_cpu,
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
        .expect("Usage: video_upscale_720p_to_1080p <rtsp-url>");

    // --- Capture Pipeline ---
    let capture_desc = format!(
        "rtspsrc location={rtsp_url} protocols=tcp latency=200 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink sync=false drop=true max-buffers=1"
    );
    let mut capture = StreamCapture::new(&capture_desc)?;
    capture.start()?;
    println!("RTSP stream started: {rtsp_url}\n");

    // --- Display Pipeline ---
    let display_desc = format!(
        "appsrc name=src format=time is-live=true block=true \
         caps=video/x-raw,format=RGB,width={},height={},framerate=30/1 ! \
         videoconvert ! autovideosink sync=false",
        OUT_W, OUT_H
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

    // Spawn the dedicated background thread for pushing buffers to the screen
    let display_thread = thread::spawn(move || {
        while let Ok((frame_idx, frame_data)) = rx.recv() {
            let mut buffer = gstreamer::Buffer::from_mut_slice(frame_data);
            
            let pts = gstreamer::ClockTime::from_mseconds(frame_idx as u64 * 33);
            buffer.get_mut().unwrap().set_pts(Some(pts));
            
            if appsrc_clone.push_buffer(buffer).is_err() {
                eprintln!("Failed to push buffer to GStreamer display. Window might have been closed.");
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
    let mut total_process_us = 0u128;
    let mut total_cast_us = 0u128;
    let mut total_resize_us = 0u128;
    let mut total_download_us = 0u128;
    let mut total_convert_us = 0u128;

    loop {
        let Some(frame) = capture.grab_rgb8()? else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };

        fps_counter.update();

        // Start total processing timer here
        let t_process_start = std::time::Instant::now();

        // GPU Processing
        let t0 = std::time::Instant::now();
        let gpu_f32 = cast_u8_to_f32_gpu(&session, &frame)?;
        total_cast_us += t0.elapsed().as_micros();

        let t1 = std::time::Instant::now();
        let gpu_out = resize_bilinear_f32(&session, &gpu_f32, out_size)?;
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());
        total_resize_us += t1.elapsed().as_micros();

        let t2 = std::time::Instant::now();
        let cpu_out = image_to_cpu(&session, &gpu_out)?;
        total_download_us += t2.elapsed().as_micros();

        // Track the CPU float-to-u8 conversion time
        let t3 = std::time::Instant::now();
        let u8_data: Vec<u8> = cpu_out
            .as_slice()
            .iter()
            .map(|&val| (val * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        total_convert_us += t3.elapsed().as_micros();

        // Stop total processing timer
        total_process_us += t_process_start.elapsed().as_micros();

        // Send to the display thread
        if tx.send((frame_count, u8_data)).is_err() {
            println!("Display window closed or thread disconnected. Exiting...");
            break;
        }

        frame_count += 1;

        if frame_count % 30 == 0 {
            let n = 30u128;
            println!(
                "Frame {:4}  fps={:5.1}  total={:5}us | cast={:4}us  resize={:4}us  download={:4}us  convert={:5}us",
                frame_count,
                fps_counter.fps(),
                total_process_us / n,
                total_cast_us / n,
                total_resize_us / n,
                total_download_us / n,
                total_convert_us / n,
            );
            total_process_us = 0;
            total_cast_us = 0;
            total_resize_us = 0;
            total_download_us = 0;
            total_convert_us = 0;
        }
    }

    // --- Graceful Teardown ---
    drop(tx);
    let _ = display_thread.join();
    appsrc.end_of_stream()?;
    
    // Shut down the pipeline safely
    display_pipeline.set_state(gstreamer::State::Null)?;
    println!("Shutdown complete.");

    Ok(())
}
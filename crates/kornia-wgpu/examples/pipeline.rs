//! examples/pipeline.rs
//!
//! Shows image and tensor ops composed in a single GPU session —
//! the kind of workflow that appears in vision model pre/post-processing.
//!
//! Two independent pipelines are demonstrated:
//!
//!   Image pipeline:
//!     CPU image → upload → nearest resize → bilinear resize → download
//!
//!   Tensor pipeline:
//!     CPU tensors → upload → add → relu → download
//!
//! Both share the same WgpuSession (same device, same pipeline cache).
//!
//! Run with:
//!   cargo run --example pipeline

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_tensor::{CpuAllocator as TensorCpuAlloc, Tensor};
use kornia_wgpu::{
    ops::image::resize::{resize_bilinear_f32, resize_nearest_f32},
    ops::tensor::elementwise::{add, relu},
    session::WgpuSession,
    transfer::{image_to_cpu, image_to_gpu},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One session shared across both pipelines.
    // The pipeline cache inside WgpuSession means each WGSL shader is
    // compiled exactly once no matter how many times you call an op.
    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.\n");

    image_pipeline(&session)?;
    tensor_pipeline(&session)?;

    println!("\nAll done.");
    Ok(())
}

// ── Image pipeline ────────────────────────────────────────────────────────────
//
// Simulates a pre-processing step: take a high-res input, downsample to
// a thumbnail (nearest), then to a precise model input size (bilinear).
// The two GPU ops are chained without any CPU round-trip between them.
fn image_pipeline(session: &WgpuSession) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Image pipeline ===");

    // 8x8 source with a simple gradient pattern
    let src_size = ImageSize {
        width: 8,
        height: 8,
    };
    let cpu_src = Image::<f32, 1, _>::new(
        src_size,
        (0..64).map(|i| i as f32 * 0.01).collect(),
        CpuAllocator,
    )?;

    println!("Input {}x{}:", src_size.width, src_size.height);
    print_image_rows(cpu_src.as_slice(), src_size.width);

    // One upload crossing the PCIe bus
    let gpu_src = image_to_gpu(session, &cpu_src)?;

    // Op 1: nearest resize 8x8 → 4x4
    let thumb_size = ImageSize {
        width: 4,
        height: 4,
    };
    let gpu_thumb = resize_nearest_f32(session, &gpu_src, thumb_size)?;

    // Op 2: bilinear resize 4x4 → 3x3
    // gpu_thumb is entirely in VRAM — no PCIe transfer between these two ops.
    let model_size = ImageSize {
        width: 3,
        height: 3,
    };
    let gpu_model_in = resize_bilinear_f32(session, &gpu_thumb, model_size)?;

    // Sync then one download crossing the PCIe bus
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());
    let cpu_result = image_to_cpu(session, &gpu_model_in)?;

    println!(
        "\nOutput {}x{}  (nearest 8→4, bilinear 4→3, zero PCIe transfers between ops):",
        model_size.width, model_size.height
    );
    print_image_rows(cpu_result.as_slice(), model_size.width);

    Ok(())
}

// ── Tensor pipeline ───────────────────────────────────────────────────────────
//
// Simulates post-processing: element-wise add two feature maps, then
// apply ReLU.  The add output lives in VRAM and feeds directly into relu.
fn tensor_pipeline(session: &WgpuSession) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Tensor pipeline ===");

    let shape = [2, 6];

    let cpu_a = Tensor::from_shape_vec(
        shape,
        vec![
            -3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ],
        TensorCpuAlloc,
    )?;
    let cpu_b = Tensor::from_shape_vec(
        shape,
        vec![
            1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
        ],
        TensorCpuAlloc,
    )?;

    println!("a        = {:?}", cpu_a.as_slice());
    println!("b        = {:?}", cpu_b.as_slice());

    let gpu_a = session.upload_tensor(&cpu_a)?;
    let gpu_b = session.upload_tensor(&cpu_b)?;

    // Op 1: add — result stays in VRAM
    let gpu_sum = add(session, &gpu_a, &gpu_b)?;

    // Op 2: relu — reads gpu_sum from VRAM, no CPU round-trip
    let gpu_activated = relu(session, &gpu_sum)?;

    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());

    let cpu_sum = session.download_tensor(&gpu_sum)?;
    let cpu_activated = session.download_tensor(&gpu_activated)?;

    println!("a + b    = {:?}", cpu_sum.as_slice());
    println!("relu = {:?}", cpu_activated.as_slice());

    assert!(
        cpu_activated.as_slice().iter().all(|&v| v >= 0.0),
        "relu must produce non-negative values"
    );
    println!("All relu outputs >= 0.0 as expected.");

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_image_rows(data: &[f32], width: usize) {
    let rows = data.len() / width;
    for r in 0..rows {
        let row: Vec<String> = data[r * width..(r + 1) * width]
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect();
        println!("  row {}: [{}]", r, row.join(", "));
    }
}

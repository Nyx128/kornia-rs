//Demonstrates GPU-accelerated image operations using kornia-wgpu.

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_wgpu::{
    ops::image::resize::{resize_bilinear_f32, resize_nearest_f32},
    session::WgpuSession,
    transfer::{image_to_cpu, image_to_gpu},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Initialise GPU session ─────────────────────────────────────────────
    //
    // WgpuSession owns the wgpu Device + Queue and the pipeline cache.
    // Creating it once and reusing it is important — pipeline compilation
    // only happens on the first dispatch for each (op, type) pair.
    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.");

    // ── 2. Build a small CPU image ────────────────────────────────────────────
    //
    // A 4x4 single-channel f32 image with a simple gradient pattern so that
    // the resize output is easy to reason about manually.
    //
    //   0.0  0.1  0.2  0.3
    //   0.4  0.5  0.6  0.7
    //   0.8  0.9  1.0  1.1   ← values > 1.0 are fine for f32
    //   1.2  1.3  1.4  1.5
    let src_size = ImageSize {
        width: 4,
        height: 4,
    };
    let src_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let cpu_src = Image::<f32, 1, _>::new(src_size, src_data, CpuAllocator)?;

    println!(
        "Input image  {}x{} (1 channel f32)",
        src_size.width, src_size.height
    );
    print_image_grid(&cpu_src);

    // ── 3. Upload to GPU ──────────────────────────────────────────────────────
    //
    // image_to_gpu copies the CPU slice into a wgpu::Buffer via
    // bytemuck::cast_slice — no unsafe, Pod-guaranteed.
    // The returned Image<f32, 1, WgpuAllocator> wraps that buffer.
    let gpu_src = image_to_gpu(&session, &cpu_src)?;

    // ── 4a. Nearest-neighbour resize  (4x4 → 8x8) ───────────────────────────
    let nn_size = ImageSize {
        width: 8,
        height: 8,
    };
    let gpu_nn = resize_nearest_f32(&session, &gpu_src, nn_size)?;

    // Synchronise before download so we measure compute time separately
    // from the transfer.  In a pipeline you can chain ops without syncing.
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());

    let cpu_nn = image_to_cpu(&session, &gpu_nn)?;
    println!(
        "\nNearest-neighbour resize  {}x{} → {}x{}",
        src_size.width, src_size.height, nn_size.width, nn_size.height
    );
    print_image_grid(&cpu_nn);

    // ── 4b. Bilinear resize  (4x4 → 2x2 downscale) ───────────────────────────
    //
    // Bilinear uses centre-aligned coordinates, so downscaling reads a
    // weighted average of the four surrounding source pixels.
    let bl_size = ImageSize {
        width: 2,
        height: 2,
    };
    let gpu_bl = resize_bilinear_f32(&session, &gpu_src, bl_size)?;
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());

    let cpu_bl = image_to_cpu(&session, &gpu_bl)?;
    println!(
        "\nBilinear resize            {}x{} → {}x{}",
        src_size.width, src_size.height, bl_size.width, bl_size.height
    );
    print_image_grid(&cpu_bl);

    // ── 5. GPU op chaining (no CPU round-trip between ops) ───────────────────
    //
    // resize_nearest_f32 returns a WgpuAllocator image whose wgpu::Buffer
    // lives entirely in VRAM.  Passing it directly into resize_bilinear_f32
    // never touches the PCIe bus — the GPU reads its own output as the next
    // op's input.
    println!("\nChained: nearest 4x4→8x8, then bilinear 8x8→3x3");
    let gpu_chained_a = resize_nearest_f32(
        &session,
        &gpu_src,
        ImageSize {
            width: 8,
            height: 8,
        },
    )?;
    let gpu_chained_b = resize_bilinear_f32(
        &session,
        &gpu_chained_a,
        ImageSize {
            width: 3,
            height: 3,
        },
    )?;
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());

    let cpu_chained = image_to_cpu(&session, &gpu_chained_b)?;
    println!("Result (3x3):");
    print_image_grid(&cpu_chained);

    println!("\nDone.");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_image_grid(img: &Image<f32, 1, CpuAllocator>) {
    let w = img.size().width;
    let h = img.size().height;
    for row in 0..h {
        let row_vals: Vec<String> = (0..w)
            .map(|col| format!("{:.2}", img.as_slice()[row * w + col]))
            .collect();
        println!("  [{}]", row_vals.join("  "));
    }
}

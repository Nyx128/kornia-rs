//! examples/tensor_ops.rs
//!
//! Demonstrates GPU-accelerated tensor operations using kornia-wgpu.
//!
//! Run with:
//!   cargo run --example tensor_ops

use kornia_tensor::{CpuAllocator, Tensor};
use kornia_wgpu::{
    ops::tensor::elementwise::{add, relu},
    session::WgpuSession,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Initialise GPU session ─────────────────────────────────────────────
    let session = pollster::block_on(WgpuSession::new())?;
    println!("GPU session ready.\n");

    // ── 2. Binary op: add ────────────────────────────────────────────────────
    //
    // CPU tensors upload via session.upload_tensor(), which uses
    // bytemuck::cast_slice for a safe byte cast before calling
    // queue.write_buffer().
    let shape = [2, 4];
    let cpu_a = Tensor::from_shape_vec(
        shape,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        CpuAllocator,
    )?;
    let cpu_b = Tensor::from_shape_vec(
        shape,
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        CpuAllocator,
    )?;

    println!("a      = {:?}", cpu_a.as_slice());
    println!("b      = {:?}", cpu_b.as_slice());

    let gpu_a = session.upload_tensor(&cpu_a)?;
    let gpu_b = session.upload_tensor(&cpu_b)?;

    // add dispatches a WGSL shader with op_kind=0.
    // The result stays in VRAM as a WgpuAllocator tensor.
    let gpu_sum = add(&session, &gpu_a, &gpu_b)?;
    let cpu_sum = session.download_tensor(&gpu_sum)?;

    println!("a + b  = {:?}", cpu_sum.as_slice());

    // ── 3. Unary op: relu ─────────────────────────────────────────────────────
    //
    // For unary ops, the shader still has two input bindings (a, b) but we
    // bind the same buffer twice — b is simply ignored.  This avoids a
    // separate shader variant just for unary ops.
    let cpu_mixed = Tensor::from_shape_vec(
        [8],
        vec![-4.0f32, 3.0, -1.0, 0.0, 2.5, -0.5, 7.0, -10.0],
        CpuAllocator,
    )?;
    println!("\ninput  = {:?}", cpu_mixed.as_slice());

    let gpu_mixed = session.upload_tensor(&cpu_mixed)?;
    let gpu_relu = relu(&session, &gpu_mixed)?;
    let cpu_relu = session.download_tensor(&gpu_relu)?;

    println!("relu   = {:?}", cpu_relu.as_slice());

    // ── 4. Chained ops (zero CPU round-trips between ops) ────────────────────
    //
    // gpu_sum is still in VRAM from step 2.  Passing it straight into relu
    // means the GPU reads its own previous output — no PCIe transfer.
    //
    // Chain: a + b → relu → result
    println!("\nChained a + b → relu:");
    let gpu_chained = relu(&session, &gpu_sum)?;
    let cpu_chained = session.download_tensor(&gpu_chained)?;
    println!("result = {:?}", cpu_chained.as_slice());
    // All values from (a+b) were positive so relu is a no-op here,
    // confirming the chain produced the same values as cpu_sum.

    // ── 5. Larger tensor to demonstrate real throughput ───────────────────────
    println!("\nLarge tensor (1M elements):");
    let n = 1_000_000usize;
    let big_a = Tensor::from_shape_vec([n], (0..n).map(|i| i as f32).collect(), CpuAllocator)?;
    let big_b = Tensor::from_shape_vec([n], (0..n).map(|i| -(i as f32)).collect(), CpuAllocator)?;

    let gpu_big_a = session.upload_tensor(&big_a)?;
    let gpu_big_b = session.upload_tensor(&big_b)?;

    let t0 = std::time::Instant::now();
    let gpu_big_sum = add(&session, &gpu_big_a, &gpu_big_b)?;
    // Poll to ensure the GPU has actually finished before we measure time.
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());
    println!("  add completed in  {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let gpu_big_relu = relu(&session, &gpu_big_sum)?;
    let _ = session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());
    println!("  relu completed in {:?}", t1.elapsed());

    // Spot-check a few values
    let cpu_big_relu = session.download_tensor(&gpu_big_relu)?;
    // a[i] + b[i] = i + (-i) = 0.0 for all i → relu(0.0) = 0.0
    let all_zero = cpu_big_relu.as_slice().iter().all(|&v| v == 0.0);
    println!("  all values == 0.0 after relu(a + (-a)): {}", all_zero);

    println!("\nDone.");
    Ok(())
}

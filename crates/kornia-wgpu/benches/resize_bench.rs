// benches/resize_bench.rs

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_native;
use kornia_wgpu::ops::resize::resize_bilinear_f32;
use kornia_wgpu::session::WgpuSession;
use kornia_wgpu::transfer::{image_to_cpu, image_to_gpu};
use std::hint::black_box;

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bilinear Resize (1080p to 720p)");

    // 1. Setup Data
    let session = pollster::block_on(WgpuSession::new()).expect("Failed to init wgpu");
    let in_size = ImageSize {
        width: 3840,
        height: 2160,
    };
    let out_size = ImageSize {
        width: 1280,
        height: 720,
    };

    let cpu_data = vec![0.5f32; in_size.width * in_size.height];
    let cpu_image = Image::<f32, 1, _>::new(in_size, cpu_data, CpuAllocator).unwrap();
    let gpu_image = image_to_gpu(&session, &cpu_image).unwrap();

    // 2. Warmup wgpu
    let _ = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
    session
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely());

    // ==========================================
    // GPU: Compute Only
    // ==========================================
    group.bench_function("GPU Compute Only", |b| {
        b.iter(|| {
            let res = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
            session
                .raw_device()
                .poll(wgpu::PollType::wait_indefinitely());
            black_box(res)
        });
    });

    // ==========================================
    // GPU: End-to-End (RAM -> VRAM -> Math -> RAM)
    // ==========================================
    group.bench_function("GPU End-to-End", |b| {
        b.iter(|| {
            let gpu_in = image_to_gpu(&session, &cpu_image).unwrap();
            let res_gpu = resize_bilinear_f32(&session, &gpu_in, out_size).unwrap();
            session
                .raw_device()
                .poll(wgpu::PollType::wait_indefinitely());
            let cpu_out = image_to_cpu(&session, &res_gpu).unwrap();
            black_box(cpu_out)
        });
    });

    // ==========================================
    // CPU: Kornia Baseline (resize_native)
    // ==========================================
    // We pre-allocate the output buffer outside the iteration to
    // strictly measure the math, not the memory allocation!
    let mut cpu_out_image = Image::<f32, 1, _>::new(
        out_size,
        vec![0.0f32; out_size.width * out_size.height],
        CpuAllocator,
    )
    .unwrap();

    group.bench_function("CPU Kornia (Native)", |b| {
        b.iter(|| {
            resize_native(&cpu_image, &mut cpu_out_image, InterpolationMode::Bilinear).unwrap();

            // Pass the reference into the black box, but add a semicolon
            // so we don't return it out of the closure!
            black_box(&mut cpu_out_image);
        });
    });

    group.finish();
}
criterion_group!(benches, bench_resize);
criterion_main!(benches);

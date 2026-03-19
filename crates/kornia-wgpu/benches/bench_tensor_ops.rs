use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor::{CpuAllocator, Tensor};
use kornia_wgpu::ops::tensor::elementwise::{add, relu};
use kornia_wgpu::session::WgpuSession;
use std::hint::black_box;

fn bench_tensor_ops(c: &mut Criterion) {
    let session = pollster::block_on(WgpuSession::new()).expect("Failed to init wgpu");

    // ── Small tensor: [256, 256] ──────────────────────────────────────────────
    {
        let shape = [256, 256];
        let n = 256 * 256;

        let cpu_a = Tensor::from_shape_vec(shape, vec![1.0f32; n], CpuAllocator).unwrap();
        let cpu_b = Tensor::from_shape_vec(shape, vec![2.0f32; n], CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        // Warmup
        let _ = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add [256, 256]");

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = add(&session, &gpu_a, &gpu_b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let a = session.upload_tensor(&cpu_a).unwrap();
                let b = session.upload_tensor(&cpu_b).unwrap();
                let res = add(&session, &a, &b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let out = session.download_tensor(&res).unwrap();
                black_box(out)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                let out: Vec<f32> = cpu_a
                    .as_slice()
                    .iter()
                    .zip(cpu_b.as_slice().iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                black_box(out)
            });
        });

        group.finish();
    }

    // ── Medium tensor: [1024, 1024] ───────────────────────────────────────────
    {
        let shape = [1024, 1024];
        let n = 1024 * 1024;

        let cpu_a = Tensor::from_shape_vec(shape, vec![1.0f32; n], CpuAllocator).unwrap();
        let cpu_b = Tensor::from_shape_vec(shape, vec![2.0f32; n], CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        // Warmup
        let _ = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add [1024, 1024]");

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = add(&session, &gpu_a, &gpu_b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let a = session.upload_tensor(&cpu_a).unwrap();
                let b = session.upload_tensor(&cpu_b).unwrap();
                let res = add(&session, &a, &b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let out = session.download_tensor(&res).unwrap();
                black_box(out)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                let out: Vec<f32> = cpu_a
                    .as_slice()
                    .iter()
                    .zip(cpu_b.as_slice().iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                black_box(out)
            });
        });

        group.finish();
    }

    // ── Large tensor: [1M] ───────────────────────────────────────────────────
    {
        let n = 1_000_000usize;
        let shape = [n];

        let cpu_a = Tensor::from_shape_vec(shape, vec![1.0f32; n], CpuAllocator).unwrap();
        let cpu_b = Tensor::from_shape_vec(shape, vec![2.0f32; n], CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        // Warmup
        let _ = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add [1_000_000]");

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = add(&session, &gpu_a, &gpu_b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let a = session.upload_tensor(&cpu_a).unwrap();
                let b = session.upload_tensor(&cpu_b).unwrap();
                let res = add(&session, &a, &b).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let out = session.download_tensor(&res).unwrap();
                black_box(out)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                let out: Vec<f32> = cpu_a
                    .as_slice()
                    .iter()
                    .zip(cpu_b.as_slice().iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                black_box(out)
            });
        });

        group.finish();
    }

    // ── ReLU: [1024, 1024] ────────────────────────────────────────────────────
    {
        let shape = [1024, 1024];
        let n = 1024 * 1024;

        // Mixed positive/negative to make relu non-trivial
        let data: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let cpu_a = Tensor::from_shape_vec(shape, data.clone(), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();

        // Warmup
        let _ = relu(&session, &gpu_a).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("ReLU [1024, 1024]");

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = relu(&session, &gpu_a).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let a = session.upload_tensor(&cpu_a).unwrap();
                let res = relu(&session, &a).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let out = session.download_tensor(&res).unwrap();
                black_box(out)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                let out: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
                black_box(out)
            });
        });

        group.finish();
    }

    // ── Chained: add → relu [1024, 1024] ─────────────────────────────────────
    //
    // Measures the cost of two ops chained entirely in VRAM vs two separate
    // CPU passes. This is the main argument for GPU chaining — the GPU reads
    // its own output directly, no PCIe transfer between ops.
    {
        let shape = [1024, 1024];
        let n = 1024 * 1024;

        let data_a: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -2.0 })
            .collect();
        let data_b: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { -3.0 } else { 5.0 })
            .collect();

        let cpu_a = Tensor::from_shape_vec(shape, data_a.clone(), CpuAllocator).unwrap();
        let cpu_b = Tensor::from_shape_vec(shape, data_b.clone(), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        // Warmup
        let tmp = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = relu(&session, &tmp).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Chained add → relu [1024, 1024]");

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let sum = add(&session, &gpu_a, &gpu_b).unwrap();
                let res = relu(&session, &sum).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let a = session.upload_tensor(&cpu_a).unwrap();
                let b = session.upload_tensor(&cpu_b).unwrap();
                let sum = add(&session, &a, &b).unwrap();
                let res = relu(&session, &sum).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let out = session.download_tensor(&res).unwrap();
                black_box(out)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                let out: Vec<f32> = data_a
                    .iter()
                    .zip(data_b.iter())
                    .map(|(&a, &b)| (a + b).max(0.0))
                    .collect();
                black_box(out)
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_tensor_ops);
criterion_main!(benches);

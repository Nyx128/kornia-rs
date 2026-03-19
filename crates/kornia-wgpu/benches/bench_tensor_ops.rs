use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor::{CpuAllocator, Tensor};
use kornia_wgpu::ops::tensor::elementwise::{add, relu};
use kornia_wgpu::session::WgpuSession;
use std::hint::black_box;

const BATCH: usize = 64;
const ROWS: usize = 1024;
const COLS: usize = 1024;
const TENSOR_N: usize = ROWS * COLS;
const FLAT_N: usize = BATCH * TENSOR_N;

fn pseudorandom_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
            let x = x ^ (x >> 16);
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_tensor_ops(c: &mut Criterion) {
    let session = pollster::block_on(WgpuSession::new()).expect("Failed to init wgpu");

    // ── Baseline: single [1024, 1024] ─────────────────────────────────────────
    // Reference point — one tensor, one dispatch.
    {
        let shape = [ROWS, COLS];

        let cpu_a =
            Tensor::from_shape_vec(shape, pseudorandom_vec(TENSOR_N, 0), CpuAllocator).unwrap();
        let cpu_b =
            Tensor::from_shape_vec(shape, pseudorandom_vec(TENSOR_N, 42), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        let _ = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add single [1024, 1024]");
        let mut cpu_out = vec![0.0f32; TENSOR_N];

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
                cpu_out
                    .iter_mut()
                    .zip(cpu_a.as_slice().iter().zip(cpu_b.as_slice().iter()))
                    .for_each(|(o, (&a, &b))| *o = a + b);
                black_box(cpu_out.as_ptr())
            });
        });

        group.finish();
    }

    // ── Batched sequential: 64 × [1024, 1024] ────────────────────────────────
    // Dispatches 64 ops without polling between them — the GPU command queue
    // accumulates all 64 submissions and executes them back-to-back.
    // One poll at the end waits for all 64 to complete.
    // This is how you'd use the current API for a batch of feature maps.
    {
        let shape = [ROWS, COLS];

        // Pre-upload all 64 tensor pairs
        let pairs: Vec<_> = (0..BATCH)
            .map(|i| {
                let cpu_a = Tensor::from_shape_vec(
                    shape,
                    pseudorandom_vec(TENSOR_N, i as u32 * 2),
                    CpuAllocator,
                )
                .unwrap();
                let cpu_b = Tensor::from_shape_vec(
                    shape,
                    pseudorandom_vec(TENSOR_N, i as u32 * 2 + 1),
                    CpuAllocator,
                )
                .unwrap();
                let gpu_a = session.upload_tensor(&cpu_a).unwrap();
                let gpu_b = session.upload_tensor(&cpu_b).unwrap();
                (cpu_a, cpu_b, gpu_a, gpu_b)
            })
            .collect();

        // Warmup
        for (_, _, ga, gb) in &pairs {
            let _ = add(&session, ga, gb).unwrap();
        }
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add batched sequential 64 × [1024, 1024]");
        let mut cpu_out = vec![0.0f32; TENSOR_N];

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                // Queue all 64 dispatches before polling — GPU runs them in pipeline
                let results: Vec<_> = pairs
                    .iter()
                    .map(|(_, _, ga, gb)| add(&session, ga, gb).unwrap())
                    .collect();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(results)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let results: Vec<_> = pairs
                    .iter()
                    .map(|(ca, cb, _, _)| {
                        let a = session.upload_tensor(ca).unwrap();
                        let b = session.upload_tensor(cb).unwrap();
                        add(&session, &a, &b).unwrap()
                    })
                    .collect();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let downloaded: Vec<_> = results
                    .iter()
                    .map(|r| session.download_tensor(r).unwrap())
                    .collect();
                black_box(downloaded)
            });
        });

        group.bench_function("CPU", |b| {
            b.iter(|| {
                for (ca, cb, _, _) in &pairs {
                    cpu_out
                        .iter_mut()
                        .zip(ca.as_slice().iter().zip(cb.as_slice().iter()))
                        .for_each(|(o, (&a, &b))| *o = a + b);
                    black_box(cpu_out.as_ptr());
                }
            });
        });

        group.finish();
    }

    // ── Batched flat: [64 * 1024 * 1024] ─────────────────────────────────────
    // Flattens the entire batch into one tensor — single dispatch, maximum
    // GPU occupancy. Best-case GPU throughput number.
    // Equivalent to processing all 64 feature maps as one contiguous buffer.
    {
        let shape = [FLAT_N];

        let cpu_a =
            Tensor::from_shape_vec(shape, pseudorandom_vec(FLAT_N, 0), CpuAllocator).unwrap();
        let cpu_b =
            Tensor::from_shape_vec(shape, pseudorandom_vec(FLAT_N, 99), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        let _ = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Add batched flat [64 * 1024 * 1024]");
        let mut cpu_out = vec![0.0f32; FLAT_N];

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
                cpu_out
                    .iter_mut()
                    .zip(cpu_a.as_slice().iter().zip(cpu_b.as_slice().iter()))
                    .for_each(|(o, (&a, &b))| *o = a + b);
                black_box(cpu_out.as_ptr())
            });
        });

        group.finish();
    }

    // ── Batched flat relu: [64 * 1024 * 1024] ────────────────────────────────
    {
        let shape = [FLAT_N];

        let cpu_a =
            Tensor::from_shape_vec(shape, pseudorandom_vec(FLAT_N, 44), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();

        let _ = relu(&session, &gpu_a).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("ReLU batched flat [64 * 1024 * 1024]");
        let mut cpu_out = vec![0.0f32; FLAT_N];

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
                cpu_out
                    .iter_mut()
                    .zip(cpu_a.as_slice().iter())
                    .for_each(|(o, &v)| *o = v.max(0.0));
                black_box(cpu_out.as_ptr())
            });
        });

        group.finish();
    }

    // ── Batched flat chained: add → relu [64 * 1024 * 1024] ──────────────────
    // The strongest argument for GPU chaining at scale — 64 feature maps,
    // two ops, one intermediate buffer, one poll.
    {
        let shape = [FLAT_N];

        let cpu_a =
            Tensor::from_shape_vec(shape, pseudorandom_vec(FLAT_N, 0), CpuAllocator).unwrap();
        let cpu_b =
            Tensor::from_shape_vec(shape, pseudorandom_vec(FLAT_N, 45), CpuAllocator).unwrap();
        let gpu_a = session.upload_tensor(&cpu_a).unwrap();
        let gpu_b = session.upload_tensor(&cpu_b).unwrap();

        let tmp = add(&session, &gpu_a, &gpu_b).unwrap();
        let _ = relu(&session, &tmp).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        let mut group = c.benchmark_group("Chained add → relu batched flat [64 * 1024 * 1024]");
        let mut cpu_out = vec![0.0f32; FLAT_N];

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
                cpu_out
                    .iter_mut()
                    .zip(cpu_a.as_slice().iter().zip(cpu_b.as_slice().iter()))
                    .for_each(|(o, (&a, &b))| *o = (a + b).max(0.0));
                black_box(cpu_out.as_ptr())
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_tensor_ops);
criterion_main!(benches);

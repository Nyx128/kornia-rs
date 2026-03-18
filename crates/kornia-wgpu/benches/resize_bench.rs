use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_native;
use kornia_wgpu::ops::image::resize::resize_bilinear_f32;
use kornia_wgpu::session::WgpuSession;
use kornia_wgpu::transfer::{image_to_cpu, image_to_gpu};
use std::hint::black_box;

fn bench_resize(c: &mut Criterion) {
    let session = pollster::block_on(WgpuSession::new()).expect("Failed to init wgpu");

    let in_size = ImageSize {
        width: 3840,
        height: 2160,
    };
    let out_size = ImageSize {
        width: 1920,
        height: 1080,
    };

    // ── 1-channel (grayscale) ─────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("Bilinear Resize 1ch (2160p to 1080p)");

        let cpu_image = Image::<f32, 1, _>::new(
            in_size,
            vec![0.5f32; in_size.width * in_size.height],
            CpuAllocator,
        )
        .unwrap();
        let gpu_image = image_to_gpu(&session, &cpu_image).unwrap();

        // Warmup
        let _ = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let gpu_in = image_to_gpu(&session, &cpu_image).unwrap();
                let res_gpu = resize_bilinear_f32(&session, &gpu_in, out_size).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let cpu_out = image_to_cpu(&session, &res_gpu).unwrap();
                black_box(cpu_out)
            });
        });

        let mut cpu_out = Image::<f32, 1, _>::new(
            out_size,
            vec![0.0f32; out_size.width * out_size.height],
            CpuAllocator,
        )
        .unwrap();
        group.bench_function("CPU Kornia (Native)", |b| {
            b.iter(|| {
                resize_native(&cpu_image, &mut cpu_out, InterpolationMode::Bilinear).unwrap();
                black_box(&mut cpu_out);
            });
        });

        group.finish();
    }

    // ── 3-channel (RGB) ───────────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("Bilinear Resize 3ch RGB (2160p to 1080p)");

        let cpu_image = Image::<f32, 3, _>::new(
            in_size,
            vec![0.5f32; in_size.width * in_size.height * 3],
            CpuAllocator,
        )
        .unwrap();
        let gpu_image = image_to_gpu(&session, &cpu_image).unwrap();

        // Warmup
        let _ = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = resize_bilinear_f32(&session, &gpu_image, out_size).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let gpu_in = image_to_gpu(&session, &cpu_image).unwrap();
                let res_gpu = resize_bilinear_f32(&session, &gpu_in, out_size).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let cpu_out = image_to_cpu(&session, &res_gpu).unwrap();
                black_box(cpu_out)
            });
        });

        let mut cpu_out = Image::<f32, 3, _>::new(
            out_size,
            vec![0.0f32; out_size.width * out_size.height * 3],
            CpuAllocator,
        )
        .unwrap();
        group.bench_function("CPU Kornia (Native)", |b| {
            b.iter(|| {
                resize_native(&cpu_image, &mut cpu_out, InterpolationMode::Bilinear).unwrap();
                black_box(&mut cpu_out);
            });
        });

        group.finish();
    }

    // ── 3-channel 720p → 1080p (video pipeline case) ─────────────────────────
    {
        let mut group = c.benchmark_group("Bilinear Resize 3ch RGB (720p to 1080p)");

        let in_720p = ImageSize {
            width: 1280,
            height: 720,
        };
        let out_1080p = ImageSize {
            width: 1920,
            height: 1080,
        };

        let cpu_image = Image::<f32, 3, _>::new(
            in_720p,
            vec![0.5f32; in_720p.width * in_720p.height * 3],
            CpuAllocator,
        )
        .unwrap();
        let gpu_image = image_to_gpu(&session, &cpu_image).unwrap();

        // Warmup
        let _ = resize_bilinear_f32(&session, &gpu_image, out_1080p).unwrap();
        let _ = session
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely());

        group.bench_function("GPU Compute Only", |b| {
            b.iter(|| {
                let res = resize_bilinear_f32(&session, &gpu_image, out_1080p).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                black_box(res)
            });
        });

        group.bench_function("GPU End-to-End", |b| {
            b.iter(|| {
                let gpu_in = image_to_gpu(&session, &cpu_image).unwrap();
                let res_gpu = resize_bilinear_f32(&session, &gpu_in, out_1080p).unwrap();
                let _ = session
                    .raw_device()
                    .poll(wgpu::PollType::wait_indefinitely());
                let cpu_out = image_to_cpu(&session, &res_gpu).unwrap();
                black_box(cpu_out)
            });
        });

        let mut cpu_out = Image::<f32, 3, _>::new(
            out_1080p,
            vec![0.0f32; out_1080p.width * out_1080p.height * 3],
            CpuAllocator,
        )
        .unwrap();
        group.bench_function("CPU Kornia (Native)", |b| {
            b.iter(|| {
                resize_native(&cpu_image, &mut cpu_out, InterpolationMode::Bilinear).unwrap();
                black_box(&mut cpu_out);
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);

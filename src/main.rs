use opencl3::command_queue::{ CommandQueue, CL_QUEUE_PROFILING_ENABLE };
use opencl3::context::Context;
use opencl3::device::{ get_all_devices, Device, CL_DEVICE_TYPE_GPU };
use opencl3::memory::{ Buffer, CL_MEM_READ_WRITE };
use opencl3::types::{ cl_float, CL_BLOCKING };
use opencl3::Result;
use std::ptr;
use std::time::Instant;

fn measure_throughput(data_size: usize) -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE).expect(
        "CommandQueue::create_default failed"
    );

    // Allocate host memory
    let mut h_data = vec![0.0f32; data_size];

    // Allocate device memory
    let mut d_data = unsafe {
        Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, data_size, ptr::null_mut())?
    };

    // Measure Host to Device transfer
    let start = Instant::now();
    unsafe {
        queue.enqueue_write_buffer(&mut d_data, CL_BLOCKING, 0, &h_data, &[])?;
    }
    queue.finish()?;
    let duration = start.elapsed();
    let h2d_throughput =
        ((data_size * std::mem::size_of::<f32>()) as f64) / duration.as_secs_f64() / 1e9;
    println!("Host to Device Throughput: {:.2} GB/s", h2d_throughput);

    // Measure Device to Host transfer
    let start = Instant::now();
    unsafe {
        queue.enqueue_read_buffer(&d_data, CL_BLOCKING, 0, &mut h_data, &[])?;
    }
    queue.finish()?;
    let duration = start.elapsed();
    let d2h_throughput =
        ((data_size * std::mem::size_of::<f32>()) as f64) / duration.as_secs_f64() / 1e9;
    println!("Device to Host Throughput: {:.2} GB/s", d2h_throughput);

    Ok(())
}

fn main() -> Result<()> {
    // Example: measure throughput for 100 million floats (~400 MB)
    let data_size = 1024 * 1024 * 1024;
    measure_throughput(data_size)?;
    Ok(())
}

use eframe::egui;
use opencl3::command_queue::{ CommandQueue, CL_QUEUE_PROFILING_ENABLE };
use opencl3::context::Context;
use opencl3::device::{ get_all_devices, Device, CL_DEVICE_TYPE_GPU };
use opencl3::memory::{ Buffer, CL_MEM_READ_WRITE };
use opencl3::types::{ cl_device_id, cl_float, CL_BLOCKING };
use opencl3::Result;
use std::collections::HashMap;
use std::ptr;
use std::sync::{ Arc, Mutex };
use std::time::Instant;

struct Throughput {
    h2d_throughput: f64,
    d2h_throughput: f64,
    h2d_duration: f64,
    d2h_duration: f64,
}

impl Throughput {
    fn new() -> Self {
        Throughput {
            h2d_throughput: 0.0,
            d2h_throughput: 0.0,
            h2d_duration: 0.0,
            d2h_duration: 0.0,
        }
    }

    fn measure(&mut self, data_size: usize, device: &Device) -> Result<()> {
        let context = Context::from_device(device).expect("Context::from_device failed");
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE).expect(
            "CommandQueue::create_default failed"
        );

        let mut h_data = vec![0.0f32; data_size];

        let mut d_data = unsafe {
            Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, data_size, ptr::null_mut())?
        };

        let start = Instant::now();
        unsafe {
            queue.enqueue_write_buffer(&mut d_data, CL_BLOCKING, 0, &h_data, &[])?;
        }
        queue.finish()?;
        let duration = start.elapsed();
        self.h2d_duration = duration.as_secs_f64();
        self.h2d_throughput =
            ((data_size * std::mem::size_of::<f32>()) as f64) / self.h2d_duration / 1e9;

        let start = Instant::now();
        unsafe {
            queue.enqueue_read_buffer(&d_data, CL_BLOCKING, 0, &mut h_data, &[])?;
        }
        queue.finish()?;
        let duration = start.elapsed();
        self.d2h_duration = duration.as_secs_f64();
        self.d2h_throughput =
            ((data_size * std::mem::size_of::<f32>()) as f64) / self.d2h_duration / 1e9;

        Ok(())
    }

    fn approximate_link_speed(&self) -> (i32, Vec<&'static str>) {
        let rounded_avg_throughput = (
            (self.h2d_throughput + self.d2h_throughput) /
            2.0
        ).round() as i32;

        let pcie_speeds: HashMap<i32, Vec<&str>> = [
            (1, vec!["PCIe 1.0 x4", "PCIe 2.0 x2", "PCIe 3.0 x1"]),
            (2, vec!["PCIe 1.0 x8", "PCIe 2.0 x4", "PCIe 3.0 x2", "PCIe 4.0 x1"]),
            (4, vec!["PCIe 1.0 x16", "PCIe 2.0 x8", "PCIe 3.0 x4", "PCIe 4.0 x2", "PCIe 5.0 x1"]),
            (8, vec!["PCIe 2.0 x16", "PCIe 3.0 x8", "PCIe 4.0 x4", "PCIe 5.0 x2"]),
            (16, vec!["PCIe 3.0 x16", "PCIe 4.0 x8", "PCIe 5.0 x4"]),
            (32, vec!["PCIe 4.0 x16", "PCIe 5.0 x8"]),
            (64, vec!["PCIe 5.0 x16"]),
        ]
            .iter()
            .cloned()
            .collect();

        let closest_match = pcie_speeds
            .iter()
            .min_by(|a, b| {
                (a.0 - rounded_avg_throughput).abs().cmp(&(b.0 - rounded_avg_throughput).abs())
            })
            .unwrap();

        (*closest_match.0, closest_match.1.clone())
    }
}

#[derive(Clone)]
struct MyDevice {
    device: Device,
    name: String,
}

impl PartialEq for MyDevice {
    fn eq(&self, other: &Self) -> bool {
        self.device.id() == other.device.id()
    }
}

impl MyDevice {
    fn new(id: cl_device_id) -> Self {
        let device = Device::new(id);
        let name = device.board_name_amd().unwrap_or_default();
        MyDevice { device, name }
    }

    fn get_device(&self) -> &Device {
        &self.device
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct App {
    throughput: Arc<Mutex<Throughput>>,
    data_size: usize,
    h2d_throughput: f64,
    d2h_throughput: f64,
    h2d_duration: f64,
    d2h_duration: f64,
    pcie_speed: (i32, Vec<&'static str>),
    selected_device: Option<MyDevice>,
    devices: Vec<MyDevice>,
    measuring: bool,
    error_message: Option<String>,
}

impl Default for App {
    fn default() -> Self {
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap_or_default()
            .into_iter()
            .map(MyDevice::new)
            .collect();
        Self {
            throughput: Arc::new(Mutex::new(Throughput::new())),
            data_size: 1024, // in MB
            h2d_throughput: 0.0,
            d2h_throughput: 0.0,
            h2d_duration: 0.0,
            d2h_duration: 0.0,
            pcie_speed: (0, vec![]),
            selected_device: None,
            devices,
            measuring: false,
            error_message: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                let (config_ui, result_ui) = columns.split_at_mut(1);
                let config_ui = &mut config_ui[0];
                let result_ui = &mut result_ui[0];

                config_ui.heading("Configuration");

                config_ui.add(
                    egui::Slider::new(&mut self.data_size, 1..=10000).text("Data Size (MB)")
                );

                config_ui.label("Select GPU Device:");

                egui::ComboBox
                    ::from_label("Device")
                    .selected_text(self.selected_device.as_ref().map_or("None", |d| d.name()))
                    .show_ui(config_ui, |ui| {
                        for device in &self.devices {
                            ui.selectable_value(
                                &mut self.selected_device,
                                Some(device.clone()),
                                device.name()
                            );
                        }
                    });

                if config_ui.button("Measure Throughput").clicked() {
                    if let Some(ref device) = self.selected_device {
                        self.measuring = true;
                        self.error_message = None;
                        let data_size = (self.data_size * 1024 * 1024) / std::mem::size_of::<f32>();
                        let device_clone = device.clone();
                        let throughput = Arc::clone(&self.throughput);
                        let error_message = Arc::new(Mutex::new(None));

                        std::thread::spawn({
                            let error_message = Arc::clone(&error_message);
                            move || {
                                let mut throughput = throughput.lock().unwrap();
                                if
                                    let Err(e) = throughput.measure(
                                        data_size,
                                        device_clone.get_device()
                                    )
                                {
                                    let mut error = error_message.lock().unwrap();
                                    *error = Some(format!("Error: {}", e));
                                }
                            }
                        });

                        self.measuring = false;
                        self.error_message = error_message.lock().unwrap().clone();
                    }
                }

                if self.measuring {
                    config_ui.spinner();
                }

                if let Some(ref msg) = self.error_message {
                    config_ui.colored_label(egui::Color32::RED, msg);
                }

                result_ui.heading("Results");

                // Lock to update the UI with the new throughput results
                {
                    let throughput = self.throughput.lock().unwrap();
                    self.h2d_throughput = throughput.h2d_throughput;
                    self.d2h_throughput = throughput.d2h_throughput;
                    self.h2d_duration = throughput.h2d_duration;
                    self.d2h_duration = throughput.d2h_duration;
                    self.pcie_speed = throughput.approximate_link_speed();
                }

                result_ui.label(
                    format!(
                        "Data Size: {} floats (~{} MB)",
                        (self.data_size * 1024 * 1024) / std::mem::size_of::<cl_float>(),
                        self.data_size
                    )
                );
                result_ui.label(
                    format!(
                        "Host to Device Throughput: {:.2} GB/s (Duration: {:.2} s)",
                        self.h2d_throughput,
                        self.h2d_duration
                    )
                );
                result_ui.label(
                    format!(
                        "Device to Host Throughput: {:.2} GB/s (Duration: {:.2} s)",
                        self.d2h_throughput,
                        self.d2h_duration
                    )
                );

                result_ui.separator();

                result_ui.label("Approximate PCIe Link Speed:");
                result_ui.label(format!("Measured Throughput: {} GB/s", self.pcie_speed.0));
                for config in &self.pcie_speed.1 {
                    result_ui.label(format!(" - {}", config));
                }
            });
        });
    }
}

fn main() -> Result<()> {
    let app = App::default();
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };
    eframe
        ::run_native(
            "GPU Throughput App",
            native_options,
            Box::new(|_| Ok(Box::new(app)))
        )
        .unwrap();

    Ok(())
}

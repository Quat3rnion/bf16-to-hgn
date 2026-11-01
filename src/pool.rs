//! Bounded multi-GPU device pool for concurrent tensor encoding.
//!
//! `--devices hip:0,cuda:0` runs one worker thread per listed GPU. Jobs are
//! pulled from a shared `mpsc` channel behind a `Mutex`; a `Condvar`-guarded
//! in-flight counter bounds the number of queued-or-running jobs so the main
//! thread can stream source reads ahead without unbounded RAM growth.
//! Standard library only: no new crate dependencies.

use crate::codec::HtBackend;
use anyhow::{ensure, Context, Result};
use std::{
    sync::{mpsc, Arc, Condvar, Mutex},
    thread::{ScopedJoinHandle, Scope},
};

/// One entry in `--devices`: a backend family plus a device index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDevice {
    pub backend: HtBackend,
    pub index: u32,
}

impl GpuDevice {
    /// Parse one `backend:index` entry (e.g. `hip:0`).
    pub fn parse(spec: &str) -> Result<Self> {
        let (name, index) = spec
            .split_once(':')
            .with_context(|| format!("device {spec:?} needs a backend:index pair, e.g. hip:0"))?;
        let backend = match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => HtBackend::Cpu,
            "hip" => HtBackend::Hip,
            "cuda" => HtBackend::Cuda,
            other => {
                anyhow::bail!("unknown device backend {other:?} (expected cpu, hip, or cuda)")
            }
        };
        let index = index
            .trim()
            .parse::<u32>()
            .with_context(|| format!("device {spec:?} has an invalid index"))?;
        let device = Self { backend, index };
        device.validate_features()?;
        Ok(device)
    }

    /// GPU entries require the matching cargo feature in the active build.
    fn validate_features(self) -> Result<()> {
        match self.backend {
            HtBackend::Cpu => Ok(()),
            HtBackend::Hip => {
                ensure!(
                    cfg!(feature = "hip"),
                    "hip devices require a build with --features hip"
                );
                Ok(())
            }
            HtBackend::Cuda => {
                ensure!(
                    cfg!(feature = "cuda"),
                    "cuda devices require a build with --features cuda"
                );
                Ok(())
            }
        }
    }
}

/// Parse a comma-separated `--devices` list (1..=8 unique entries).
pub fn parse_devices(spec: &str) -> Result<Vec<GpuDevice>> {
    let mut devices = Vec::new();
    for entry in spec.split(',').map(str::trim).filter(|entry| !entry.is_empty()) {
        let device =
            GpuDevice::parse(entry).with_context(|| format!("invalid --devices entry {entry:?}"))?;
        if devices.contains(&device) {
            anyhow::bail!("duplicate --devices entry {entry:?}");
        }
        devices.push(device);
    }
    ensure!(!devices.is_empty(), "--devices needs 1..=8 entries");
    ensure!(devices.len() <= 8, "--devices accepts at most 8 entries");
    Ok(devices)
}

type Job<'scope> = Box<dyn FnOnce() + Send + 'scope>;

/// Bounded-queue accounting shared by the producer and every worker.
struct Queue {
    inflight: Mutex<usize>,
    not_full: Condvar,
}

impl Queue {
    /// Wait for a free queue slot, then count the incoming job.
    fn acquire(&self, depth: usize) {
        let mut inflight = self.inflight.lock().unwrap();
        while *inflight >= depth {
            inflight = self.not_full.wait(inflight).unwrap();
        }
        *inflight += 1;
    }

    /// Release the slot after a worker finishes a job.
    fn release(&self) {
        let mut inflight = self.inflight.lock().unwrap();
        *inflight -= 1;
        self.not_full.notify_one();
    }

    fn count(&self) -> usize {
        *self.inflight.lock().unwrap()
    }
}

/// One worker thread per device pulling jobs from a shared bounded queue.
/// `submit` blocks once `depth` jobs are queued-or-running. The pool borrows
/// its channel through `Arc`/`Mutex`, so the channel state is owned here and
/// outlives the workers.
pub struct DevicePool<'scope> {
    sender: mpsc::Sender<Job<'scope>>,
    receiver: Arc<Mutex<mpsc::Receiver<Job<'scope>>>>,
    queue: Arc<Queue>,
    workers: Vec<ScopedJoinHandle<'scope, ()>>,
    depth: usize,
}

impl<'scope> DevicePool<'scope> {
    /// Spawn one worker per device. Workers exit when the sender is dropped
    /// (see [`DevicePool::join`]). The scope must outlive the pool.
    pub fn new<'env>(
        scope: &'scope Scope<'scope, 'env>,
        devices: &[GpuDevice],
        depth: usize,
    ) -> Self
    where
        'env: 'scope,
    {
        let (sender, receiver) = mpsc::channel::<Job<'scope>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let queue = Arc::new(Queue {
            inflight: Mutex::new(0),
            not_full: Condvar::new(),
        });
        let workers = devices
            .iter()
            .map(|_| {
                let receiver = receiver.clone();
                let queue = queue.clone();
                scope.spawn(move || loop {
                    let job = { receiver.lock().unwrap().recv() };
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                    queue.release();
                })
            })
            .collect::<Vec<ScopedJoinHandle<'scope, ()>>>();
        Self {
            sender,
            receiver,
            queue,
            workers,
            depth,
        }
    }

    /// Queue one job, blocking while `depth` jobs are already in flight.
    pub fn submit<F: FnOnce() + Send + 'scope>(&self, job: F) -> Result<()> {
        self.queue.acquire(self.depth);
        self.sender
            .send(Box::new(job))
            .map_err(|_| anyhow::anyhow!("device pool workers exited early"))
    }

    /// Jobs submitted and not yet finished (queued plus running).
    pub fn in_flight(&self) -> usize {
        self.queue.count()
    }

    /// Close the queue and join every worker.
    pub fn join(self) -> Result<()> {
        drop(self.sender);
        for worker in self.workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("device pool worker panicked"))?;
        }
        Ok(())
    }
}

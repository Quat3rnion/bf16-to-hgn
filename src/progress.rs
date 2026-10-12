//! Optional terminal progress for long conversion and verification stages.
//!
//! Library entry points accept a [`Progress`]. Its default is disabled: no bar
//! is allocated and nothing is written. The CLI opts in with
//! [`Progress::stderr`], which animates only when standard error is a terminal,
//! so redirected logs and the JSON written to standard output never contain
//! progress output.
//!
//! A [`Stage`] reports one bounded unit of work. Callers advance it by work
//! actually completed (a chunk or row block, never a value), and its final
//! line claims completion only when the reported position equals the declared
//! total. Generic stages extrapolate their recent aggregate rate. Conversion
//! instead sums remaining work at separately measured operation/codec/device
//! rates, and labels unmeasured classes and uncounted finalization explicitly.
//!
//! A live stage draws a block ordered from overall to local work, laid out
//! again for the terminal's current width at every redraw:
//!
//! ```text
//! ⠙ Convert passes  37% [===========>                  ] 512.00G/1.37T values
//!   elapsed 04:12:34 · ETA ~7h · 33.79M values/s
//!   encode HT3 on HIP · layers.12.mlp.experts.gate_up_proj.weight
//!   expert 37 of 512 · gate+up 384/640 row pairs (60%)
//! ```
//!
//! The first two lines are the stage's overall work and time; the last two
//! show the latest [`Detail`] (or one [`Stage::set_item`] line) and appear
//! only once the caller reports one. To fit a narrow terminal, lines drop
//! optional fields (rate, gauge, counts, part, item percentage) and shorten
//! names in the middle, keeping a tensor's layer scope; ETA wording is never
//! shortened. A stopped stage keeps its context so a failure shows where it
//! stopped; a completed stage collapses to one "done" line.

use crate::codec::{Encoding, HtBackend, I4rBackend, Q4cpBackend};
use indicatif::{
    BinaryBytes, FormattedDuration, HumanCount, HumanDuration, ProgressBar, ProgressDrawTarget,
    ProgressState, ProgressStyle,
};
use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    io::IsTerminal as _,
    iter, mem,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

/// Redraw interval while a stage is live, so elapsed time, the spinner, the
/// latest detail, the rate estimate and the layout for the terminal's current
/// width stay current.
const TICK: Duration = Duration::from_millis(250);
/// Terminal width assumed when none is reported, as indicatif does.
const FALLBACK_COLUMNS: usize = 80;
/// The gauge takes the header's spare columns, up to `GAUGE_MAX`, and is left
/// out when fewer than `GAUGE_MIN` remain.
const GAUGE_MIN: usize = 8;
const GAUGE_MAX: usize = 30;
/// Columns of a shortened tensor name's end that its kept layer scope must
/// leave; the name gets at least this much room even on a crowded line.
const NAME_TAIL: usize = 16;
/// Starts each line below the first, indented by `INDENT` columns.
const LINE: &str = "\n  ";
const INDENT: usize = 2;
/// Separates fields of a line.
const FIELD: &str = " · ";
const FINALIZING: &str = "finalizing (outside ETA)";

/// Progress sink handed to long-running library operations.
///
/// Cheap to copy. There is no global state: each caller decides whether its
/// operation reports progress.
#[derive(Clone, Copy, Debug, Default)]
pub struct Progress {
    enabled: bool,
}

impl Progress {
    /// No output and no per-stage allocation.
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Animated progress on standard error when it is a terminal whose `TERM`
    /// is set and not `dumb` (where indicatif would hide it); otherwise
    /// disabled.
    pub fn stderr() -> Self {
        Self {
            enabled: std::io::stderr().is_terminal()
                && std::env::var_os("TERM").is_some_and(|term| term != "dumb"),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start one stage of `total` units. Stages are sequential: finish or drop
    /// the previous stage before starting the next one.
    pub fn stage(&self, label: impl Into<Cow<'static, str>>, total: u64, unit: Unit) -> Stage {
        self.start_stage(label, total, unit, None)
    }

    /// A conversion stage whose ETA uses only its planned, measured work classes.
    pub(crate) fn work_stage(
        &self,
        label: impl Into<Cow<'static, str>>,
        estimate: WorkEstimate,
    ) -> Stage {
        self.start_stage(label, estimate.total(), Unit::Values, Some(estimate))
    }

    fn start_stage(
        &self,
        label: impl Into<Cow<'static, str>>,
        total: u64,
        unit: Unit,
        estimate: Option<WorkEstimate>,
    ) -> Stage {
        if !self.enabled {
            return Stage { live: None, unit };
        }
        let label = label.into();
        let context = Arc::new(Mutex::new(Context {
            label_columns: width(&label),
            estimate,
            ..Context::default()
        }));
        let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr())
            .with_style(unit.style(Frame::Running, &context))
            .with_prefix(label);
        bar.enable_steady_tick(TICK);
        Stage {
            live: Some(Live { bar, context }),
            unit,
        }
    }
}

/// Device that actually runs a work class's chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device {
    Cpu,
    Hip,
    Cuda,
}

impl Device {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Hip => "HIP",
            Self::Cuda => "CUDA",
        }
    }
}

impl From<HtBackend> for Device {
    fn from(backend: HtBackend) -> Self {
        match backend {
            HtBackend::Cpu => Self::Cpu,
            HtBackend::Hip => Self::Hip,
            HtBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<Q4cpBackend> for Device {
    fn from(backend: Q4cpBackend) -> Self {
        match backend {
            Q4cpBackend::Cpu => Self::Cpu,
            Q4cpBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<I4rBackend> for Device {
    fn from(backend: I4rBackend) -> Self {
        match backend {
            I4rBackend::Cpu => Self::Cpu,
            I4rBackend::Hip => Self::Hip,
            I4rBackend::Cuda => Self::Cuda,
        }
    }
}

/// End-to-end value-processing passes. The HT backend selects HT3/HT4
/// encoding's device and the independent I4R backend selects I4R's; every
/// other codec runs on CPU. Reads, rotations, hashing and writes within that
/// chunk are included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkClass {
    Copy(Encoding),
    Encode(Encoding, Device),
    WriteRotation,
    ScanFp8Scale,
}

const ENCODING_CLASSES: usize = 11;
/// HT3/HT4 on HIP and CUDA, I4R on HIP and CUDA, and Q4C-P on CUDA.
const GPU_CLASSES: usize = 7;
const WORK_CLASSES: usize = ENCODING_CLASSES + GPU_CLASSES + 3 + 2;

impl WorkClass {
    pub(crate) fn encode(
        encoding: Encoding,
        ht_backend: HtBackend,
        i4r_backend: I4rBackend,
        q4cp_backend: Q4cpBackend,
    ) -> Self {
        let device = if encoding.ht_bits().is_some() {
            ht_backend.into()
        } else if encoding == Encoding::I4r {
            i4r_backend.into()
        } else if encoding == Encoding::Q4Cp {
            q4cp_backend.into()
        } else {
            Device::Cpu
        };
        Self::Encode(encoding, device)
    }

    pub(crate) fn backend_label(self) -> &'static str {
        match self {
            Self::Encode(_, device) => device.label(),
            _ => "CPU",
        }
    }

    /// Fixed, dense slots: updates never allocate or search the conversion plan.
    fn index(self) -> usize {
        match self {
            Self::Encode(encoding, Device::Cpu) => match encoding {
                Encoding::Bf16 => 0,
                Encoding::F16 => 1,
                Encoding::F32 => 2,
                Encoding::I64 => 3,
                Encoding::Q4Cp => 4,
                Encoding::Q8G64 => 5,
                Encoding::Q6G64 => 6,
                Encoding::Fp8G128 => 7,
                Encoding::Ht => 8,
                Encoding::Ht3 => 9,
                Encoding::I4r => 10,
            },
            Self::Encode(Encoding::Ht, Device::Hip) => ENCODING_CLASSES,
            Self::Encode(Encoding::Ht3, Device::Hip) => ENCODING_CLASSES + 1,
            Self::Encode(Encoding::Ht, Device::Cuda) => ENCODING_CLASSES + 2,
            Self::Encode(Encoding::Ht3, Device::Cuda) => ENCODING_CLASSES + 3,
            Self::Encode(Encoding::I4r, Device::Cuda) => ENCODING_CLASSES + 4,
            Self::Encode(Encoding::I4r, Device::Hip) => ENCODING_CLASSES + 5,
            Self::Encode(Encoding::Q4Cp, Device::Cuda) => ENCODING_CLASSES + 6,
            Self::Encode(_, Device::Hip | Device::Cuda) => {
                unreachable!("only HT, I4R and Q4C-P encoding run on a GPU")
            }
            Self::Copy(Encoding::Bf16) => ENCODING_CLASSES + GPU_CLASSES,
            Self::Copy(Encoding::F32) => ENCODING_CLASSES + GPU_CLASSES + 1,
            Self::Copy(Encoding::I64) => ENCODING_CLASSES + GPU_CLASSES + 2,
            Self::Copy(_) => unreachable!("unsupported direct-copy encoding"),
            Self::WriteRotation => ENCODING_CLASSES + GPU_CLASSES + 3,
            Self::ScanFp8Scale => ENCODING_CLASSES + GPU_CLASSES + 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ClassEstimate {
    remaining: u64,
    observed_values: u64,
    observed_seconds: f64,
}

/// Rates survive class switches and exhausted tensors within this invocation.
/// Only explicitly timed completed chunks contribute samples, never gaps
/// between chunks, another class, or uncounted finalization.
#[derive(Debug, Default)]
pub(crate) struct WorkEstimate {
    classes: [ClassEstimate; WORK_CLASSES],
    total: u64,
    remaining: u64,
}

impl WorkEstimate {
    pub(crate) fn add(&mut self, class: WorkClass, values: u64) -> Option<()> {
        let total = self.total.checked_add(values)?;
        let slot = &mut self.classes[class.index()];
        slot.remaining = slot.remaining.checked_add(values)?;
        self.total = total;
        self.remaining += values;
        Some(())
    }

    pub(crate) fn total(&self) -> u64 {
        self.total
    }

    #[cfg(test)]
    pub(crate) fn planned_values(&self, class: WorkClass) -> u64 {
        self.classes[class.index()].remaining
    }

    fn record(&mut self, class: WorkClass, values: u64, elapsed: Duration) {
        let slot = &mut self.classes[class.index()];
        slot.remaining = slot
            .remaining
            .checked_sub(values)
            .expect("completed work exceeds its planned class total");
        self.remaining = self
            .remaining
            .checked_sub(values)
            .expect("completed work exceeds the planned total");
        if values > 0 && !elapsed.is_zero() {
            slot.observed_values += values;
            slot.observed_seconds += elapsed.as_secs_f64();
        }
    }

    fn remaining(&self) -> WorkRemaining {
        if self.remaining == 0 {
            return WorkRemaining::Complete;
        }
        let mut measured_seconds = 0.0;
        let mut measured_classes = 0;
        let mut learning = 0;
        for slot in &self.classes {
            if slot.remaining == 0 {
                continue;
            }
            if slot.observed_values == 0
                || !slot.observed_seconds.is_finite()
                || slot.observed_seconds <= 0.0
            {
                learning += 1;
            } else {
                measured_seconds +=
                    slot.remaining as f64 * slot.observed_seconds / slot.observed_values as f64;
                measured_classes += 1;
            }
        }
        let measured = match Duration::try_from_secs_f64(measured_seconds) {
            Ok(duration) => duration,
            Err(_) => {
                learning += measured_classes;
                Duration::ZERO
            }
        };
        WorkRemaining::Estimate { measured, learning }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkRemaining {
    Complete,
    Estimate { measured: Duration, learning: usize },
}

impl WorkRemaining {
    /// No measured estimate applies: every counted value is done, or the
    /// current step is deliberately outside the value-work ETA.
    fn finalizing(self, outside_eta: bool) -> bool {
        outside_eta || self == Self::Complete
    }

    fn write(self, outside_eta: bool, out: &mut dyn fmt::Write) -> fmt::Result {
        if self.finalizing(outside_eta) {
            return out.write_str(FINALIZING);
        }
        let Self::Estimate { measured, learning } = self else {
            unreachable!()
        };
        match (measured.is_zero(), learning) {
            (_, 0) => write!(out, "~{:#}", HumanDuration(measured)),
            (true, 1) => out.write_str("learning (1 class)"),
            (true, _) => write!(out, "learning ({learning} classes)"),
            (false, _) => write!(
                out,
                "partial ~{:#} + {learning} unmeasured",
                HumanDuration(measured)
            ),
        }
    }
}

/// Unit of a stage's total and of every [`Stage::advance`] amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Bytes,
    Values,
    Tensors,
}

impl Unit {
    /// Everything after the label is one `frame` key, so a redraw lays out all
    /// of the stage's lines together for the terminal's current width. The
    /// label is indicatif's prefix: the console crate's `NO_COLOR`/`CLICOLOR`
    /// rules decide whether it is bold, and no information depends on that.
    fn style(self, frame: Frame, context: &Arc<Mutex<Context>>) -> ProgressStyle {
        let context = Arc::clone(context);
        ProgressStyle::with_template(frame.template())
            .expect("static progress template")
            .with_key(
                "frame",
                move |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let mut context = lock(&context);
                    // One reused buffer, handed to indicatif in a single write.
                    let mut text = mem::take(&mut context.text);
                    text.clear();
                    let _ = context.write_frame(self, frame, state, &mut text);
                    let _ = out.write_str(&text);
                    context.text = text;
                },
            )
    }

    /// Rounded-down percentage of counted work, a gauge and the counts within
    /// `columns`. The counts outrank the gauge, which repeats the percentage.
    fn write_header(self, state: &ProgressState, columns: usize, out: &mut String) -> fmt::Result {
        let Some(total) = state.len().filter(|&total| total > 0) else {
            return self.write_work(state, out);
        };
        write!(out, "{:>3}%", percent(state.pos(), total))?;
        let mut spare = columns.saturating_sub(4);
        let mut counts = ColumnCount(0);
        self.write_work(state, &mut counts)?;
        let show_counts = counts.0 < spare;
        if show_counts {
            spare -= counts.0 + 1;
        }
        if spare >= GAUGE_MIN + 3 {
            out.push_str(" [");
            write_gauge(state.pos(), total, (spare - 3).min(GAUGE_MAX), out);
            out.push(']');
        }
        if show_counts {
            out.push(' ');
            self.write_work(state, out)?;
        }
        Ok(())
    }

    /// The line a completed stage leaves; its rate is the stage average.
    fn write_done(self, state: &ProgressState, columns: usize, out: &mut String) -> fmt::Result {
        fit(out, columns, 2, |out, variant| {
            out.push_str("done ");
            self.write_amount(out, state.pos())?;
            write!(out, " in {}", FormattedDuration(state.elapsed()))?;
            if variant == 0 {
                out.push_str(" (avg ");
                self.write_rate(state, out)?;
                out.push(')');
            }
            Ok(())
        })
    }

    /// Where an unfinished stage stopped, never rounded up to complete.
    fn write_stopped(self, state: &ProgressState, columns: usize, out: &mut String) -> fmt::Result {
        let total = state.len().filter(|&total| total > 0);
        fit(out, columns, 2, |out, variant| {
            out.push_str("stopped at ");
            match total {
                Some(total) if variant == 1 => write!(out, "{}%", percent(state.pos(), total))?,
                _ => {
                    self.write_work(state, out)?;
                    if let Some(total) = total {
                        write!(out, " ({}%)", percent(state.pos(), total))?;
                    }
                }
            }
            write!(out, " after {}", FormattedDuration(state.elapsed()))
        })
    }
}

/// Which rendering of a stage indicatif draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Running,
    Done,
    Stopped,
}

impl Frame {
    fn template(self) -> &'static str {
        match self {
            Self::Running => "{spinner} {prefix:.bold} {frame}",
            Self::Done | Self::Stopped => "{prefix:.bold}: {frame}",
        }
    }

    /// Columns the template adds to the label on the first line.
    fn lead(self) -> usize {
        match self {
            Self::Running => 3,
            Self::Done | Self::Stopped => 2,
        }
    }
}

/// Live gauge that never fills: a stage whose position already reached its
/// total may still be running final checks, and only [`Stage::finish`] may
/// report completion (finished stages draw no gauge).
fn write_gauge(position: u64, total: u64, cells: usize, out: &mut String) {
    let filled = u128::from(position) * cells as u128 / u128::from(total);
    let filled = filled.min(cells as u128 - 1) as usize;
    out.extend(iter::repeat_n('=', filled));
    out.push('>');
    out.extend(iter::repeat_n(' ', cells - filled - 1));
}

/// Whole percent, rounded down so 100% appears only once all of the counted
/// work is reported.
fn percent(completed: u64, total: u64) -> u128 {
    u128::from(completed) * 100 / u128::from(total)
}

impl Unit {
    fn write_amount(self, out: &mut dyn fmt::Write, amount: u64) -> fmt::Result {
        match self {
            Self::Bytes => write!(out, "{}", BinaryBytes(amount)),
            Self::Values => {
                write_si(out, amount as f64)?;
                out.write_str(" values")
            }
            Self::Tensors => write!(out, "{} tensors", HumanCount(amount)),
        }
    }

    fn write_work(self, state: &ProgressState, out: &mut dyn fmt::Write) -> fmt::Result {
        let total = state.len().unwrap_or(0);
        match self {
            Self::Bytes => write!(out, "{}/{}", BinaryBytes(state.pos()), BinaryBytes(total)),
            Self::Values => {
                write_si(out, state.pos() as f64)?;
                out.write_char('/')?;
                write_si(out, total as f64)?;
                out.write_str(" values")
            }
            Self::Tensors => write!(
                out,
                "{}/{} tensors",
                HumanCount(state.pos()),
                HumanCount(total)
            ),
        }
    }

    /// Recent rate while live; the stage average once finished or stopped.
    fn write_rate(self, state: &ProgressState, out: &mut dyn fmt::Write) -> fmt::Result {
        let rate = state.per_sec();
        if !rate.is_finite() || rate < 0.0 {
            return out.write_str("?/s");
        }
        match self {
            Self::Bytes => write!(out, "{}/s", BinaryBytes(rate as u64)),
            Self::Values => {
                write_si(out, rate)?;
                out.write_str(" values/s")
            }
            Self::Tensors => write!(out, "{rate:.1} tensors/s"),
        }
    }
}

/// A generic stage's ETA at its recent aggregate rate: `?` before measured
/// progress (instead of indicatif's zero placeholder), and finalizing once all
/// counted work is reported while the stage still runs its closing checks.
fn write_remaining(state: &ProgressState, out: &mut String) -> fmt::Result {
    if state.len().is_some_and(|total| state.pos() >= total) {
        return out.write_str(FINALIZING);
    }
    let rate = state.per_sec();
    if state.pos() == 0 || !rate.is_finite() || rate <= 0.0 {
        return out.write_str("ETA ?");
    }
    write!(out, "ETA ~{:#}", HumanDuration(state.eta()))
}

fn write_si(out: &mut dyn fmt::Write, value: f64) -> fmt::Result {
    const PREFIXES: [&str; 6] = ["", "k", "M", "G", "T", "P"];
    let mut scaled = value;
    let mut prefix = 0;
    while scaled >= 1000.0 && prefix + 1 < PREFIXES.len() {
        scaled /= 1000.0;
        prefix += 1;
    }
    if prefix == 0 {
        write!(out, "{scaled:.0}")
    } else {
        write!(out, "{scaled:.2}{}", PREFIXES[prefix])
    }
}

/// Where a stage is within its current item, for [`Stage::set_detail`].
///
/// Every field is borrowed, so building one costs nothing; a live stage copies
/// it into reused buffers and formats it only when redrawing, and a disabled
/// stage ignores it. Empty strings and `None` fields are left out of the
/// display.
#[derive(Clone, Copy, Debug)]
pub struct Detail<'a> {
    /// Phase step, such as `encode`, `copy`, `scan` or `decode`.
    pub operation: &'a str,
    /// Output tensor name.
    pub tensor: &'a str,
    /// Scope-qualified location such as `layers.2`, `mtp.layers.0` or
    /// `visual.blocks.3`; `None` for tensors outside any layer. A scope that
    /// begins `tensor` stays visible when a narrow terminal shortens the name;
    /// any other scope is shown before it.
    pub layer: Option<&'a str>,
    /// Store label of the tensor's encoding.
    pub encoding: &'a str,
    /// Device actually running this step, such as `CPU`, `HIP` or `CUDA`.
    pub backend: &'a str,
    /// `(index, count)`: zero-based index of the current expert and the number
    /// of experts in the tensor. Shown as `expert 7 of 512`, so the last
    /// expert of 512 reads `expert 511 of 512`.
    pub expert: Option<(u64, u64)>,
    /// Part of the expert or tensor, such as `gate`, `up` or `gate+up`, drawn
    /// just before the item progress it qualifies.
    pub part: Option<&'a str>,
    /// Work on the current item finished so far, out of `total`, in `unit`
    /// (for example `rows`, `row pairs`, `values` or `bytes`). A `total` of
    /// zero leaves item progress out.
    pub completed: u64,
    pub total: u64,
    pub unit: &'a str,
}

/// Append the first of `variants`, written from fullest to most compact, that
/// fits `columns`. The most compact stays even when it overflows: indicatif
/// counts the wrapped rows, and wrapping beats hiding a required field.
fn fit(
    out: &mut String,
    columns: usize,
    variants: usize,
    mut write: impl FnMut(&mut String, usize) -> fmt::Result,
) -> fmt::Result {
    let start = out.len();
    for variant in 0..variants {
        out.truncate(start);
        write(out, variant)?;
        if width(&out[start..]) <= columns {
            break;
        }
    }
    Ok(())
}

/// Columns of drawn text. Everything drawn here (ASCII, `·`, `…` and the
/// tool's tensor names) takes one column per character; wider text at worst
/// wraps, and indicatif counts wrapped rows.
fn width(text: &str) -> usize {
    text.chars().count()
}

/// Measures formatted text without storing it.
struct ColumnCount(usize);

impl fmt::Write for ColumnCount {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 += width(text);
        Ok(())
    }
}

/// Append `text` within `columns`. A longer text keeps its first `head`
/// columns and as much of its end as fits, joined by `…`.
fn push_elided(out: &mut String, text: &str, head: usize, columns: usize) {
    let length = width(text);
    if length <= columns {
        out.push_str(text);
        return;
    }
    let kept = columns.saturating_sub(1);
    let head = head.min(kept);
    out.extend(text.chars().take(head));
    out.push('…');
    out.extend(text.chars().skip(length - (kept - head)));
}

/// Start another field of a line that began at `start`.
fn separate(out: &mut String, start: usize) {
    if out.len() > start {
        out.push_str(FIELD);
    }
}

/// One count of a free-form detail unit: binary sizes for `bytes`, SI
/// prefixes for `values`, grouped digits otherwise.
fn write_count(out: &mut String, unit: &str, value: u64) -> fmt::Result {
    match unit {
        "bytes" => write!(out, "{}", BinaryBytes(value)),
        "values" => write_si(out, value as f64),
        _ => write!(out, "{}", HumanCount(value)),
    }
}

/// Width of the terminal on standard error, read at each redraw so the frame
/// after a resize is laid out for the new width.
#[cfg(unix)]
fn terminal_columns() -> Option<usize> {
    use std::os::fd::AsRawFd as _;
    let fd = std::io::stderr().as_raw_fd();
    let mut size = libc::winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: TIOCGWINSZ only writes one `winsize` into this live local.
    let status = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut size) };
    let columns = usize::from(size.ws_col);
    (status == 0 && columns > 0).then_some(columns)
}

#[cfg(not(unix))]
fn terminal_columns() -> Option<usize> {
    None
}

/// What a live stage is doing below its overall counters. Updates copy into
/// these reused buffers; formatting waits for a redraw.
#[derive(Debug, Default)]
struct Item {
    kind: ItemKind,
    /// A note's text, or a detail's operation.
    text: String,
    tensor: String,
    layer: String,
    encoding: String,
    backend: String,
    expert: Option<(u64, u64)>,
    part: String,
    completed: u64,
    total: u64,
    unit: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ItemKind {
    #[default]
    Empty,
    Note,
    Detail,
}

impl Item {
    fn set_note(&mut self, text: &str) {
        self.kind = ItemKind::Note;
        replace(&mut self.text, text);
    }

    fn set_detail(&mut self, detail: &Detail<'_>) {
        self.kind = ItemKind::Detail;
        replace(&mut self.text, detail.operation);
        replace(&mut self.tensor, detail.tensor);
        replace(&mut self.layer, detail.layer.unwrap_or_default());
        replace(&mut self.encoding, detail.encoding);
        replace(&mut self.backend, detail.backend);
        self.expert = detail.expert;
        replace(&mut self.part, detail.part.unwrap_or_default());
        self.completed = detail.completed;
        self.total = detail.total;
        replace(&mut self.unit, detail.unit);
    }

    /// The lines below the overall counters, each fitted to `columns` after
    /// its indent: a note, or a detail's target and position.
    fn write_lines(&self, columns: usize, out: &mut String) -> fmt::Result {
        match self.kind {
            ItemKind::Empty => {}
            ItemKind::Note => {
                out.push_str(LINE);
                push_elided(out, &self.text, columns.saturating_sub(1) / 2, columns);
            }
            ItemKind::Detail => {
                out.push_str(LINE);
                self.write_target(columns, out);
                if self.expert.is_some() || !self.part.is_empty() || self.total > 0 {
                    out.push_str(LINE);
                    fit(out, columns, 3, |out, variant| {
                        self.write_position(variant, out)
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Operation, encoding and backend, then the tensor, for example
    /// `encode HT3 on HIP · layers.12.mlp.experts.gate_up_proj.weight`. Only
    /// the name is shortened, keeping its layer scope while room allows.
    fn write_target(&self, columns: usize, out: &mut String) {
        let start = out.len();
        for word in [&self.text, &self.encoding] {
            if !word.is_empty() {
                if out.len() > start {
                    out.push(' ');
                }
                out.push_str(word);
            }
        }
        if !self.backend.is_empty() {
            if out.len() > start {
                out.push_str(" on ");
            }
            out.push_str(&self.backend);
        }
        let scope = self.scope();
        if scope.is_none() && !self.layer.is_empty() {
            separate(out, start);
            out.push_str(&self.layer);
        }
        if !self.tensor.is_empty() {
            separate(out, start);
            let used = width(&out[start..]);
            let room = columns.saturating_sub(used).max(NAME_TAIL + 1);
            let head = match scope {
                Some(scope) if scope + NAME_TAIL < room => scope,
                _ => 0,
            };
            push_elided(out, &self.tensor, head, room);
        }
    }

    /// Columns of the layer scope and its dot that begin the tensor name, when
    /// the layer is such a prefix.
    fn scope(&self) -> Option<usize> {
        if self.layer.is_empty() {
            return None;
        }
        let rest = self.tensor.strip_prefix(self.layer.as_str())?;
        match rest.chars().next() {
            None => Some(width(&self.layer)),
            Some('.') => Some(width(&self.layer) + 1),
            Some(_) => None,
        }
    }

    /// Position inside the tensor, for example
    /// `expert 37 of 512 · gate+up 384/640 row pairs (60%)`. Compact variants
    /// drop the part, then the item percentage, which rounds down so it reads
    /// 100% only once all of the item's work is reported.
    fn write_position(&self, variant: usize, out: &mut String) -> fmt::Result {
        let start = out.len();
        if let Some((index, count)) = self.expert {
            write!(out, "expert {index} of {count}")?;
        }
        let part = variant == 0 && !self.part.is_empty();
        if part {
            separate(out, start);
            out.push_str(&self.part);
        }
        if self.total == 0 {
            return Ok(());
        }
        if part {
            out.push(' ');
        } else {
            separate(out, start);
        }
        write_count(out, &self.unit, self.completed)?;
        out.push('/');
        write_count(out, &self.unit, self.total)?;
        if self.unit != "bytes" && !self.unit.is_empty() {
            out.push(' ');
            out.push_str(&self.unit);
        }
        if variant < 2 {
            write!(out, " ({}%)", percent(self.completed, self.total))?;
        }
        Ok(())
    }
}

fn replace(buffer: &mut String, text: &str) {
    buffer.clear();
    buffer.push_str(text);
}

/// A live stage's latest state between redraws.
#[derive(Debug, Default)]
struct Context {
    /// Columns of the stage label, which indicatif draws as the prefix.
    label_columns: usize,
    item: Item,
    /// Present only for classified conversion work; generic stages keep their
    /// existing indicatif ETA.
    estimate: Option<WorkEstimate>,
    /// The current step has no value-work budget and is excluded from ETA.
    outside_eta: bool,
    /// Reused text of the frame being drawn.
    text: String,
}

impl Context {
    /// Everything after the label, laid out for the terminal's current width.
    fn write_frame(
        &self,
        unit: Unit,
        frame: Frame,
        state: &ProgressState,
        out: &mut String,
    ) -> fmt::Result {
        let columns = terminal_columns().unwrap_or(FALLBACK_COLUMNS);
        let first = columns.saturating_sub(self.label_columns + frame.lead());
        let below = columns.saturating_sub(INDENT);
        match frame {
            Frame::Running => {
                unit.write_header(state, first, out)?;
                out.push_str(LINE);
                self.write_time(unit, state, below, out)?;
            }
            Frame::Done => return unit.write_done(state, first, out),
            Frame::Stopped => unit.write_stopped(state, first, out)?,
        }
        self.item.write_lines(below, out)
    }

    /// Elapsed time and the ETA, then the recent rate when it fits. ETA
    /// wording is never shortened: its qualifiers are what keep it honest.
    fn write_time(
        &self,
        unit: Unit,
        state: &ProgressState,
        columns: usize,
        out: &mut String,
    ) -> fmt::Result {
        fit(out, columns, 2, |out, variant| {
            write!(out, "elapsed {}", FormattedDuration(state.elapsed()))?;
            out.push_str(FIELD);
            self.write_eta(state, out)?;
            if variant == 0 && state.pos() > 0 {
                out.push_str(FIELD);
                unit.write_rate(state, out)?;
            }
            Ok(())
        })
    }

    fn write_eta(&self, state: &ProgressState, out: &mut String) -> fmt::Result {
        let Some(estimate) = &self.estimate else {
            return write_remaining(state, out);
        };
        let remaining = estimate.remaining();
        if !remaining.finalizing(self.outside_eta) {
            out.push_str("ETA ");
        }
        remaining.write(self.outside_eta, out)
    }
}

/// Context stays drawable even if a panic poisoned it mid-update; the next
/// update rewrites it completely.
fn lock(context: &Mutex<Context>) -> MutexGuard<'_, Context> {
    context.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Live {
    bar: ProgressBar,
    context: Arc<Mutex<Context>>,
}

/// One live stage. Pass `&Stage` to the code doing the work.
///
/// Dropping an unfinished stage (an error return or panic) leaves a
/// "stopped at" line with the last reported position and context.
#[must_use = "dropping a stage reports it as stopped"]
pub struct Stage {
    live: Option<Live>,
    unit: Unit,
}

impl Stage {
    /// False for disabled progress. Guard any label formatting or context
    /// derivation with this so disabled stages cost no allocation.
    pub fn is_enabled(&self) -> bool {
        self.live.is_some()
    }

    /// Record `completed` more units of finished work. Atomic; terminal
    /// redraws are rate limited, so per-chunk calls are cheap.
    pub fn advance(&self, completed: u64) {
        if let Some(live) = &self.live {
            live.bar.inc(completed);
        }
    }

    /// Start timing only the next real chunk. Disabled progress does not read
    /// the clock; no rate includes time waiting between this and other chunks.
    pub(crate) fn start_work(&self) -> Option<Instant> {
        let live = self.live.as_ref()?;
        let mut context = lock(&live.context);
        if context.estimate.is_none() {
            return None;
        }
        context.outside_eta = false;
        drop(context);
        Some(Instant::now())
    }

    /// Complete one explicitly timed chunk and advance its remaining class
    /// budget alongside the unchanged overall value counter.
    pub(crate) fn advance_work(&self, class: WorkClass, completed: u64, started: Option<Instant>) {
        if let Some(live) = &self.live {
            let elapsed = started.map_or(Duration::ZERO, |start| start.elapsed());
            if let Some(estimate) = &mut lock(&live.context).estimate {
                estimate.record(class, completed, elapsed);
            }
            live.bar.inc(completed);
        }
    }

    /// Show a free-form item, such as a hashing or finalization step, in place
    /// of any earlier [`Detail`], whose operation, layer and expert would no
    /// longer describe the work. Drawn on the next redraw, shortened in the
    /// middle when the terminal is too narrow.
    pub fn set_item(&self, item: &str) {
        self.set_item_state(item, false);
    }

    /// Show a step whose duration is deliberately outside conversion's value
    /// ETA, including file hashing between files with values still remaining.
    pub(crate) fn set_uncounted_item(&self, item: &str) {
        self.set_item_state(item, true);
    }

    fn set_item_state(&self, item: &str, outside_eta: bool) {
        if let Some(live) = &self.live {
            let mut context = lock(&live.context);
            context.item.set_note(item);
            context.outside_eta = outside_eta;
        }
    }

    /// Show where the stage is within its current item. Call at chunk or band
    /// boundaries next to [`Stage::advance`]; each call replaces the previous
    /// detail by copying it into reused buffers, and only a redraw formats it,
    /// so frequent calls are cheap.
    pub fn set_detail(&self, detail: &Detail<'_>) {
        if let Some(live) = &self.live {
            lock(&live.context).item.set_detail(detail);
        }
    }

    /// End the stage. Reports "done" only when the position equals the total;
    /// any other position is reported as stopped, never rounded to complete.
    pub fn finish(mut self) {
        if let Some(live) = self.live.take() {
            let frame = if live.bar.length() == Some(live.bar.position()) {
                Frame::Done
            } else {
                Frame::Stopped
            };
            self.close(&live, frame);
        }
    }

    fn close(&self, live: &Live, frame: Frame) {
        // Join the ticker first so it cannot redraw the running frame between
        // the style change and the final draw.
        live.bar.disable_steady_tick();
        live.bar.set_style(self.unit.style(frame, &live.context));
        // Abandon keeps the reported position; indicatif's finish would set it
        // to the total.
        live.bar.abandon();
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(live) = self.live.take() {
            self.close(&live, Frame::Stopped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HT4: WorkClass = WorkClass::Encode(Encoding::Ht, Device::Cpu);
    const I4R: WorkClass = WorkClass::Encode(Encoding::I4r, Device::Cpu);
    const I4R_CUDA: WorkClass = WorkClass::Encode(Encoding::I4r, Device::Cuda);
    const I4R_HIP: WorkClass = WorkClass::Encode(Encoding::I4r, Device::Hip);

    fn plan(classes: &[(WorkClass, u64)]) -> WorkEstimate {
        let mut estimate = WorkEstimate::default();
        for &(class, values) in classes {
            estimate.add(class, values).unwrap();
        }
        estimate
    }

    fn expected(seconds: u64, learning: usize) -> WorkRemaining {
        WorkRemaining::Estimate {
            measured: Duration::from_secs(seconds),
            learning,
        }
    }

    #[test]
    fn mixed_rates_sum_remaining_work_including_partial_current_class() {
        let mut estimate = plan(&[(HT4, 1000), (I4R, 1000)]);
        estimate.record(HT4, 100, Duration::from_secs(10));
        assert_eq!(estimate.remaining(), expected(90, 1));
        estimate.record(I4R, 500, Duration::from_secs(1));
        assert_eq!(estimate.remaining(), expected(91, 0));
        estimate.record(HT4, 200, Duration::from_secs(20));
        assert_eq!(estimate.remaining(), expected(71, 0));
    }

    #[test]
    fn revisiting_a_class_keeps_its_rate_without_charging_other_class_time() {
        let mut estimate = plan(&[(HT4, 300), (I4R, 100)]);
        // The first HT tensor finishes; two later tensors remain in its class.
        estimate.record(HT4, 100, Duration::from_secs(10));
        estimate.record(I4R, 100, Duration::from_secs(100));
        assert_eq!(estimate.remaining(), expected(20, 0));
        // Neither the long I4R chunk nor an unreported idle/finalization gap
        // changes the HT rate. Only the next explicitly timed HT chunk does.
        estimate.record(HT4, 100, Duration::from_secs(10));
        assert_eq!(estimate.remaining(), expected(10, 0));
        estimate.record(HT4, 100, Duration::from_secs(10));
        assert_eq!(estimate.remaining(), WorkRemaining::Complete);
    }

    #[test]
    fn completed_class_rates_survive_a_later_budget_for_the_same_class() {
        let mut estimate = plan(&[(HT4, 100)]);
        estimate.record(HT4, 100, Duration::from_secs(10));
        assert_eq!(estimate.remaining(), WorkRemaining::Complete);
        estimate.add(HT4, 50).unwrap();
        assert_eq!(estimate.total(), 150);
        assert_eq!(estimate.remaining(), expected(5, 0));
    }

    #[test]
    fn operation_encoding_and_actual_backend_have_independent_rates() {
        let cpu = I4rBackend::Cpu;
        let classes = [
            HT4,
            WorkClass::encode(Encoding::Ht, HtBackend::Hip, cpu, Q4cpBackend::Cpu),
            WorkClass::encode(Encoding::Ht3, HtBackend::Cpu, cpu, Q4cpBackend::Cpu),
            WorkClass::encode(Encoding::Ht3, HtBackend::Hip, cpu, Q4cpBackend::Cpu),
            WorkClass::encode(Encoding::Ht, HtBackend::Cuda, cpu, Q4cpBackend::Cpu),
            WorkClass::encode(Encoding::Ht3, HtBackend::Cuda, cpu, Q4cpBackend::Cpu),
            I4R,
            I4R_CUDA,
            I4R_HIP,
            WorkClass::encode(Encoding::Fp8G128, HtBackend::Hip, I4rBackend::Cuda, Q4cpBackend::Cpu),
            WorkClass::ScanFp8Scale,
            WorkClass::Copy(Encoding::Bf16),
            WorkClass::encode(Encoding::Bf16, HtBackend::Hip, I4rBackend::Cuda, Q4cpBackend::Cpu),
            WorkClass::WriteRotation,
        ];
        let mut estimate = WorkEstimate::default();
        for class in classes {
            estimate.add(class, 200).unwrap();
        }
        assert_eq!(estimate.remaining(), expected(0, classes.len()));
        for (index, class) in classes.into_iter().enumerate() {
            estimate.record(class, 100, Duration::from_secs(index as u64 + 1));
            let sampled = index + 1;
            let seconds = (sampled * (sampled + 1) / 2) as u64;
            assert_eq!(
                estimate.remaining(),
                expected(seconds, classes.len() - sampled)
            );
        }
        let all = classes.len();
        assert_eq!(
            estimate.remaining(),
            expected((all * (all + 1) / 2) as u64, 0)
        );
    }

    #[test]
    fn ht_and_i4r_backends_select_devices_independently() {
        for ht in [HtBackend::Cpu, HtBackend::Hip, HtBackend::Cuda] {
            assert_eq!(WorkClass::encode(Encoding::I4r, ht, I4rBackend::Cpu, Q4cpBackend::Cpu), I4R);
            assert_eq!(
                WorkClass::encode(Encoding::I4r, ht, I4rBackend::Cuda, Q4cpBackend::Cpu),
                I4R_CUDA
            );
            assert_eq!(
                WorkClass::encode(Encoding::I4r, ht, I4rBackend::Hip, Q4cpBackend::Cpu),
                I4R_HIP
            );
            for i4r in [I4rBackend::Cpu, I4rBackend::Hip, I4rBackend::Cuda] {
                for encoding in [Encoding::Ht, Encoding::Ht3] {
                    let class = WorkClass::encode(encoding, ht, i4r, Q4cpBackend::Cpu);
                    assert_eq!(class, WorkClass::Encode(encoding, ht.into()));
                    assert_eq!(class.backend_label(), ht.label());
                }
                for encoding in [Encoding::Fp8G128, Encoding::Q4Cp, Encoding::Bf16] {
                    assert_eq!(WorkClass::encode(encoding, ht, i4r, Q4cpBackend::Cpu).backend_label(), "CPU");
                }
            }
        }
        assert_eq!(I4R.backend_label(), "CPU");
        assert_eq!(I4R_CUDA.backend_label(), "CUDA");
        assert_eq!(I4R_HIP.backend_label(), "HIP");
        assert_ne!(I4R_HIP.index(), I4R_CUDA.index());
        assert_ne!(I4R_HIP.index(), I4R.index());
        assert_ne!(
            I4R_HIP.index(),
            WorkClass::Encode(Encoding::Ht, Device::Hip).index()
        );
    }

    #[test]
    fn unmeasured_remaining_classes_are_explicitly_partial_or_learning() {
        let mut estimate = plan(&[(HT4, 100), (I4R, 100)]);
        assert_eq!(estimate.remaining(), expected(0, 2));
        estimate.record(HT4, 50, Duration::from_secs(5));
        assert_eq!(estimate.remaining(), expected(5, 1));
        let mut rendered = String::new();
        estimate.remaining().write(false, &mut rendered).unwrap();
        assert_eq!(
            rendered,
            format!(
                "partial ~{:#} + 1 unmeasured",
                HumanDuration(Duration::from_secs(5))
            )
        );
        estimate.record(HT4, 50, Duration::from_secs(5));
        assert_eq!(estimate.remaining(), expected(0, 1));
        rendered.clear();
        estimate.remaining().write(false, &mut rendered).unwrap();
        assert_eq!(rendered, "learning (1 class)");
    }

    #[test]
    fn completion_needs_no_rate_and_zero_duration_is_not_a_sample() {
        let mut estimate = plan(&[(HT4, 10)]);
        estimate.record(HT4, 0, Duration::from_secs(100));
        assert_eq!(estimate.remaining(), expected(0, 1));
        estimate.record(HT4, 5, Duration::ZERO);
        assert_eq!(estimate.remaining(), expected(0, 1));
        estimate.record(HT4, 5, Duration::ZERO);
        assert_eq!(estimate.remaining(), WorkRemaining::Complete);
        assert_eq!(WorkEstimate::default().remaining(), WorkRemaining::Complete);
        let mut rendered = String::new();
        estimate.remaining().write(false, &mut rendered).unwrap();
        assert_eq!(rendered, "finalizing (outside ETA)");
    }

    #[test]
    fn finalization_is_outside_eta_even_when_later_file_values_remain() {
        let mut estimate = plan(&[(HT4, 100)]);
        estimate.record(HT4, 50, Duration::from_secs(5));
        let remaining = estimate.remaining();
        let mut rendered = String::new();
        remaining.write(true, &mut rendered).unwrap();
        assert_eq!(rendered, "finalizing (outside ETA)");
        assert_eq!(estimate.remaining(), expected(5, 0));
    }

    #[test]
    fn work_budgets_reject_overflow_without_changing_existing_totals() {
        let mut estimate = plan(&[(HT4, u64::MAX)]);
        assert_eq!(estimate.add(I4R, 1), None);
        assert_eq!(estimate.total(), u64::MAX);
        assert_eq!(estimate.planned_values(I4R), 0);
    }
}

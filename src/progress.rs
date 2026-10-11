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
//! A live stage draws a short block. For the stage labels and context this
//! crate reports, each line normally fits in 80 columns, so the overall
//! counters never push the context off a narrow terminal (a longer line wraps
//! and the redraw accounts for it):
//!
//! ```text
//! ⠙ Convert passes [=========>              ] 512.00G/1.37T values
//!   12.34M values/s  elapsed 00:12:34  eta ~2h
//!   encode Ht3 on HIP · layers.12 · expert 37 (512 total) · gate+up
//!   384/1,280 rows (30%) · layers.12.mlp.experts.gate_up_proj.weight
//! ```
//!
//! The first two lines are the stage's overall work; the last two show the
//! latest [`Detail`] (or a single [`Stage::set_item`] label) and appear only
//! once the caller reports one. A stopped stage keeps them so a failure shows
//! where it stopped; a completed stage collapses to one "done" line.

use crate::codec::{Encoding, HtBackend};
use indicatif::{
    BinaryBytes, HumanCount, HumanDuration, ProgressBar, ProgressDrawTarget, ProgressState,
    ProgressStyle,
};
use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    io::IsTerminal as _,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

/// Redraw interval while a stage is live, so elapsed time, the spinner, the
/// latest detail and the rate estimate stay current during long blocks.
const TICK: Duration = Duration::from_millis(250);
const GAUGE_WIDTH: u64 = 24;
const RUNNING: &str = "{spinner} {prefix} [{gauge}] {work}\n  {rate}  elapsed {elapsed_precise}  eta {remaining}{detail}";
const DONE: &str = "{prefix}: done {amount} in {elapsed_precise} ({rate})";
const STOPPED: &str = "{prefix}: stopped at {work} after {elapsed_precise}{detail}";
/// Separates fields of a detail line.
const FIELD: &str = " · ";

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
        let context = Arc::new(Mutex::new(Context {
            estimate,
            ..Context::default()
        }));
        let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr())
            .with_style(unit.style(RUNNING, &context))
            .with_prefix(label);
        bar.enable_steady_tick(TICK);
        Stage {
            live: Some(Live { bar, context }),
            unit,
        }
    }
}

/// End-to-end value-processing passes. Backend selection applies only to HT
/// encoding; reads, rotations, hashing and writes within that chunk are included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkClass {
    Copy(Encoding),
    Encode(Encoding, HtBackend),
    WriteRotation,
    ScanFp8Scale,
}

const ENCODING_CLASSES: usize = 11;
const WORK_CLASSES: usize = ENCODING_CLASSES + 2 + 3 + 2;

impl WorkClass {
    pub(crate) fn encode(encoding: Encoding, configured_backend: HtBackend) -> Self {
        let backend = if encoding.ht_bits().is_some() {
            configured_backend
        } else {
            HtBackend::Cpu
        };
        Self::Encode(encoding, backend)
    }

    pub(crate) fn backend_label(self) -> &'static str {
        match self {
            Self::Encode(_, HtBackend::Hip) => "HIP",
            _ => "CPU",
        }
    }

    /// Fixed, dense slots: updates never allocate or search the conversion plan.
    fn index(self) -> usize {
        match self {
            Self::Encode(encoding, HtBackend::Cpu) => match encoding {
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
            Self::Encode(Encoding::Ht, HtBackend::Hip) => ENCODING_CLASSES,
            Self::Encode(Encoding::Ht3, HtBackend::Hip) => ENCODING_CLASSES + 1,
            Self::Encode(_, HtBackend::Hip) => unreachable!("only HT encoding runs on HIP"),
            Self::Copy(Encoding::Bf16) => ENCODING_CLASSES + 2,
            Self::Copy(Encoding::F32) => ENCODING_CLASSES + 3,
            Self::Copy(Encoding::I64) => ENCODING_CLASSES + 4,
            Self::Copy(_) => unreachable!("unsupported direct-copy encoding"),
            Self::WriteRotation => ENCODING_CLASSES + 5,
            Self::ScanFp8Scale => ENCODING_CLASSES + 6,
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
    fn write(self, outside_eta: bool, out: &mut dyn fmt::Write) -> fmt::Result {
        if outside_eta || self == Self::Complete {
            return out.write_str("finalizing (outside ETA)");
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
    fn style(self, template: &str, context: &Arc<Mutex<Context>>) -> ProgressStyle {
        let detail_context = Arc::clone(context);
        let eta_context = Arc::clone(context);
        ProgressStyle::with_template(template)
            .expect("static progress template")
            .with_key(
                "detail",
                move |_: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = lock(&detail_context).write_lines(out);
                },
            )
            .with_key(
                "gauge",
                |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = write_gauge(state, out);
                },
            )
            .with_key(
                "work",
                move |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = self.write_work(state, out);
                },
            )
            .with_key(
                "amount",
                move |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = self.write_amount(out, state.pos());
                },
            )
            .with_key(
                "rate",
                move |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = self.write_rate(state, out);
                },
            )
            .with_key(
                "remaining",
                move |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let context = lock(&eta_context);
                    let _ = match &context.estimate {
                        Some(estimate) => estimate.remaining().write(context.outside_eta, out),
                        None => write_remaining(state, out),
                    };
                },
            )
    }
}

/// Live gauge that never fills: a stage whose position already reached its
/// total may still be running final checks, and only [`Stage::finish`] may
/// report completion (finished stages use templates without a gauge).
fn write_gauge(state: &ProgressState, out: &mut dyn fmt::Write) -> fmt::Result {
    let filled = match state.len() {
        Some(total) if total > 0 => {
            let filled = u128::from(state.pos()) * u128::from(GAUGE_WIDTH) / u128::from(total);
            filled.min(u128::from(GAUGE_WIDTH - 1)) as u64
        }
        _ => 0,
    };
    for _ in 0..filled {
        out.write_char('=')?;
    }
    out.write_char('>')?;
    for _ in filled + 1..GAUGE_WIDTH {
        out.write_char(' ')?;
    }
    Ok(())
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

/// Remaining time at the recent rate, or `?` when unknown: before measured
/// progress (instead of indicatif's zero placeholder) and once all counted
/// work is reported but the stage is still running its closing checks.
fn write_remaining(state: &ProgressState, out: &mut dyn fmt::Write) -> fmt::Result {
    let rate = state.per_sec();
    let counted_all = state.len().is_some_and(|total| state.pos() >= total);
    if state.pos() == 0 || counted_all || !rate.is_finite() || rate <= 0.0 {
        return out.write_char('?');
    }
    write!(out, "~{:#}", HumanDuration(state.eta()))
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
/// it into reused buffers and a disabled stage ignores it. Empty strings and
/// `None` fields are left out of the display.
#[derive(Clone, Copy, Debug)]
pub struct Detail<'a> {
    /// Phase step, such as `encode`, `copy`, `scan` or `decode`.
    pub operation: &'a str,
    /// Output tensor name.
    pub tensor: &'a str,
    /// Scope-qualified location such as `layers.2`, `mtp.layers.0` or
    /// `visual.blocks.3`; `None` for tensors outside any layer.
    pub layer: Option<&'a str>,
    /// Store label of the tensor's encoding.
    pub encoding: &'a str,
    /// Device actually running this step, such as `CPU` or `HIP`.
    pub backend: &'a str,
    /// `(index, count)`: zero-based index of the current expert and the number
    /// of experts in the tensor. Shown as `expert 7 (512 total)`, so the last
    /// expert of 512 reads `expert 511 (512 total)`.
    pub expert: Option<(u64, u64)>,
    /// Part of the expert or tensor, such as `gate`, `up` or `gate+up`.
    pub part: Option<&'a str>,
    /// Work on the current item finished so far, out of `total`, in `unit`
    /// (for example `rows`, `bands`, `values` or `bytes`). A `total` of zero
    /// leaves item progress out.
    pub completed: u64,
    pub total: u64,
    pub unit: &'a str,
}

impl Detail<'_> {
    /// Operation, encoding and backend, then location, for example
    /// `encode Ht3 on HIP · layers.12 · expert 37 (512 total) · gate+up`.
    fn write_scope(&self, out: &mut String) -> fmt::Result {
        for word in [self.operation, self.encoding] {
            if !word.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(word);
            }
        }
        if !self.backend.is_empty() {
            if !out.is_empty() {
                out.push_str(" on ");
            }
            out.push_str(self.backend);
        }
        if let Some(layer) = self.layer {
            push_field(out, layer);
        }
        if let Some((index, count)) = self.expert {
            separate(out);
            write!(out, "expert {index} ({count} total)")?;
        }
        if let Some(part) = self.part {
            push_field(out, part);
        }
        Ok(())
    }

    /// Item progress, then the tensor name, for example
    /// `384/1,280 rows (30%) · layers.12.mlp.experts.gate_up_proj.weight`.
    /// The percentage rounds down, so it reads 100% only once all of the
    /// item's work is reported.
    fn write_item(&self, out: &mut String) -> fmt::Result {
        if self.total > 0 {
            write_count(out, self.unit, self.completed)?;
            out.push('/');
            write_count(out, self.unit, self.total)?;
            if self.unit != "bytes" && !self.unit.is_empty() {
                out.push(' ');
                out.push_str(self.unit);
            }
            let percent = u128::from(self.completed) * 100 / u128::from(self.total);
            write!(out, " ({percent}%)")?;
        }
        if !self.tensor.is_empty() {
            push_field(out, self.tensor);
        }
        Ok(())
    }
}

/// Start another field of a detail line.
fn separate(out: &mut String) {
    if !out.is_empty() {
        out.push_str(FIELD);
    }
}

fn push_field(out: &mut String, field: &str) {
    separate(out);
    out.push_str(field);
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

/// Latest context of a live stage, drawn below its overall counters. Updated
/// in place so steady-state updates reuse the same buffers.
#[derive(Debug, Default)]
struct Context {
    /// Operation, encoding, backend and location; empty after
    /// [`Stage::set_item`].
    scope: String,
    /// Item progress and tensor name, or a free-form item label.
    item: String,
    /// Present only for classified conversion work; generic stages keep their
    /// existing indicatif ETA.
    estimate: Option<WorkEstimate>,
    /// The current step has no value-work budget and is excluded from ETA.
    outside_eta: bool,
}

impl Context {
    /// Each non-empty line, indented under the stage's counters.
    fn write_lines(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        for line in [&self.scope, &self.item] {
            if !line.is_empty() {
                out.write_str("\n  ")?;
                out.write_str(line)?;
            }
        }
        Ok(())
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
    /// longer describe the work. Drawn on the next redraw.
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
            context.scope.clear();
            context.item.clear();
            context.item.push_str(item);
            context.outside_eta = outside_eta;
        }
    }

    /// Show where the stage is within its current item. Call at chunk or band
    /// boundaries next to [`Stage::advance`]; each call replaces the previous
    /// detail and is drawn on the next redraw, so frequent calls are cheap.
    pub fn set_detail(&self, detail: &Detail<'_>) {
        if let Some(live) = &self.live {
            let mut context = lock(&live.context);
            let Context { scope, item, .. } = &mut *context;
            scope.clear();
            item.clear();
            let _ = detail.write_scope(scope);
            let _ = detail.write_item(item);
        }
    }

    /// End the stage. Reports "done" only when the position equals the total;
    /// any other position is reported as stopped, never rounded to complete.
    pub fn finish(mut self) {
        if let Some(live) = self.live.take() {
            let template = if live.bar.length() == Some(live.bar.position()) {
                DONE
            } else {
                STOPPED
            };
            self.close(&live, template);
        }
    }

    fn close(&self, live: &Live, template: &str) {
        // Join the ticker first so it cannot redraw the running template
        // between the style change and the final draw.
        live.bar.disable_steady_tick();
        live.bar.set_style(self.unit.style(template, &live.context));
        // Abandon keeps the reported position; indicatif's finish would set it
        // to the total.
        live.bar.abandon();
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(live) = self.live.take() {
            self.close(&live, STOPPED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HT4: WorkClass = WorkClass::Encode(Encoding::Ht, HtBackend::Cpu);
    const I4R: WorkClass = WorkClass::Encode(Encoding::I4r, HtBackend::Cpu);

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
        let classes = [
            HT4,
            WorkClass::encode(Encoding::Ht, HtBackend::Hip),
            WorkClass::encode(Encoding::Ht3, HtBackend::Cpu),
            WorkClass::encode(Encoding::Ht3, HtBackend::Hip),
            I4R,
            WorkClass::encode(Encoding::Fp8G128, HtBackend::Hip),
            WorkClass::ScanFp8Scale,
            WorkClass::Copy(Encoding::Bf16),
            WorkClass::encode(Encoding::Bf16, HtBackend::Hip),
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
        assert_eq!(estimate.remaining(), expected(55, 0));
        assert_eq!(WorkClass::encode(Encoding::I4r, HtBackend::Hip), I4R);
        assert_eq!(
            WorkClass::encode(Encoding::Fp8G128, HtBackend::Hip).backend_label(),
            "CPU"
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

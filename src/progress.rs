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
//! total. Rates and remaining time extrapolate the recent rate of that unit;
//! they are estimates, not predictions, for work whose cost per unit varies.
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

use indicatif::{
    BinaryBytes, HumanCount, HumanDuration, ProgressBar, ProgressDrawTarget, ProgressState,
    ProgressStyle,
};
use std::{
    borrow::Cow,
    fmt::{self, Write as _},
    io::IsTerminal as _,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
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
        if !self.enabled {
            return Stage { live: None, unit };
        }
        let context = Arc::default();
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

/// Unit of a stage's total and of every [`Stage::advance`] amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Bytes,
    Values,
    Tensors,
}

impl Unit {
    fn style(self, template: &str, context: &Arc<Mutex<Context>>) -> ProgressStyle {
        let context = Arc::clone(context);
        ProgressStyle::with_template(template)
            .expect("static progress template")
            .with_key(
                "detail",
                move |_: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = lock(&context).write_lines(out);
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
                |state: &ProgressState, out: &mut dyn fmt::Write| {
                    let _ = write_remaining(state, out);
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

    /// Show a free-form item, such as a hashing or finalization step, in place
    /// of any earlier [`Detail`], whose operation, layer and expert would no
    /// longer describe the work. Drawn on the next redraw.
    pub fn set_item(&self, item: &str) {
        if let Some(live) = &self.live {
            let mut context = lock(&live.context);
            context.scope.clear();
            context.item.clear();
            context.item.push_str(item);
        }
    }

    /// Show where the stage is within its current item. Call at chunk or band
    /// boundaries next to [`Stage::advance`]; each call replaces the previous
    /// detail and is drawn on the next redraw, so frequent calls are cheap.
    pub fn set_detail(&self, detail: &Detail<'_>) {
        if let Some(live) = &self.live {
            let mut context = lock(&live.context);
            let Context { scope, item } = &mut *context;
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

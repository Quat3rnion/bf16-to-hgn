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

use indicatif::{
    BinaryBytes, HumanCount, HumanDuration, ProgressBar, ProgressDrawTarget, ProgressState,
    ProgressStyle,
};
use std::{borrow::Cow, fmt, io::IsTerminal as _, time::Duration};

/// Redraw interval while a stage is live, so elapsed time, the spinner, the
/// latest item label and the rate estimate stay current during long blocks.
const TICK: Duration = Duration::from_millis(250);
const GAUGE_WIDTH: u64 = 24;
const RUNNING: &str =
    "{spinner} {prefix} [{gauge}] {work}  {rate}  {elapsed_precise}  eta {remaining}  {wide_msg}";
const DONE: &str = "{prefix}: done {amount} in {elapsed_precise} ({rate})";
const STOPPED: &str = "{prefix}: stopped at {work} after {elapsed_precise}  {wide_msg}";

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
            return Stage { bar: None, unit };
        }
        let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr())
            .with_style(unit.style(RUNNING))
            .with_prefix(label);
        bar.enable_steady_tick(TICK);
        Stage {
            bar: Some(bar),
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
    fn style(self, template: &str) -> ProgressStyle {
        ProgressStyle::with_template(template)
            .expect("static progress template")
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

/// One live stage. Pass `&Stage` to the code doing the work.
///
/// Dropping an unfinished stage (an error return or panic) leaves a
/// "stopped at" line with the last reported position and item.
#[must_use = "dropping a stage reports it as stopped"]
pub struct Stage {
    bar: Option<ProgressBar>,
    unit: Unit,
}

impl Stage {
    /// False for disabled progress. Guard any label formatting with this so
    /// disabled stages cost no allocation.
    pub fn is_enabled(&self) -> bool {
        self.bar.is_some()
    }

    /// Record `completed` more units of finished work. Atomic; terminal
    /// redraws are rate limited, so per-chunk calls are cheap.
    pub fn advance(&self, completed: u64) {
        if let Some(bar) = &self.bar {
            bar.inc(completed);
        }
    }

    /// Show the item now being processed, normally a tensor name. Redraws are
    /// coalesced to the terminal refresh rate; the latest label is shown on
    /// the next tick.
    pub fn set_item(&self, item: &str) {
        if let Some(bar) = &self.bar {
            bar.set_message(item.to_owned());
        }
    }

    /// End the stage. Reports "done" only when the position equals the total;
    /// any other position is reported as stopped, never rounded to complete.
    pub fn finish(mut self) {
        if let Some(bar) = self.bar.take() {
            let template = if bar.length() == Some(bar.position()) {
                DONE
            } else {
                STOPPED
            };
            self.close(&bar, template);
        }
    }

    fn close(&self, bar: &ProgressBar, template: &str) {
        // Join the ticker first so it cannot redraw the running template
        // between the style change and the final draw.
        bar.disable_steady_tick();
        bar.set_style(self.unit.style(template));
        // Abandon keeps the reported position; indicatif's finish would set it
        // to the total.
        bar.abandon();
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(bar) = self.bar.take() {
            self.close(&bar, STOPPED);
        }
    }
}

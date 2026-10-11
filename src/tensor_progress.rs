use crate::{
    codec::Encoding,
    progress::{Detail, Stage},
    TensorPlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Projection {
    Gate,
    Up,
    GateUp,
}

/// Output-plan geometry only: source importers may flatten or fuse axes.
#[derive(Debug, PartialEq, Eq)]
struct RowProgress {
    expert: Option<(u64, u64)>,
    part: Option<Projection>,
    completed: u64,
    total: u64,
}

fn layer(name: &str) -> Option<&str> {
    for prefix in ["layers.", "mtp.layers.", "visual.blocks."] {
        if let Some(tail) = name.strip_prefix(prefix) {
            let digits = tail.split('.').next()?;
            if !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit()) {
                return Some(&name[..prefix.len() + digits.len()]);
            }
        }
    }
    None
}

fn row_progress(
    tensor: &TensorPlan,
    row: u64,
    take: u64,
    finished: bool,
    paired: bool,
) -> Option<RowProgress> {
    let (_, axes) = tensor.shape.split_last()?;
    let rows = axes.iter().try_fold(1u64, |n, &axis| n.checked_mul(axis))?;
    let end = row.checked_add(take)?;
    if take == 0 || end > rows {
        return None;
    }
    let completed = if finished { end } else { row };
    let mut work = RowProgress {
        expert: None,
        part: None,
        completed,
        total: rows,
    };
    let experts = tensor.shape.len() == 3
        && (tensor.encoding == Encoding::I4r || tensor.name.contains(".experts."));
    if !experts {
        return Some(work);
    }
    let expert_rows = tensor.shape[1];
    if expert_rows == 0 || row / expert_rows != (end - 1) / expert_rows {
        // A chunk crossing experts cannot honestly name a single expert.
        return Some(work);
    }
    work.expert = Some((row / expert_rows, tensor.shape[0]));
    work.completed = completed - row / expert_rows * expert_rows;
    work.total = expert_rows;
    let gate_up =
        tensor.name.ends_with(".gate_up_proj.weight") || tensor.name.ends_with(".gate_up");
    if gate_up && expert_rows % 2 == 0 {
        let half = expert_rows / 2;
        let local = row % expert_rows;
        if local / half == (local + take - 1) / half {
            if paired && local < half {
                work.part = Some(Projection::GateUp);
            } else {
                work.part = Some(if local < half {
                    Projection::Gate
                } else {
                    Projection::Up
                });
            }
            work.completed -= local / half * half;
            work.total = half;
        }
    }
    Some(work)
}

pub(crate) fn encoding_name(encoding: Encoding) -> &'static str {
    match encoding {
        Encoding::Bf16 => "BF16",
        Encoding::F16 => "F16",
        Encoding::F32 => "F32",
        Encoding::I64 => "I64",
        Encoding::Q4Cp => "Q4Cp",
        Encoding::Q8G64 => "Q8G64",
        Encoding::Q6G64 => "Q6G64",
        Encoding::Fp8G128 => "FP8G128",
        Encoding::Ht => "HT4",
        Encoding::Ht3 => "HT3",
        Encoding::I4r => "I4R",
    }
}

pub(crate) fn item(
    stage: &Stage,
    tensor: &TensorPlan,
    operation: &str,
    backend: &str,
    completed: u64,
    total: u64,
    unit: &str,
    part: Option<&str>,
) {
    if !stage.is_enabled() {
        return;
    }
    stage.set_detail(&Detail {
        operation,
        tensor: &tensor.name,
        layer: layer(&tensor.name),
        encoding: encoding_name(tensor.encoding),
        backend,
        expert: None,
        part,
        completed,
        total,
        unit,
    });
}

pub(crate) fn rows(
    stage: &Stage,
    tensor: &TensorPlan,
    operation: &str,
    backend: &str,
    row: usize,
    take: usize,
    finished: bool,
    paired: bool,
) {
    if !stage.is_enabled() {
        return;
    }
    if let Some(work) = row_progress(tensor, row as u64, take as u64, finished, paired) {
        stage.set_detail(&Detail {
            operation,
            tensor: &tensor.name,
            layer: layer(&tensor.name),
            encoding: encoding_name(tensor.encoding),
            backend,
            expert: work.expert,
            part: work.part.map(|part| match part {
                Projection::Gate => "gate",
                Projection::Up => "up",
                Projection::GateUp => "gate+up",
            }),
            completed: work.completed,
            total: work.total,
            unit: if paired { "row pairs" } else { "rows" },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Origin;

    fn tensor(name: &str, shape: &[u64], encoding: Encoding) -> TensorPlan {
        TensorPlan {
            name: name.into(),
            shape: shape.into(),
            encoding,
            origin: Origin::Tensor {
                name: "flattened.source".into(),
            },
            rotation: None,
        }
    }

    #[test]
    fn expert_crossing_never_reports_the_last_expert_as_the_whole_band() {
        let plan = tensor(
            "layers.2.mlp.experts.down_proj.weight",
            &[3, 256, 128],
            Encoding::Ht,
        );
        let last = row_progress(&plan, 128, 128, true, false).unwrap();
        assert_eq!(last.expert, Some((0, 3)));
        assert_eq!((last.completed, last.total), (256, 256));
        let next = row_progress(&plan, 256, 128, false, false).unwrap();
        assert_eq!(next.expert, Some((1, 3)));
        assert_eq!((next.completed, next.total), (0, 256));
        let crossing = row_progress(&plan, 192, 128, true, false).unwrap();
        assert_eq!(crossing.expert, None);
        assert_eq!((crossing.completed, crossing.total), (320, 768));
        assert!(row_progress(&plan, 700, 128, false, false).is_none());
    }

    #[test]
    fn paired_encoding_and_sequential_scan_use_different_row_coordinates() {
        let plan = tensor(
            "layers.0.mlp.experts.gate_up_proj.weight",
            &[2, 512, 128],
            Encoding::Ht3,
        );
        let gate = row_progress(&plan, 512 + 128, 128, true, false).unwrap();
        let up = row_progress(&plan, 512 + 256, 128, false, false).unwrap();
        assert_eq!(gate.expert, up.expert);
        assert_eq!((gate.completed, gate.total), (256, 256));
        assert_eq!((up.completed, up.total), (0, 256));
        assert_eq!(gate.part, Some(Projection::Gate));
        assert_eq!(up.part, Some(Projection::Up));
        let pair = row_progress(&plan, 512 + 128, 128, true, true).unwrap();
        assert_eq!((pair.completed, pair.total), (256, 256));
        assert_eq!(pair.expert, Some((1, 2)));
        assert_eq!(pair.part, Some(Projection::GateUp));
        let crossing = row_progress(&plan, 192, 128, true, false).unwrap();
        assert_eq!(crossing.expert, Some((0, 2)));
        assert_eq!(crossing.part, None);
        assert_eq!((crossing.completed, crossing.total), (320, 512));
    }

    #[test]
    fn rank_three_convolution_and_flattened_unknown_matrix_are_not_experts() {
        for plan in [
            tensor(
                "layers.4.linear_attn.conv1d.weight",
                &[10240, 1, 4],
                Encoding::Bf16,
            ),
            tensor("fixture.experts.down", &[512, 128], Encoding::Ht),
            tensor("unknown.weight", &[512, 128], Encoding::I4r),
            tensor("unknown.weight", &[2, 256, 128], Encoding::Q4Cp),
        ] {
            let work = row_progress(&plan, 0, 128, true, false).unwrap();
            assert_eq!(work.expert, None);
            assert_eq!(work.part, None);
            assert_eq!(work.completed, 128);
            assert!(work.total >= work.completed);
        }
        let fixture = tensor("fixture.quantized", &[2, 256, 128], Encoding::I4r);
        assert_eq!(
            row_progress(&fixture, 256, 128, false, false)
                .unwrap()
                .expert,
            Some((1, 2))
        );
    }

    #[test]
    fn layer_scope_is_not_confused_with_other_numeric_components() {
        assert_eq!(
            layer("layers.12.mlp.experts.down_proj.weight"),
            Some("layers.12")
        );
        assert_eq!(
            layer("mtp.layers.0.self_attn.q_proj.weight"),
            Some("mtp.layers.0")
        );
        assert_eq!(
            layer("visual.blocks.3.attn.qkv.weight"),
            Some("visual.blocks.3")
        );
        for name in [
            "layers.x.weight",
            "layers..weight",
            "model.layers.2.weight",
            "ngram.shard_3",
        ] {
            assert_eq!(layer(name), None);
        }
    }
}

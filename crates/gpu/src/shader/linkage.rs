//! Component-wise semantic linkage, shared by frontends and native emission.
use super::*;
use std::collections::BTreeMap;

impl VerifiedShaderIr {
    /// Remove user outputs not consumed by rasterization, then their dead input
    /// and arithmetic dependencies. Call only for the final pre-raster stage
    /// without transform feedback. Builtins remain observable. This is compile-
    /// time specialization, not a substitute for disabled-attribute semantics.
    /// Branching programs are left intact until CFG liveness is implemented.
    /// https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-iointerfaces
    #[must_use]
    pub fn prune_raster_outputs(
        &self,
        consumed: impl IntoIterator<Item = (ShaderIoLocation, u8)>,
    ) -> Self {
        assert!(matches!(
            self.ir().stage(),
            ShaderStage::Vertex | ShaderStage::TessellationEvaluation | ShaderStage::Geometry
        ));
        let consumed: BTreeSet<_> = consumed.into_iter().collect();
        let observable = |location, component| {
            !matches!(location, ShaderIoLocation::Generic(_))
                || consumed.contains(&(location, component))
        };
        let mut result = self.clone();
        result.0.outputs = self
            .ir()
            .outputs()
            .iter()
            .filter(|e| observable(e.location(), e.component()))
            .copied()
            .collect();
        let mut code = Vec::with_capacity(self.ir().instructions().len());
        for instruction in self.ir().instructions() {
            if let ShaderOperation::StoreOutput {
                sources,
                location,
                first_component,
                scalar_type,
            } = instruction.operation()
                && sources
                    .iter()
                    .enumerate()
                    .any(|(i, _)| !observable(*location, *first_component + i as u8))
            {
                for (i, source) in sources.iter().enumerate() {
                    let component = *first_component + i as u8;
                    if observable(*location, component) {
                        code.push(ShaderInstruction::new(
                            instruction.source(),
                            instruction.predicate(),
                            ShaderOperation::StoreOutput {
                                sources: vec![*source].into(),
                                location: *location,
                                first_component: component,
                                scalar_type: *scalar_type,
                            },
                        ));
                    }
                }
            } else {
                code.push(instruction.clone());
            }
        }
        result.0.instructions = code.into();
        let Ok(live) = liveness::live_instructions(result.ir()) else {
            return self.clone();
        };
        result.0.instructions = result
            .0
            .instructions
            .iter()
            .zip(live)
            .filter(|(_, keep)| *keep)
            .map(|(i, _)| i.clone())
            .collect();
        // Only vertex-fetch declarations disappear here. Patch-stage input
        // declarations still participate in upstream TCS/TES linkage and array
        // cardinality validation, even when their loads are dead.
        if result.ir().stage() != ShaderStage::Vertex {
            return result;
        }
        let mut inputs = BTreeSet::new();
        for instruction in result.ir().instructions() {
            match instruction.operation() {
                ShaderOperation::LoadInput {
                    destinations,
                    location,
                    first_component,
                    ..
                } => {
                    for i in 0..destinations.len() {
                        inputs.insert((*location, *first_component + i as u8));
                    }
                }
                ShaderOperation::LoadControlPoint {
                    location,
                    component,
                    output: false,
                    ..
                }
                | ShaderOperation::InterpolateInput {
                    location,
                    component,
                    ..
                } => {
                    inputs.insert((*location, *component));
                }
                _ => {}
            }
        }
        result.0.inputs = result
            .ir()
            .inputs()
            .iter()
            .filter(|e| inputs.contains(&(e.location(), e.component())))
            .copied()
            .collect();
        result
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderStageInterfaceError {
    pub producer: ShaderStage,
    pub consumer: ShaderStage,
    pub location: ShaderIoLocation,
    pub component: u8,
    pub reason: &'static str,
}

/// Validate one adjacent active stage pair on a translation/cache miss. System
/// generated inputs do not need a preceding shader output; per-patch and
/// per-vertex attributes do. Interpolation belongs to the fragment consumer.
/// https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-iointerfaces
pub fn validate_shader_stage_link(
    producer: &ShaderIr,
    consumer: &ShaderIr,
) -> Result<(), ShaderStageInterfaceError> {
    let outputs: BTreeMap<_, _> = producer
        .outputs()
        .iter()
        .map(|e| ((e.location(), e.component()), e.scalar_type()))
        .collect();
    for input in consumer.inputs() {
        let linked = match input.location() {
            ShaderIoLocation::Generic(_)
            | ShaderIoLocation::Color(_)
            | ShaderIoLocation::Patch(_) => true,
            ShaderIoLocation::Position | ShaderIoLocation::PointSize => {
                consumer.stage() != ShaderStage::Fragment
            }
            ShaderIoLocation::TessLevelOuter | ShaderIoLocation::TessLevelInner => {
                producer.stage() == ShaderStage::TessellationControl
            }
            _ => false,
        };
        if !linked {
            continue;
        }
        let key = (input.location(), input.component());
        let reason = match outputs.get(&key) {
            None => "stage input component has no output in the preceding active stage",
            Some(ty) if *ty != input.scalar_type() => {
                "adjacent stage output/input scalar types differ"
            }
            Some(_) => continue,
        };
        return Err(ShaderStageInterfaceError {
            producer: producer.stage(),
            consumer: consumer.stage(),
            location: input.location(),
            component: input.component(),
            reason,
        });
    }
    Ok(())
}

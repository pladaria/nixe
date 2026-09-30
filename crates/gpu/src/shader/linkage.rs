//! Component-wise semantic linkage, shared by frontends and native emission.
use super::*;
use std::collections::BTreeMap;

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

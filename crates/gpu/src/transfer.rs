//! Byte-addressed, ordered device transfers independent of guest GPU classes.
use crate::BufferRegion;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferLayout {
    Pitch {
        pitch: u32,
    },
    BlockLinear {
        row_bytes: u32,
        origin_x_bytes: u32,
        origin_y: u32,
        block_height_log2: u8,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferComponent {
    Source(u8),
    ConstantA,
    ConstantB,
    Preserve,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferTransform {
    pub source: BufferRegion,
    pub destination: BufferRegion,
    pub source_layout: TransferLayout,
    pub destination_layout: TransferLayout,
    pub width: u32,
    pub height: u32,
    pub component_bytes: u8,
    pub source_components: u8,
    pub destination_components: u8,
    pub components: [TransferComponent; 4],
    pub constant_a: u32,
    pub constant_b: u32,
}

impl BufferTransform {
    /// Validate at the device consumer before allocating or encoding work.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 {
            return Err("device transfer has an empty extent");
        }
        if !matches!(self.component_bytes, 1 | 2 | 4)
            || !(1..=4).contains(&self.source_components)
            || !(1..=4).contains(&self.destination_components)
        {
            return Err("device transfer has invalid component geometry");
        }
        for component in &self.components[..usize::from(self.destination_components)] {
            if matches!(component, TransferComponent::Source(index) if *index >= self.source_components)
            {
                return Err("device transfer references a missing source component");
            }
        }
        for (layout, components, range) in [
            (
                self.source_layout,
                self.source_components,
                self.source.range,
            ),
            (
                self.destination_layout,
                self.destination_components,
                self.destination.range,
            ),
        ] {
            let element = u64::from(components) * u64::from(self.component_bytes);
            let x = u64::from(self.width - 1) * element;
            let y = u64::from(self.height - 1);
            let last = match layout {
                TransferLayout::Pitch { pitch } => {
                    if u64::from(pitch) < u64::from(self.width) * element {
                        return Err("device transfer pitch is smaller than its row");
                    }
                    y * u64::from(pitch) + x
                }
                TransferLayout::BlockLinear {
                    row_bytes,
                    origin_x_bytes,
                    origin_y,
                    block_height_log2,
                } => {
                    if block_height_log2 > 5
                        || !element.is_power_of_two()
                        || !u64::from(origin_x_bytes).is_multiple_of(element)
                    {
                        return Err("device transfer has unsupported block geometry");
                    }
                    let x = x + u64::from(origin_x_bytes);
                    let y = y + u64::from(origin_y);
                    if x + element > u64::from(row_bytes) {
                        return Err("device transfer exceeds its tiled surface width");
                    }
                    let height = 1_u64 << block_height_log2;
                    let row = u64::from(row_bytes).div_ceil(64) * 512 * height;
                    if row > u64::from(u32::MAX) {
                        return Err("device transfer tiled row overflows shader addressing");
                    }
                    y / (8 * height) * row
                        + x / 64 * 512 * height
                        + y % (8 * height) / 8 * 512
                        + x % 64 / 32 * 256
                        + y % 8 / 2 * 64
                        + x % 32 / 16 * 32
                        + y % 2 * 16
                        + x % 16
                }
            };
            if last + element > range.size() || range.end() > u64::from(u32::MAX) - 3 {
                return Err("device transfer exceeds its retained region or shader addressing");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferId, BufferRange};

    #[test]
    fn tiled_elements_require_alignment_and_a_retained_surface() {
        let region = BufferRegion {
            buffer: BufferId::new(1),
            range: BufferRange::new(0, 4096).unwrap(),
        };
        let mut copy = BufferTransform {
            source: region,
            destination: region,
            source_layout: TransferLayout::BlockLinear {
                row_bytes: 64,
                origin_x_bytes: 4,
                origin_y: 0,
                block_height_log2: 0,
            },
            destination_layout: TransferLayout::Pitch { pitch: 16 },
            width: 4,
            height: 4,
            component_bytes: 4,
            source_components: 1,
            destination_components: 1,
            components: [TransferComponent::Source(0); 4],
            constant_a: 0,
            constant_b: 0,
        };
        assert_eq!(copy.validate(), Ok(()));
        if let TransferLayout::BlockLinear { origin_x_bytes, .. } = &mut copy.source_layout {
            *origin_x_bytes = 1;
        }
        assert_eq!(
            copy.validate(),
            Err("device transfer has unsupported block geometry")
        );
        copy.source_layout = TransferLayout::Pitch { pitch: 16 };
        copy.source.range = BufferRange::new(0, 8).unwrap();
        assert_eq!(
            copy.validate(),
            Err("device transfer exceeds its retained region or shader addressing")
        );
    }
}

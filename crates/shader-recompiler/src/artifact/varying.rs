use crate::ir::Attribute;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VaryingState {
    pub mask: [u64; 8],
}

impl VaryingState {
    pub fn set(&mut self, attribute: Attribute, state: bool) {
        let index = attribute.index();
        let bit = 1 << (index % 64);
        if state {
            self.mask[index / 64] |= bit;
        } else {
            self.mask[index / 64] &= !bit;
        }
    }

    pub fn get(&self, attribute: Attribute) -> bool {
        self.bit(attribute.index())
    }

    fn bit(&self, index: usize) -> bool {
        self.mask[index / 64] & (1 << (index % 64)) != 0
    }

    pub fn any_component(&self, base: Attribute) -> bool {
        (0..4).any(|c| self.bit(base.index() + c))
    }

    pub fn all_components(&self, base: Attribute) -> bool {
        (0..4).all(|c| self.bit(base.index() + c))
    }

    pub fn is_uniform(&self, base: Attribute) -> bool {
        self.any_component(base) == self.all_components(base)
    }

    pub fn generic_component(&self, index: usize, component: usize) -> bool {
        self.get(Attribute::generic(index, component).expect("Invalid generic attribute"))
    }

    pub fn generic(&self, index: usize) -> bool {
        (0..4).any(|component| self.generic_component(index, component))
    }

    pub fn clip_distances(&self) -> bool {
        self.any_component(Attribute::ClipDistance0) || self.any_component(Attribute::ClipDistance4)
    }

    pub fn legacy(&self) -> bool {
        [
            Attribute::ColorFrontDiffuseR,
            Attribute::ColorFrontSpecularR,
            Attribute::ColorBackDiffuseR,
            Attribute::ColorBackSpecularR,
        ]
        .into_iter()
        .any(|base| self.any_component(base))
            || self.fixed_function_texture()
            || self.get(Attribute::FogCoordinate)
    }

    pub fn fixed_function_texture(&self) -> bool {
        (192..232).any(|index| self.bit(index))
    }
}

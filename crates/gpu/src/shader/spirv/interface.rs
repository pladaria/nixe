//! Scalar user locations keep component-wise linkage independent of vector
//! packing. Builtins retain the aggregate types required by Vulkan.
//! https://docs.vulkan.org/spec/latest/chapters/interfaces.html
use super::*;
use rspirv::dr::Operand;

#[derive(Clone, Copy)]
pub(super) struct Element {
    pub(super) interpolation: Option<ShaderInterpolation>,
    variable: u32,
    scalar: u32,
    scalar_pointer: u32,
    arrayed: bool,
    component: Option<u32>,
}

impl Emitter {
    pub(super) fn emit_interfaces(&mut self, ir: &ShaderIr) -> Result<()> {
        for (input, elements) in [(true, ir.inputs()), (false, ir.outputs())] {
            let mut builtins = BTreeMap::new();
            let mut interpolation = BTreeMap::new();
            let mut output_types = BTreeMap::new();
            let mut user_slots = BTreeSet::new();
            for element in elements {
                let location = element.location;
                if location == ShaderIoLocation::Position
                    && element.scalar_type != ShaderScalarType::Float32
                {
                    return Err(SpirvShaderError::Interface(location));
                }
                let scalar = match element.scalar_type {
                    ShaderScalarType::Float32 => self.float,
                    ShaderScalarType::Unsigned32 => self.uint,
                    ShaderScalarType::Signed32 => self.int,
                    _ => return Err(SpirvShaderError::Interface(location)),
                };
                let storage = if input {
                    spv::StorageClass::Input
                } else {
                    spv::StorageClass::Output
                };
                let arrayed = tessellation::per_vertex(location)
                    && (ir.stage == ShaderStage::TessellationControl
                        || (input && ir.stage == ShaderStage::TessellationEvaluation));
                let (builtin, components, array_components) = match location {
                    ShaderIoLocation::Position if input && ir.stage == ShaderStage::Fragment => {
                        (Some(spv::BuiltIn::FragCoord), 4, false)
                    }
                    ShaderIoLocation::Position if input && ir.stage == ShaderStage::Vertex => {
                        return Err(SpirvShaderError::Interface(location));
                    }
                    ShaderIoLocation::Position if !input && ir.stage == ShaderStage::Fragment => {
                        return Err(SpirvShaderError::Interface(location));
                    }
                    ShaderIoLocation::Position => (Some(spv::BuiltIn::Position), 4, false),
                    ShaderIoLocation::VertexId | ShaderIoLocation::InstanceId
                        if input && ir.stage == ShaderStage::Vertex && scalar != self.float =>
                    {
                        (
                            Some(if location == ShaderIoLocation::VertexId {
                                spv::BuiltIn::VertexIndex
                            } else {
                                spv::BuiltIn::InstanceIndex
                            }),
                            1,
                            false,
                        )
                    }
                    ShaderIoLocation::FragmentDepth
                        if !input && ir.stage == ShaderStage::Fragment && scalar == self.float =>
                    {
                        (Some(spv::BuiltIn::FragDepth), 1, false)
                    }
                    ShaderIoLocation::SampleMask
                        if ir.stage == ShaderStage::Fragment && scalar != self.float =>
                    {
                        (Some(spv::BuiltIn::SampleMask), 1, true)
                    }
                    ShaderIoLocation::InvocationId => (Some(spv::BuiltIn::InvocationId), 1, false),
                    ShaderIoLocation::PatchVertices => {
                        (Some(spv::BuiltIn::PatchVertices), 1, false)
                    }
                    ShaderIoLocation::PrimitiveId if ir.stage != ShaderStage::Fragment => {
                        (Some(spv::BuiltIn::PrimitiveId), 1, false)
                    }
                    ShaderIoLocation::TessCoord => (Some(spv::BuiltIn::TessCoord), 3, false),
                    ShaderIoLocation::TessLevelOuter => {
                        (Some(spv::BuiltIn::TessLevelOuter), 4, true)
                    }
                    ShaderIoLocation::TessLevelInner => {
                        (Some(spv::BuiltIn::TessLevelInner), 2, true)
                    }
                    ShaderIoLocation::Generic(_) | ShaderIoLocation::Patch(_) => (None, 1, false),
                    ShaderIoLocation::Color(_) if !input && ir.stage == ShaderStage::Fragment => {
                        (None, 1, false)
                    }
                    _ => return Err(SpirvShaderError::Interface(location)),
                };
                // Vulkan requires one interpolation policy per fragment input
                // location even if its components are separate scalar variables.
                if input
                    && ir.stage == ShaderStage::Fragment
                    && builtin.is_none()
                    && (interpolation
                        .insert(location, element.interpolation)
                        .is_some_and(|previous| previous != element.interpolation)
                        || (scalar != self.float
                            && element.interpolation != Some(ShaderInterpolation::Constant)))
                {
                    return Err(SpirvShaderError::Interface(location));
                }
                let variable = if let Some(&(variable, builtin_scalar)) = builtins.get(&location) {
                    if builtin_scalar != scalar {
                        return Err(SpirvShaderError::Interface(location));
                    }
                    variable
                } else {
                    let mut ty = if array_components {
                        let count = self.constant(components);
                        self.b.type_array(scalar, count)
                    } else if components > 1 {
                        self.b.type_vector(scalar, components)
                    } else {
                        scalar
                    };
                    if arrayed {
                        let count = if input {
                            self.options.input_control_points
                        } else {
                            ir.tessellation_control_points.unwrap()
                        };
                        let count = self.constant(count);
                        ty = self.b.type_array(ty, count);
                    }
                    let pointer = self.b.type_pointer(None, storage, ty);
                    let variable = self.b.variable(pointer, None, storage, None);
                    self.variables.push(variable);
                    if let Some(builtin) = builtin {
                        self.b.decorate(
                            variable,
                            spv::Decoration::BuiltIn,
                            [Operand::BuiltIn(builtin)],
                        );
                        builtins.insert(location, (variable, scalar));
                    } else {
                        let (ShaderIoLocation::Generic(index)
                        | ShaderIoLocation::Patch(index)
                        | ShaderIoLocation::Color(index)) = location
                        else {
                            unreachable!()
                        };
                        if !user_slots.insert((
                            matches!(location, ShaderIoLocation::Patch(_)),
                            index,
                            element.component,
                        )) {
                            return Err(SpirvShaderError::Interface(location));
                        }
                        if !input
                            && ir.stage == ShaderStage::Fragment
                            && output_types
                                .insert(index, scalar)
                                .is_some_and(|ty| ty != scalar)
                        {
                            return Err(SpirvShaderError::Interface(location));
                        }
                        self.b.decorate(
                            variable,
                            spv::Decoration::Location,
                            [Operand::LiteralBit32(u32::from(index))],
                        );
                        self.b.decorate(
                            variable,
                            spv::Decoration::Component,
                            [Operand::LiteralBit32(u32::from(element.component))],
                        );
                    }
                    if matches!(
                        location,
                        ShaderIoLocation::Patch(_)
                            | ShaderIoLocation::TessLevelOuter
                            | ShaderIoLocation::TessLevelInner
                    ) {
                        self.b.decorate(variable, spv::Decoration::Patch, []);
                    }
                    match element.interpolation {
                        None | Some(ShaderInterpolation::Perspective) => {}
                        Some(ShaderInterpolation::Constant) => {
                            self.b.decorate(variable, spv::Decoration::Flat, [])
                        }
                        Some(ShaderInterpolation::ScreenLinear) => {
                            self.b
                                .decorate(variable, spv::Decoration::NoPerspective, [])
                        }
                    }
                    variable
                };
                self.interfaces.insert(
                    (input, location, element.component),
                    Element {
                        interpolation: element.interpolation,
                        variable,
                        scalar,
                        scalar_pointer: self.b.type_pointer(None, storage, scalar),
                        arrayed,
                        component: (components > 1 || array_components)
                            .then_some(u32::from(element.component)),
                    },
                );
            }
        }
        Ok(())
    }

    fn interface_pointer(
        &mut self,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
    ) -> Result<(u32, u32)> {
        let element = *self
            .interfaces
            .get(&(input, location, component))
            .ok_or(SpirvShaderError::Interface(location))?;
        if element.arrayed != vertex.is_some() {
            return Err(SpirvShaderError::Interface(location));
        }
        let mut indexes = [0; 2];
        let mut count = 0;
        if let Some(vertex) = vertex {
            indexes[count] = vertex;
            count += 1;
        }
        if let Some(component) = element.component {
            indexes[count] = self.constant(component);
            count += 1;
        }
        let pointer = if count == 0 {
            element.variable
        } else {
            self.b.access_chain(
                element.scalar_pointer,
                None,
                element.variable,
                indexes[..count].iter().copied(),
            )?
        };
        Ok((pointer, element.scalar))
    }

    pub(super) fn load_interface(
        &mut self,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
    ) -> Result<u32> {
        let (pointer, scalar) = self.interface_pointer(input, location, component, vertex)?;
        let value = self.b.load(scalar, None, pointer, None, [])?;
        Ok(if scalar == self.uint {
            value
        } else {
            self.b.bitcast(self.uint, None, value)?
        })
    }

    pub(super) fn store_interface(
        &mut self,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
        bits: u32,
    ) -> Result<()> {
        let (pointer, scalar) = self.interface_pointer(false, location, component, vertex)?;
        let value = if scalar == self.uint {
            bits
        } else {
            self.b.bitcast(scalar, None, bits)?
        };
        self.b.store(pointer, value, None, [])?;
        Ok(())
    }
}

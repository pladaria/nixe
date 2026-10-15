//! Scalar user locations keep component-wise linkage independent of vector
//! packing. Builtins retain the aggregate types required by Vulkan.
//! https://docs.vulkan.org/spec/latest/chapters/interfaces.html
use super::*;
use rspirv::dr::Operand;

#[derive(Clone, Copy)]
pub(crate) struct Element {
    pub(crate) interpolation: Option<ShaderInterpolation>,
    variable: u32,
    scalar: u32,
    scalar_pointer: u32,
    arrayed: bool,
    component: Option<u32>,
}

#[derive(Default)]
pub(crate) struct InterfaceLayout {
    pub(crate) elements: BTreeMap<(bool, ShaderIoLocation, u8), Element>,
    pub(crate) variables: Vec<u32>,
}

impl InterfaceLayout {
    pub(crate) fn emit_interfaces(
        &mut self,
        math: &mut Math,
        stage: ShaderStage,
        inputs: &[ShaderInterfaceElement],
        outputs: &[ShaderInterfaceElement],
        control_points: Option<u32>,
    ) -> Result<()> {
        for (input, elements) in [(true, inputs), (false, outputs)] {
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
                let scalar = if location == ShaderIoLocation::FrontFacing
                    && input
                    && stage == ShaderStage::Fragment
                    && element.scalar_type == ShaderScalarType::Unsigned32
                {
                    math.boolean
                } else {
                    match element.scalar_type {
                        ShaderScalarType::Float32 => math.float,
                        ShaderScalarType::Unsigned32 => math.uint,
                        ShaderScalarType::Signed32 => math.int,
                        _ => return Err(SpirvShaderError::Interface(location)),
                    }
                };
                let storage = if input {
                    spv::StorageClass::Input
                } else {
                    spv::StorageClass::Output
                };
                let arrayed = tessellation::per_vertex(location)
                    && (stage == ShaderStage::TessellationControl
                        || (input && stage == ShaderStage::TessellationEvaluation));
                let (builtin, components, array_components) = match location {
                    ShaderIoLocation::Position if input && stage == ShaderStage::Fragment => {
                        (Some(spv::BuiltIn::FragCoord), 4, false)
                    }
                    ShaderIoLocation::Position if input && stage == ShaderStage::Vertex => {
                        return Err(SpirvShaderError::Interface(location));
                    }
                    ShaderIoLocation::Position if !input && stage == ShaderStage::Fragment => {
                        return Err(SpirvShaderError::Interface(location));
                    }
                    ShaderIoLocation::Position => (Some(spv::BuiltIn::Position), 4, false),
                    // PointSize is a float scalar output of the vertex stage.
                    // https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-builtin-variables-pointsize
                    ShaderIoLocation::PointSize
                        if !input && stage == ShaderStage::Vertex && scalar == math.float =>
                    {
                        (Some(spv::BuiltIn::PointSize), 1, false)
                    }
                    ShaderIoLocation::VertexId | ShaderIoLocation::InstanceId
                        if input && stage == ShaderStage::Vertex && scalar != math.float =>
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
                        if !input && stage == ShaderStage::Fragment && scalar == math.float =>
                    {
                        (Some(spv::BuiltIn::FragDepth), 1, false)
                    }
                    ShaderIoLocation::SampleMask
                        if stage == ShaderStage::Fragment && scalar != math.float =>
                    {
                        (Some(spv::BuiltIn::SampleMask), 1, true)
                    }
                    ShaderIoLocation::FrontFacing
                        if input && stage == ShaderStage::Fragment && scalar == math.boolean =>
                    {
                        (Some(spv::BuiltIn::FrontFacing), 1, false)
                    }
                    ShaderIoLocation::InvocationId => (Some(spv::BuiltIn::InvocationId), 1, false),
                    ShaderIoLocation::PatchVertices => {
                        (Some(spv::BuiltIn::PatchVertices), 1, false)
                    }
                    ShaderIoLocation::PrimitiveId if stage != ShaderStage::Fragment => {
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
                    ShaderIoLocation::Color(_) if !input && stage == ShaderStage::Fragment => {
                        (None, 1, false)
                    }
                    _ => return Err(SpirvShaderError::Interface(location)),
                };
                // Vulkan requires one interpolation policy per fragment input
                // location even if its components are separate scalar variables.
                if input
                    && stage == ShaderStage::Fragment
                    && builtin.is_none()
                    && (interpolation
                        .insert(location, element.interpolation)
                        .is_some_and(|previous| previous != element.interpolation)
                        || (scalar != math.float
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
                        let count = math.constant(components);
                        math.b.type_array(scalar, count)
                    } else if components > 1 {
                        math.b.type_vector(scalar, components)
                    } else {
                        scalar
                    };
                    if arrayed {
                        let count = if input {
                            math.options.input_control_points
                        } else {
                            control_points.ok_or(SpirvShaderError::Options(
                                "control stage has no output cardinality",
                            ))?
                        };
                        if count == 0 {
                            return Err(SpirvShaderError::Options(
                                "arrayed patch interface requires a nonzero cardinality",
                            ));
                        }
                        let count = math.constant(count);
                        ty = math.b.type_array(ty, count);
                    }
                    let pointer = math.b.type_pointer(None, storage, ty);
                    let variable = math.b.variable(pointer, None, storage, None);
                    self.variables.push(variable);
                    if let Some(builtin) = builtin {
                        math.b.decorate(
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
                            && stage == ShaderStage::Fragment
                            && output_types
                                .insert(index, scalar)
                                .is_some_and(|ty| ty != scalar)
                        {
                            return Err(SpirvShaderError::Interface(location));
                        }
                        math.b.decorate(
                            variable,
                            spv::Decoration::Location,
                            [Operand::LiteralBit32(u32::from(index))],
                        );
                        math.b.decorate(
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
                        math.b.decorate(variable, spv::Decoration::Patch, []);
                    }
                    match element.interpolation {
                        None | Some(ShaderInterpolation::Perspective) => {}
                        Some(ShaderInterpolation::Constant) => {
                            math.b.decorate(variable, spv::Decoration::Flat, [])
                        }
                        Some(ShaderInterpolation::ScreenLinear) => {
                            math.b
                                .decorate(variable, spv::Decoration::NoPerspective, [])
                        }
                    }
                    variable
                };
                self.elements.insert(
                    (input, location, element.component),
                    Element {
                        interpolation: element.interpolation,
                        variable,
                        scalar,
                        scalar_pointer: math.b.type_pointer(None, storage, scalar),
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
        math: &mut Math,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
    ) -> Result<(u32, u32)> {
        let element = *self
            .elements
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
            indexes[count] = math.constant(component);
            count += 1;
        }
        let pointer = if count == 0 {
            element.variable
        } else {
            math.b.access_chain(
                element.scalar_pointer,
                None,
                element.variable,
                indexes[..count].iter().copied(),
            )?
        };
        Ok((pointer, element.scalar))
    }

    pub(crate) fn load_interface(
        &mut self,
        math: &mut Math,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
    ) -> Result<u32> {
        let (pointer, scalar) = self.interface_pointer(math, input, location, component, vertex)?;
        let value = math.b.load(scalar, None, pointer, None, [])?;
        Ok(if location == ShaderIoLocation::FrontFacing {
            let yes = math.constant(u32::MAX);
            let no = math.constant(0);
            math.b.select(math.uint, None, value, yes, no)?
        } else if scalar == math.uint {
            value
        } else {
            math.b.bitcast(math.uint, None, value)?
        })
    }

    pub(crate) fn store_interface(
        &mut self,
        math: &mut Math,
        location: ShaderIoLocation,
        component: u8,
        vertex: Option<u32>,
        bits: u32,
    ) -> Result<()> {
        let (pointer, scalar) = self.interface_pointer(math, false, location, component, vertex)?;
        let value = if scalar == math.uint {
            bits
        } else {
            math.b.bitcast(scalar, None, bits)?
        };
        math.b.store(pointer, value, None, [])?;
        Ok(())
    }
    pub(crate) fn final_position(&mut self, math: &mut Math) -> Result<()> {
        if let Some(transform) = math.options.vertex_position_transform {
            for component in 0..2 {
                let value =
                    self.load_interface(math, false, ShaderIoLocation::Position, component, None)?;
                let value = math.b.bitcast(math.float, None, value)?;
                let scale = math.constant(transform.scale()[component as usize].to_bits());
                let offset = math.constant(transform.offset()[component as usize].to_bits());
                let scale = math.b.bitcast(math.float, None, scale)?;
                let offset = math.b.bitcast(math.float, None, offset)?;
                let value = math.b.f_mul(math.float, None, value, scale)?;
                math.b.decorate(value, spv::Decoration::NoContraction, []);
                let value = math.b.f_add(math.float, None, value, offset)?;
                math.b.decorate(value, spv::Decoration::NoContraction, []);
                let value = math.b.bitcast(math.uint, None, value)?;
                self.store_interface(math, ShaderIoLocation::Position, component, None, value)?;
            }
        }
        if math.options.depth_clip_negative_one_to_one {
            let z = self.load_interface(math, false, ShaderIoLocation::Position, 2, None)?;
            let w = self.load_interface(math, false, ShaderIoLocation::Position, 3, None)?;
            let z = math.b.bitcast(math.float, None, z)?;
            let w = math.b.bitcast(math.float, None, w)?;
            let half = math.constant(0.5_f32.to_bits());
            let half = math.b.bitcast(math.float, None, half)?;
            let z = math.b.f_mul(math.float, None, z, half)?;
            let w = math.b.f_mul(math.float, None, w, half)?;
            math.b.decorate(z, spv::Decoration::NoContraction, []);
            math.b.decorate(w, spv::Decoration::NoContraction, []);
            let z = math.b.f_add(math.float, None, z, w)?;
            math.b.decorate(z, spv::Decoration::NoContraction, []);
            let z = math.b.bitcast(math.uint, None, z)?;
            self.store_interface(math, ShaderIoLocation::Position, 2, None, z)?;
        }
        Ok(())
    }
}

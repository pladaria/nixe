//! Compute variants preserve runtime data and specialize only proven uniform
//! control edges. A bounded cache uses edge choices, never raw dispatch values.
use super::*;

impl MaxwellNativeFrontend {
    pub(super) fn specialize_compute(
        &mut self,
        mut translated: Arc<crate::MaxwellShaderCode>,
        launch: &super::super::super::compute::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
    ) -> Result<Arc<crate::MaxwellShaderCode>, Error> {
        if let Some(control) = &translated.uniform_control {
            let selection = control
                .select(|resource, offset| {
                    launch.native_uniform_word(
                        &translated.programs[0],
                        resource,
                        offset,
                        space,
                        writes,
                    )
                })
                .map_err(Error::ComputeLaunch)?;
            if !selection.edges().is_empty() {
                let key = (translated.native.identity, selection);
                translated = if let Some(variant) = self.compute_variants.get(&key) {
                    Arc::clone(variant)
                } else {
                    let mut program = translated.programs[0].clone();
                    key.1
                        .apply(Arc::make_mut(&mut program.function))
                        .map_err(compilation)?;
                    program.retain_live_bindings();
                    let mut edges = Vec::new();
                    for &(block, target) in key.1.edges() {
                        edges.extend_from_slice(&(block.index() as u32).to_le_bytes());
                        edges.extend_from_slice(&(target.index() as u32).to_le_bytes());
                    }
                    let identity = nixe_gpu::ssa::artifact::ContentId::of(
                        b"nixe-maxwell-uniform-control-v1",
                        &[&key.0.0, &edges],
                    );
                    let bytes = program.function.values.len()
                        * std::mem::size_of::<nixe_gpu::ssa::Value>()
                        + program
                            .function
                            .blocks
                            .iter()
                            .map(|b| {
                                std::mem::size_of_val(b)
                                    + b.instructions
                                        .iter()
                                        .map(|i| {
                                            std::mem::size_of_val(i)
                                                + i.arguments.len() * 4
                                                + i.results.len() * 4
                                        })
                                        .sum::<usize>()
                                    + b.phis
                                        .iter()
                                        .map(|p| std::mem::size_of_val(p) + p.incoming.len() * 8)
                                        .sum::<usize>()
                            })
                            .sum::<usize>();
                    let variant = Arc::new(crate::MaxwellShaderCode {
                        translation_ns: 0,
                        native: Arc::new(GuestShaderCode {
                            identity,
                            functions: vec![Arc::clone(&program.function)].into_boxed_slice(),
                        }),
                        programs: vec![program].into_boxed_slice(),
                        uniform_control: None,
                    });
                    self.compute_variants
                        .insert(key, Arc::clone(&variant), bytes);
                    variant
                };
            }
        }
        Ok(translated)
    }
}

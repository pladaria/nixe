//! Native image and memory handles retain canonical backing across aliases.
use super::*;
pub(super) fn dispatch(
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    match request.command_id {
        114 => {
            // @SPECULATIVE: SDK requests bind (id, process address, size,
            // flags, offset). Flags 0/2 use the same coherent process backing;
            // their cache-policy distinction is undocumented. Offset must be zero.
            // Captured physical ranges retain storage independently of CPU VA.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            if request.data.len() != 36
                || has_ipc_descriptors(hipc)
                || !matches!(request_u64(request.data, 20), Some(0 | 2))
                || request_u64(request.data, 28) != Some(0)
            {
                return Err(IpcWireError::GraphicsService(
                    format!(
                        "unsupported native process memory bind: {:02x?}",
                        request.data
                    )
                    .into(),
                ));
            }
            let id = request_u32(request.data, 0).unwrap();
            let address =
                nixe_memory::GuestVirtualAddress::new(request_u64(request.data, 4).unwrap());
            let size = request_u64(request.data, 12).unwrap();
            let mut state = session.state.lock().unwrap();
            let process = state
                .process
                .as_ref()
                .ok_or(IpcWireError::Malformed("GSV process is not registered"))?;
            let ranges = process
                .memory()
                .space()
                .capture(address, size, nixe_memory::MemoryPermissions::READ_WRITE)
                .and_then(nixe_memory::CapturedMemory::new)
                .map_err(|e| IpcWireError::GraphicsService(e.to_string().into()))?;
            let memory = state
                .native_images
                .memories
                .get_mut(&id)
                .ok_or(IpcWireError::Malformed("unknown native memory object"))?;
            if memory.backing.is_some() || size != memory.size {
                return Err(IpcWireError::Malformed(
                    "native process memory size/lifecycle mismatch",
                ));
            }
            memory.backing = Some(std::sync::Arc::new(crate::gsv::image::Backing {
                ranges,
                gpu: None,
            }));
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        423 => {
            // @SPECULATIVE: observed SDK bind layout: memory, CPU address,
            // bytes, parent GPU memory, parent offset, then two zero options.
            // The reference does not interpret the final option words. Reject
            // nonzero values until their effects are understood.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            if request.data.len() != 48
                || has_ipc_descriptors(hipc)
                || request.data[32..].iter().any(|&b| b != 0)
            {
                return Err(IpcWireError::Malformed(
                    "unsupported native GPU memory bind layout/options",
                ));
            }
            let id = request_u32(request.data, 0).unwrap();
            let address = request_u64(request.data, 4).unwrap();
            let size = request_u64(request.data, 12).unwrap();
            let parent = request_u32(request.data, 20).unwrap();
            let offset = request_u64(request.data, 24).unwrap();
            let mut state = session.state.lock().unwrap();
            let source = state
                .gpu
                .as_ref()
                .and_then(|gpu| gpu.memories.get(&parent))
                .cloned()
                .ok_or(IpcWireError::Malformed("unknown native memory parent"))?;
            let memory = state
                .native_images
                .memories
                .get_mut(&id)
                .ok_or(IpcWireError::Malformed("unknown native memory object"))?;
            if memory.backing.is_some()
                || size != memory.size
                || source.cpu_address().get().checked_add(offset) != Some(address)
            {
                return Err(IpcWireError::Malformed(
                    "native memory backing range disagrees with its owner",
                ));
            }
            let ranges = source
                .ranges()
                .capture(offset, size)
                .and_then(nixe_memory::CapturedMemory::new)
                .map_err(|e| IpcWireError::GraphicsService(e.to_string().into()))?;
            memory.backing = Some(std::sync::Arc::new(crate::gsv::image::Backing {
                ranges,
                gpu: Some(crate::gsv::image::GpuBacking { source, offset }),
            }));
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        429 => {
            if request.data.len() != 72 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "invalid native image pool creation",
                ));
            }
            let preference = std::array::from_fn(|i| request_u32(request.data, i * 4).unwrap());
            let count = request_u32(request.data, 64).unwrap();
            let memory_id = request_u32(request.data, 68).unwrap();
            let (bytes, _) = crate::gsv::image::requirements(preference, count)
                .map_err(IpcWireError::GraphicsService)?;
            let mut state = session.state.lock().unwrap();
            let context = &mut state.native_images;
            let backing = context
                .memories
                .get(&memory_id)
                .and_then(|m| m.backing.clone())
                .ok_or(IpcWireError::Malformed(
                    "native image pool requires bound memory",
                ))?;
            if bytes > backing.ranges.bytes() {
                return Err(IpcWireError::Malformed(
                    "native images exceed their memory backing",
                ));
            }
            // @SPECULATIVE: native SDK pool handles reserve consecutive IDs;
            // the client derives image i as first + i, as in the reference.
            let first = context
                .next_image
                .checked_add(1)
                .ok_or(IpcWireError::HostResourceExhausted("native image IDs"))?;
            context.next_image = context
                .next_image
                .checked_add(count)
                .ok_or(IpcWireError::HostResourceExhausted("native image IDs"))?;
            context.pools.insert(
                first,
                std::sync::Arc::new(crate::gsv::image::Pool {
                    preference,
                    count,
                    backing,
                    images: (0..count).map(|_| Default::default()).collect(),
                }),
            );
            Ok(encode_tipc_response(0, &first.to_le_bytes(), &[], &[])?)
        }
        430 => {
            if request.data.len() != 4 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "invalid native image pool destruction",
                ));
            }
            if session
                .state
                .lock()
                .unwrap()
                .native_images
                .pools
                .remove(&request_u32(request.data, 0).unwrap())
                .is_none()
            {
                return Err(IpcWireError::Malformed("unknown native image pool"));
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        314 | 315 => {
            // @SPECULATIVE: window aliases retain an existing native image
            // pool; registration does not present images or release fences.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            if request.data.len() != if request.command_id == 315 { 8 } else { 4 }
                || has_ipc_descriptors(hipc)
            {
                return Err(IpcWireError::Malformed(
                    "invalid window image alias request",
                ));
            }
            let id = request_u32(request.data, 0).unwrap();
            let mut state = session.state.lock().unwrap();
            let context = &mut state.native_images;
            if request.command_id == 315 {
                let pool = context
                    .pools
                    .get(&id)
                    .ok_or(IpcWireError::Malformed("unknown native image pool"))?;
                let count = request_u32(request.data, 4).unwrap();
                let (bytes, _) = crate::gsv::image::requirements(pool.preference, count)
                    .map_err(IpcWireError::GraphicsService)?;
                if count != pool.count
                    || bytes > pool.backing.ranges.bytes()
                    || context.window_pools.contains_key(&id)
                {
                    return Err(IpcWireError::Malformed(
                        "invalid window image pool count/backing/lifecycle",
                    ));
                }
                context.window_pools.insert(id, pool.clone());
            } else if context.window_pools.remove(&id).is_none() {
                return Err(IpcWireError::Malformed("unknown window image pool"));
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        111 => {
            // @SPECULATIVE: mode 3 creates an initially unbound native image
            // memory object. The reference observes this SDK contract but the
            // complete mode enumeration is not published.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            if request.data.len() != 12 || has_ipc_descriptors(hipc) || request.data[0] != 3 {
                return Err(IpcWireError::Malformed(
                    "unsupported native memory creation mode",
                ));
            }
            let size = request_u64(request.data, 4).unwrap();
            let mut state = session.state.lock().unwrap();
            if size == 0 || state.native_memory_features.is_none() {
                return Err(IpcWireError::Malformed(
                    "invalid native memory lifecycle/size",
                ));
            }
            let context = &mut state.native_images;
            context.next_memory = context
                .next_memory
                .checked_add(1)
                .ok_or(IpcWireError::HostResourceExhausted("native memory IDs"))?;
            let id = context.next_memory;
            context.memories.insert(
                id,
                crate::gsv::image::Memory {
                    size,
                    backing: None,
                },
            );
            Ok(encode_tipc_response(0, &id.to_le_bytes(), &[], &[])?)
        }
        112 => {
            if request.data.len() != 4 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("invalid native memory destruction"));
            }
            if session
                .state
                .lock()
                .unwrap()
                .native_images
                .memories
                .remove(&request_u32(request.data, 0).unwrap())
                .is_none()
            {
                return Err(IpcWireError::Malformed("unknown native memory object"));
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        434 => {
            if request.data.len() != 4 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("invalid GPU address-space query"));
            }
            let id = request_u32(request.data, 0).unwrap();
            let mut state = session.state.lock().unwrap();
            let space = state
                .gpu
                .as_ref()
                .and_then(|gpu| gpu.spaces.get(&id))
                .cloned()
                .ok_or(IpcWireError::Malformed("unknown GPU address space"))?;
            state.native_images.spaces.insert(id, space);
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        435 | 1510 => {
            if request.data.len() != 68 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "invalid native image requirements request",
                ));
            }
            let initialized = {
                let state = session.state.lock().unwrap();
                if request.command_id == 1510 {
                    state
                        .inter_module_images
                        .as_ref()
                        .is_some_and(|c| c.initialized)
                } else {
                    state.gpu.as_ref().is_some_and(|c| c.initialized)
                }
            };
            if !initialized {
                return Err(IpcWireError::Malformed(
                    "native image context is not initialized",
                ));
            }
            let mut preference = [0; 16];
            for (index, word) in preference.iter_mut().enumerate() {
                *word = request_u32(request.data, index * 4).unwrap();
            }
            let (size, alignment) =
                crate::gsv::image::requirements(preference, request_u32(request.data, 64).unwrap())
                    .map_err(IpcWireError::GraphicsService)?;
            let mut result = size.to_le_bytes().to_vec();
            if request.command_id == 1510 {
                result.extend_from_slice(&alignment.to_le_bytes());
            }
            Ok(encode_tipc_response(0, &result, &[], &[])?)
        }
        _ => unreachable!("routed native image command"),
    }
}

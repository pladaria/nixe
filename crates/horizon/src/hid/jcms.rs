//! Joy-Con 2 mouse-sensor transfer-memory lifetime, separate from Npad FullKey.
use super::*;
use nixe_runtime::TransferMemoryObject;

impl HidSystem {
    pub(crate) async fn initialize_jcms(
        &self,
        memory: Arc<TransferMemoryObject>,
    ) -> Result<(), Arc<MemoryError>> {
        // @SPECULATIVE: with no Joy-Con 2 mouse sensors exposed, an empty
        // transfer buffer represents no samples. The precise Jcms LIFO layout
        // is not yet documented; do not fabricate connected devices or samples.
        // The reference also accepts initialization without producing Jcms data.
        // https://switchbrew.org/wiki/HID_services#InitializeJcms
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/hid/hid_server.cpp
        let machine = memory.ranges()[0].allocation().machine();
        let mut offset = 0;
        while offset < memory.size() {
            let bytes = vec![0; (memory.size() - offset).min(4096) as usize];
            let count = bytes.len() as u64;
            if let Access::Pending(pending) = machine
                .write_captured_owned(memory.ranges(), offset, bytes)
                .map_err(Arc::new)?
            {
                pending.retain_owner(memory.clone());
                match pending.await? {
                    AccessResult::Written(_) => (),
                    _ => {
                        return Err(Arc::new(MemoryError::Execution(
                            "invalid Jcms initialization write completion".into(),
                        )));
                    }
                }
            }
            offset += count;
        }
        log::debug!(
            "hid initialized Jcms transfer memory: bytes={:#x}, no mouse sensors connected",
            memory.size()
        );
        *self.state.jcms.lock().unwrap() = Some(memory);
        Ok(())
    }

    pub(crate) fn finalize_jcms(&self) {
        self.state.jcms.lock().unwrap().take();
    }
}

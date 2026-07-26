use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator};

use super::diagnostics::NvDrvCallError;
use super::ioctl::{NvDrvIoctlRequest, NvDrvIoctlResponse};
use super::{NvDrvFileDescriptor, NvDrvSession, UnsupportedNvDrvOperation};

/// Semantic NVIDIA service failure before Horizon response encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NvDrvServiceError {
    DriverResult(u32),
    Unsupported(UnsupportedNvDrvOperation),
}

impl From<NvDrvCallError> for NvDrvServiceError {
    fn from(error: NvDrvCallError) -> Self {
        match error {
            NvDrvCallError::GuestResult(result) => Self::DriverResult(result),
            NvDrvCallError::Unsupported(operation) => Self::Unsupported(operation),
        }
    }
}

/// Semantic service boundary used by Horizon's HIPC/CMIF wire adapter.
///
/// This type accepts decoded values and never reads guest descriptors or
/// encodes a Horizon response.
pub(crate) struct NvDrvService<'a> {
    session: &'a NvDrvSession,
}

impl<'a> NvDrvService<'a> {
    pub(crate) const fn new(session: &'a NvDrvSession) -> Self {
        Self { session }
    }

    pub(crate) fn open(
        &self,
        path: &[u8],
        process_id: u64,
    ) -> Result<NvDrvFileDescriptor, NvDrvServiceError> {
        self.session.open(path, process_id).map_err(Into::into)
    }

    pub(crate) fn ioctl(
        &self,
        fd: NvDrvFileDescriptor,
        request: u32,
        input: &[u8],
        process_id: u64,
        address_space: AddressSpaceId,
        translator: &dyn CanonicalRangeTranslator,
    ) -> Result<NvDrvIoctlResponse, UnsupportedNvDrvOperation> {
        let request = NvDrvIoctlRequest {
            fd,
            request,
            input,
            process_id,
            address_space,
            translator,
        };
        let (output, driver_result) = self.session.ioctl_with_memory(
            request.fd,
            request.request,
            request.input,
            request.process_id,
            request.address_space,
            request.translator,
        )?;
        Ok(NvDrvIoctlResponse {
            output,
            driver_result,
        })
    }

    pub(crate) fn close(&self, fd: NvDrvFileDescriptor) -> u32 {
        self.session.close(fd)
    }

    pub(crate) fn initialize(&self) {
        self.session.initialize();
    }

    pub(crate) fn set_aruid(&self, process_id: u64, applet_resource_user_id: u64) {
        self.session.set_aruid(process_id, applet_resource_user_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::{NV_NOT_INITIALIZED, NvDrvDeviceKind};

    #[test]
    fn semantic_service_does_not_encode_horizon_wire_results() {
        let session = NvDrvSession::new();
        let service = NvDrvService::new(&session);

        assert_eq!(
            service.open(NvDrvDeviceKind::NvMap.path().as_bytes(), 7),
            Err(NvDrvServiceError::DriverResult(NV_NOT_INITIALIZED))
        );
    }
}

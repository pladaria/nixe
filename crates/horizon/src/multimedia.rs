//! Guest multimedia performance requests, independent of host CPU/GPU clocks.
//!
//! Nixe's engines execute on host workers rather than Tegra clock domains. These
//! requests retain the guest's lower/upper performance policy; they do not skip
//! engine work or signal completion of a video submission.
//! https://switchbrew.org/wiki/Display_services#mm:u
//! https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/mm/mm_u.cpp

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct Request {
    module: u32,
    minimum: u32,
    maximum: i32,
}

#[derive(Debug, Default)]
struct State {
    next_id: u32,
    requests: BTreeMap<u32, Request>,
}

#[derive(Clone, Debug, Default)]
pub struct MultimediaSession(Arc<Mutex<State>>);

impl MultimediaSession {
    pub(crate) fn initialize(&self, module: u32) -> Result<u32, &'static str> {
        if module > 8 {
            return Err("unknown multimedia clock module");
        }
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state
            .next_id
            .checked_add(1)
            .ok_or("multimedia request identifiers exhausted")?;
        state.next_id = id;
        state.requests.insert(
            id,
            Request {
                module,
                minimum: 0,
                maximum: -1,
            },
        );
        Ok(id)
    }

    pub(crate) fn request(
        &self,
        command: u32,
        key: u32,
        bounds: Option<(u32, i32)>,
    ) -> Result<Option<u32>, &'static str> {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = if command < 4 {
            state
                .requests
                .iter()
                .find(|(_, request)| request.module == key)
                .map(|(&id, _)| id)
        } else {
            Some(key)
        }
        .ok_or("multimedia module is not initialized")?;
        let request = state
            .requests
            .get_mut(&id)
            .ok_or("multimedia request is not initialized")?;
        match command {
            1 | 5 => {
                state.requests.remove(&id);
                Ok(None)
            }
            2 | 6 => {
                let (minimum, maximum) = bounds.expect("set command has bounds");
                if maximum < -1 || (maximum >= 0 && minimum > maximum as u32) {
                    return Err("invalid multimedia performance bounds");
                }
                request.minimum = minimum;
                request.maximum = maximum;
                Ok(None)
            }
            3 | 7 => Ok(Some(request.minimum)),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multimedia_requests_keep_independent_bounds_and_shared_lifetimes() {
        let session = MultimediaSession::default();
        let decoder = session.initialize(5).unwrap();
        let encoder = session.initialize(6).unwrap();
        session
            .request(6, decoder, Some((307_200_000, -1)))
            .unwrap();
        assert_eq!(
            session.clone().request(7, decoder, None),
            Ok(Some(307_200_000))
        );
        assert_eq!(session.request(3, 6, None), Ok(Some(0)));
        assert!(session.request(6, decoder, Some((200, 100))).is_err());
        assert_eq!(session.request(3, 5, None), Ok(Some(307_200_000)));
        session.request(1, 5, None).unwrap();
        assert!(session.request(7, decoder, None).is_err());
        assert_eq!(session.request(7, encoder, None), Ok(Some(0)));
    }
}

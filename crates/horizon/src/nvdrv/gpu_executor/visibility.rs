//! Coalesced external CPU demands remain outside the recording/completion path.
use super::*;
use std::collections::HashMap;

type Reply = mpsc::SyncSender<Result<Box<[u8]>, Box<str>>>;

#[derive(Default)]
pub(super) struct CpuVisibilityQueue {
    requests: HashMap<CpuVisibilityRequest, Vec<Reply>>,
}
impl CpuVisibilityQueue {
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn request<E: GpuExecution>(
        &mut self,
        backend: Option<&mut E>,
        request: CpuVisibilityRequest,
        reply: Reply,
    ) {
        if let Some(replies) = self.requests.get_mut(&request) {
            replies.push(reply);
            return;
        }
        match service(backend, request) {
            Ok(None) => {
                nixe_memory::trace::detail_event(
                    "cpu_visibility_deferred",
                    request.visible_at.get(),
                    request.size as u64,
                    0,
                );
                self.requests.insert(request, vec![reply]);
            }
            result => {
                let _ = reply.send(result.map(Option::unwrap));
            }
        }
    }

    pub fn progress<E: GpuExecution>(&mut self, mut backend: Option<&mut E>) {
        self.requests.retain(
            |&request, replies| match service(backend.as_deref_mut(), request) {
                Ok(None) => true,
                result => {
                    nixe_memory::trace::detail_event(
                        "cpu_visibility_reply",
                        request.visible_at.get(),
                        request.size as u64,
                        replies.len() as u64,
                    );
                    reply_all(std::mem::take(replies), result.map(Option::unwrap));
                    false
                }
            },
        );
    }

    pub fn cancel(&mut self) {
        for (_, replies) in self.requests.drain() {
            reply_all(replies, Err("GPU backend owner is shutting down".into()));
        }
    }
}

fn service<E: GpuExecution>(
    backend: Option<&mut E>,
    request: CpuVisibilityRequest,
) -> Result<Option<Box<[u8]>>, Box<str>> {
    backend
        .ok_or_else(|| Box::<str>::from("GPU backend is unavailable"))?
        .service_cpu_visibility(request)
        .map_err(|error| error.to_string().into_boxed_str())
}

fn reply_all(mut replies: Vec<Reply>, result: Result<Box<[u8]>, Box<str>>) {
    if let Some(last) = replies.pop() {
        for reply in replies {
            let _ = reply.send(result.clone());
        }
        let _ = last.send(result);
    }
}

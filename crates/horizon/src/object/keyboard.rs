//! Asynchronous AM software-keyboard channels and completion events.
use super::*;
use nixe_applets::keyboard::{KeyboardConfig, KeyboardHost, Submission, Validation};
#[derive(Debug)]
pub(crate) enum KeyboardValidationError {
    Invalid(&'static str),
    Memory(Arc<nixe_memory::MemoryError>),
}
impl From<&'static str> for KeyboardValidationError {
    fn from(message: &'static str) -> Self {
        Self::Invalid(message)
    }
}

impl AppletSession {
    pub(crate) fn create_transfer_storage(
        &self,
        memory: Arc<nixe_runtime::TransferMemoryObject>,
        size: usize,
        writable: bool,
    ) -> Result<u32, CreateAppletStorageError> {
        let mut domain = self.domain.lock().unwrap();
        if !domain.converted {
            return Err(CreateAppletStorageError::NotDomain);
        }
        if domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return Err(CreateAppletStorageError::DomainCapacityExhausted);
        }
        let object = domain.next_object_id;
        let next = object
            .checked_add(1)
            .ok_or(CreateAppletStorageError::DomainCapacityExhausted)?;
        let storage = domain.next_storage_id;
        let next_storage = storage
            .checked_add(1)
            .ok_or(CreateAppletStorageError::AllocationFailed)?;
        domain.storages.insert(
            storage,
            AppletStorage::Transfer {
                memory,
                size,
                writable,
            },
        );
        domain.objects.insert(
            object,
            AppletObject::Storage {
                storage_id: storage,
            },
        );
        domain.next_object_id = next;
        domain.next_storage_id = next_storage;
        Ok(object)
    }
    pub(crate) fn start_keyboard(
        &self,
        object: u32,
        config: KeyboardConfig,
        host: &KeyboardHost,
    ) -> Result<(), &'static str> {
        let weak = Arc::downgrade(&self.domain);
        let mut domain = self.domain.lock().unwrap();
        let active = domain
            .active_library_applet
            .as_mut()
            .filter(|a| a.object_id == object)
            .ok_or("keyboard accessor is no longer active")?;
        if active.keyboard.is_some() || active.result.is_some() {
            return Err("software keyboard was already started");
        }
        let keyboard = host.open(config, move |value| {
            let Some(domain) = weak.upgrade() else {
                return;
            };
            let mut domain = domain.lock().unwrap();
            let Some(active) = domain
                .active_library_applet
                .as_mut()
                .filter(|a| a.object_id == object && a.result.is_none())
            else {
                return;
            };
            let keyboard = active.keyboard.as_ref().unwrap();
            match value {
                Submission::Accept { text, confirmed } => {
                    if !keyboard.config.valid(&text) {
                        return;
                    }
                    if confirmed
                        && active.confirmation_text.take().as_deref() != Some(text.as_str())
                    {
                        return;
                    }
                    if keyboard.config.text_check && !confirmed {
                        if active.check_text.is_some() {
                            return;
                        }
                        active
                            .interactive_output
                            .push_back(keyboard.config.encode_check(&text));
                        active.check_text = Some(text);
                        active.interactive_event.signal();
                        return;
                    }
                    active.output = Some(keyboard.config.encode_result(&text, false));
                }
                Submission::Cancel => {
                    if keyboard.config.cancel_disabled {
                        return;
                    }
                    active.output = Some(keyboard.config.encode_result("", true));
                }
            }
            active.result = Some(0);
            active.output_event.signal();
            keyboard.finish();
            active.state_changed_event.signal();
        })?;
        active.keyboard = Some(keyboard);
        domain.library_applet_launchable_event.clear();
        Ok(())
    }
    pub(crate) fn interactive_event(&self, object: u32) -> Option<ReadableEventObject> {
        let domain = self.domain.lock().unwrap();
        let active = domain
            .active_library_applet
            .as_ref()
            .filter(|a| a.object_id == object)?;
        Some(active.interactive_reader.clone())
    }
    pub(crate) fn pop_interactive_output(
        &self,
        object: u32,
    ) -> Result<u32, PopAppletLaunchParameterError> {
        let mut domain = self.domain.lock().unwrap();
        let active = domain
            .active_library_applet
            .as_ref()
            .filter(|a| a.object_id == object)
            .ok_or(PopAppletLaunchParameterError::NotAvailable)?;
        if active.interactive_output.is_empty() {
            return Err(PopAppletLaunchParameterError::NotAvailable);
        }
        if domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return Err(PopAppletLaunchParameterError::DomainCapacityExhausted);
        }
        let id = domain.next_object_id;
        let next = id
            .checked_add(1)
            .ok_or(PopAppletLaunchParameterError::ObjectIdExhausted)?;
        let storage = domain.next_storage_id;
        let next_storage = storage
            .checked_add(1)
            .ok_or(PopAppletLaunchParameterError::StorageIdExhausted)?;
        let active = domain.active_library_applet.as_mut().unwrap();
        let output = active.interactive_output.pop_front().unwrap();
        if active.interactive_output.is_empty() {
            active.interactive_event.clear();
        }
        domain
            .storages
            .insert(storage, AppletStorage::Owned(output));
        domain.objects.insert(
            id,
            AppletObject::Storage {
                storage_id: storage,
            },
        );
        domain.next_object_id = next;
        domain.next_storage_id = next_storage;
        Ok(id)
    }
    pub(crate) async fn keyboard_validation(
        &self,
        object: u32,
        storage_object: u32,
    ) -> Result<(), KeyboardValidationError> {
        let read = {
            let domain = self.domain.lock().unwrap();
            let Some(AppletObject::Storage { storage_id }) = domain.objects.get(&storage_object)
            else {
                return Err("keyboard validation input is not an IStorage".into());
            };
            let source = domain
                .storages
                .get(storage_id)
                .ok_or("keyboard validation storage is missing")?;

            source
                .read(0, source.len())
                .map_err(|_| "cannot read keyboard validation storage")?
        };
        let bytes = read.resolve().await.map_err(|error| match error {
            AppletStorageAccessError::Memory(error) => KeyboardValidationError::Memory(error),
            _ => KeyboardValidationError::Invalid("cannot read keyboard validation storage"),
        })?;
        if bytes.len() != 0x7d8 {
            return Err("invalid keyboard validation response size".into());
        }
        let result = u32::from_le_bytes(bytes[..4].try_into().unwrap());

        let mut domain = self.domain.lock().unwrap();
        let active = domain
            .active_library_applet
            .as_mut()
            .filter(|a| a.object_id == object)
            .ok_or("keyboard validation accessor is no longer active")?;
        let keyboard = active
            .keyboard
            .as_ref()
            .ok_or("no software keyboard is running")?;
        let message = if result == 1 || result == 2 {
            if keyboard.config.utf8 {
                let bytes = &bytes[4..];
                let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
                std::str::from_utf8(&bytes[..end])
                    .map_err(|_| "invalid UTF-8 keyboard validation message")?
                    .to_owned()
            } else {
                nixe_applets::keyboard::utf16(&bytes[4..])?
            }
        } else {
            String::new()
        };
        let text = active
            .check_text
            .take()
            .ok_or("keyboard validation has no pending submission")?;
        match result {
            0 => {
                active.output = Some(keyboard.config.encode_result(&text, false));
                active.result = Some(0);
                active.output_event.signal();
                keyboard.finish();
                active.state_changed_event.signal();
            }
            1 => keyboard.feedback(Validation::Failure(message)),
            2 => {
                active.confirmation_text = Some(text);
                keyboard.feedback(Validation::Confirm(message));
            }
            3 => keyboard.feedback(Validation::Retry),
            _ => return Err("unsupported software keyboard validation result".into()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_memory::run;
    fn config(check: bool, utf8: bool) -> KeyboardConfig {
        let mut common = vec![0; 32];
        common[..4].copy_from_slice(&1u32.to_le_bytes());
        common[4..8].copy_from_slice(&32u32.to_le_bytes());
        common[8..12].copy_from_slice(&0x8000du32.to_le_bytes());
        let mut args = vec![0; 0x4c8];
        args[0x3d0] = u8::from(check);
        args[0x3bd] = u8::from(utf8);
        KeyboardConfig::decode(&common, &args, &[]).unwrap()
    }
    fn response(session: &AppletSession, result: u32, message: &str, utf8: bool) -> u32 {
        let storage = session.create_storage(0x7d8).unwrap();
        let AppletObject::Storage { storage_id } = session.object(storage).unwrap() else {
            unreachable!()
        };
        run(session.write_storage(storage_id, 0, &result.to_le_bytes())).unwrap();
        let bytes: Vec<_> = if utf8 {
            message.as_bytes().to_vec()
        } else {
            message.encode_utf16().flat_map(u16::to_le_bytes).collect()
        };
        run(session.write_storage(storage_id, 4, &bytes)).unwrap();
        storage
    }
    #[test]
    fn asynchronous_validation_rejection_confirmation_and_completion() {
        for utf8 in [false, true] {
            let session = AppletSession::new(OperationMode::Console);
            session.convert_to_domain();
            let applet = session
                .create_library_applet(
                    LibraryAppletId::SoftwareKeyboard,
                    LibraryAppletMode::AllForeground,
                )
                .unwrap();
            let host = KeyboardHost::new(|| {});
            session
                .start_keyboard(applet, config(true, utf8), &host)
                .unwrap();
            let keyboard = host.take_request().unwrap();
            let normal_event = session.library_applet_output_event(applet).unwrap();
            assert!(!normal_event.is_signalled());
            let event = session.interactive_event(applet).unwrap();
            keyboard.submit(Submission::Accept {
                text: "Nixe".into(),
                confirmed: false,
            });
            assert!(event.is_signalled());
            assert!(session.library_applet_result(applet).is_none());
            let output = session.pop_interactive_output(applet).unwrap();
            assert!(!event.is_signalled());
            let AppletObject::Storage { storage_id } = session.object(output).unwrap() else {
                unreachable!()
            };
            let bytes = run(session.read_storage(storage_id, 0, 0x7dc)).unwrap();
            assert_eq!(
                u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                if utf8 { 5 } else { 10 }
            );
            let rejection = response(&session, 1, "Try again", utf8);
            run(session.keyboard_validation(applet, rejection)).unwrap();
            assert!(
                matches!(keyboard.take_feedback(),Some(Validation::Failure(ref text)) if text=="Try again")
            );
            keyboard.submit(Submission::Accept {
                text: "Nixe".into(),
                confirmed: false,
            });
            session.pop_interactive_output(applet).unwrap();
            let confirmation = response(&session, 2, "Use name?", utf8);
            run(session.keyboard_validation(applet, confirmation)).unwrap();
            assert!(matches!(
                keyboard.take_feedback(),
                Some(Validation::Confirm(_))
            ));
            keyboard.submit(Submission::Accept {
                text: "Nixe".into(),
                confirmed: true,
            });
            assert_eq!(session.library_applet_result(applet), Some(0));
            assert!(keyboard.finished());
            assert!(
                session
                    .library_applet_state_changed_event(applet)
                    .unwrap()
                    .is_signalled()
            );
            assert!(normal_event.is_signalled());
            let output = session.pop_library_applet_output(applet).unwrap();
            assert!(!normal_event.is_signalled());
            assert!(
                session
                    .library_applet_state_changed_event(applet)
                    .unwrap()
                    .is_signalled()
            );
            let AppletObject::Storage { storage_id } = session.object(output).unwrap() else {
                unreachable!()
            };
            let bytes = run(session.read_storage(storage_id, 0, 0x7d8)).unwrap();
            assert_eq!(bytes, keyboard.config.encode_result("Nixe", false));
            assert!(session.close_object(applet));
        }
    }
    #[test]
    fn transfer_storage_preserves_aliases_permissions_and_handle_lifetime() {
        let mut initial = [0; 0x2000];
        initial[0xffc..0x1004].copy_from_slice(b"initial!");
        let transfer = crate::test_memory::transfer(&initial);
        let ranges = transfer.ranges().clone();
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let id = session
            .create_transfer_storage(Arc::clone(&transfer), 0x2000, true)
            .unwrap();
        let AppletObject::Storage { storage_id } = session.object(id).unwrap() else {
            unreachable!()
        };
        drop(transfer);
        assert_eq!(
            run(session.read_storage(storage_id, 0xffc, 8)).unwrap(),
            b"initial!"
        );
        run(session.write_storage(storage_id, 0xffe, b"change")).unwrap();
        let mut actual = [0; 8];
        let alias = ranges.capture(0xffc, 8).unwrap();
        crate::test_memory::read(&alias, &mut actual);
        assert_eq!(&actual, b"inchange");
        let reader = session.open_storage_accessor(storage_id).unwrap();
        session.close_object(id);
        assert_eq!(
            run(session.read_storage(storage_id, 0xffc, 8)).unwrap(),
            b"inchange"
        );
        session.close_object(reader);
        assert!(matches!(
            run(session.read_storage(storage_id, 0, 1)),
            Err(AppletStorageAccessError::NotFound)
        ));
        let transfer = crate::test_memory::transfer(&[0; 0x2000]);
        let id = session
            .create_transfer_storage(transfer, 0x1000, false)
            .unwrap();
        let AppletObject::Storage { storage_id } = session.object(id).unwrap() else {
            unreachable!()
        };
        assert!(matches!(
            run(session.write_storage(storage_id, 0, b"x")),
            Err(AppletStorageAccessError::ReadOnly)
        ));
        assert!(matches!(
            run(session.read_storage(storage_id, 0x1000, 1)),
            Err(AppletStorageAccessError::OutOfRange)
        ));
    }
    #[test]
    fn cancellation_and_accessor_close_finish_host_keyboard() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let applet = session
            .create_library_applet(
                LibraryAppletId::SoftwareKeyboard,
                LibraryAppletMode::AllForeground,
            )
            .unwrap();
        let host = KeyboardHost::new(|| {});
        session
            .start_keyboard(applet, config(false, false), &host)
            .unwrap();
        let keyboard = host.take_request().unwrap();
        keyboard.submit(Submission::Cancel);
        let output = session.pop_library_applet_output(applet).unwrap();
        let AppletObject::Storage { storage_id } = session.object(output).unwrap() else {
            unreachable!()
        };
        assert_eq!(
            run(session.read_storage(storage_id, 0, 4)).unwrap(),
            1u32.to_le_bytes()
        );
        session.close_object(applet);
        host.close(&keyboard);
        let applet = session
            .create_library_applet(
                LibraryAppletId::SoftwareKeyboard,
                LibraryAppletMode::AllForeground,
            )
            .unwrap();
        session
            .start_keyboard(applet, config(false, false), &host)
            .unwrap();
        let keyboard = host.take_request().unwrap();
        session.close_object(applet);
        assert!(keyboard.finished());
    }
}

use super::prelude::*;

enum AccountTarget {
    Root,
    BaasManagerForApplication(AccountManagerForApplicationSession),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AccountCommand {
    GetUserExistence,
    InitializeApplicationInfo,
    GetBaasAccountManagerForApplication,
    IsUserAccountSwitchLocked,
}

impl AccountCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            1 => Some(Self::GetUserExistence),
            // Application-info initialization versions share the caller-PID
            // association. V2 also carries only the PID placeholder, with no
            // scalar output; this session stores the authenticated process ID.
            // https://switchbrew.org/wiki/Account_services#acc:u0
            100 | 140 | 160 => Some(Self::InitializeApplicationInfo),
            150 => Some(Self::IsUserAccountSwitchLocked),
            101 => Some(Self::GetBaasAccountManagerForApplication),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_account(
    process: &mut ExceptionProcessContext<'_>,
    session: &AccountSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    user_account_switch_locked: Option<bool>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            let result = if session.close_object(*object_id) {
                HorizonIpcResult::SUCCESS
            } else {
                HorizonIpcResult::CMIF_TARGET_NOT_FOUND
            };
            return account_response(session, request.token, result);
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return account_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            if *object_id == 1 {
                AccountTarget::Root
            } else {
                let Some(object) = session.object(*object_id) else {
                    return account_response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_TARGET_NOT_FOUND,
                    );
                };
                match object {
                    AccountObject::BaasManagerForApplication(manager) => {
                        AccountTarget::BaasManagerForApplication(manager)
                    }
                }
            }
        }
        None if session.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain acc:u0 request omitted its domain header",
            ));
        }
        None => AccountTarget::Root,
    };

    match target {
        AccountTarget::Root => {
            dispatch_account_root(process, session, request, hipc, user_account_switch_locked)
        }
        AccountTarget::BaasManagerForApplication(manager) => {
            dispatch_account_manager_for_application(&manager, request)
        }
    }
}

fn dispatch_account_root(
    process: &mut ExceptionProcessContext<'_>,
    session: &AccountSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    user_account_switch_locked: Option<bool>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = AccountCommand::decode(request.command_id) else {
        return unsupported_service_command("acc:u0", request.command_id);
    };

    match command {
        // A well-formed UID absent from the local account database produces
        // false, rather than an unsupported account-manager operation.
        // https://switchbrew.org/wiki/Account_services#acc:u0
        AccountCommand::GetUserExistence => {
            if !request.has_payload_size(16) || has_ipc_descriptors(hipc) {
                return account_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            let exists = is_configured_user(session, &request.data[..16]);
            semantic_success(
                request.token,
                session.is_domain(),
                &[u8::from(exists)],
                &[],
                &[],
                None,
            )
        }
        // libnx sends the caller PID descriptor and a zero u64 placeholder:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/acc.c#L61-L67
        AccountCommand::InitializeApplicationInfo => {
            if hipc.pid.is_none()
                || request_u64(request.data, 0) != Some(0)
                || !request.has_payload_size(8)
                || has_ipc_descriptors_other_than_pid(hipc)
            {
                return account_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            session.initialize_application_info(process.process_id());
            account_response(session, request.token, HorizonIpcResult::SUCCESS)
        }
        // This is the application's NACP UserAccountSwitchLock bit, not a
        // consequence of how many local users happen to be configured.
        // https://switchbrew.org/wiki/Account_services#acc:u0
        // https://switchbrew.org/wiki/NACP
        AccountCommand::IsUserAccountSwitchLocked => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return account_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            let locked = user_account_switch_locked.ok_or_else(|| {
                IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
                    service: "acc:u0",
                    command_id: request.command_id,
                    detail: "application account-switch policy is unavailable",
                })
            })?;
            semantic_success(
                request.token,
                session.is_domain(),
                &[u8::from(locked)],
                &[],
                &[],
                None,
            )
        }
        // GetBaasAccountManagerForApplication associates the local AccountUid
        // with the Nintendo-account manager returned to this application. The
        // manager's existence does not mean that an online account is linked.
        // https://switchbrew.org/w/index.php?title=Account_services&oldid=14813#acc:u0
        AccountCommand::GetBaasAccountManagerForApplication => {
            if !request.has_payload_size(16) || has_ipc_descriptors(hipc) {
                return account_response(
                    session,
                    request.token,
                    HorizonIpcResult::CMIF_INVALID_IN_HEADER,
                );
            }
            let user = session.user();
            if !is_configured_user(session, &request.data[..16]) {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandVariant {
                        service: "acc:u0",
                        command_id: 101,
                        detail: "requested account UID is not configured",
                    },
                ));
            }

            let manager = AccountManagerForApplicationSession::new(user);
            if session.is_domain() {
                let Some(object_id) =
                    session.insert_object(AccountObject::BaasManagerForApplication(manager))
                else {
                    return account_response(
                        session,
                        request.token,
                        HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES,
                    );
                };
                log::debug!(
                    "acc:u0 opened IManagerForApplication as domain object {object_id:#x} for user {} ({})",
                    user.name(),
                    user.id(),
                );
                Ok((
                    encode_domain_response(
                        request.token,
                        HorizonIpcResult::SUCCESS,
                        &[],
                        &[],
                        &[object_id],
                    )?,
                    None,
                ))
            } else {
                let handle = process
                    .handles_mut()
                    .insert(HorizonIpcObject::AccountManagerForApplication(manager))
                    .map_err(|_| {
                        IpcWireError::HostResourceExhausted(
                            "installing an account application-manager handle",
                        )
                    })?;
                log::debug!(
                    "acc:u0 opened IManagerForApplication handle {handle:#x} for user {} ({})",
                    user.name(),
                    user.id(),
                );
                Ok((
                    encode_response(request.token, HorizonIpcResult::SUCCESS, &[], Some(handle))?,
                    Some(handle),
                ))
            }
        }
    }
}

fn is_configured_user(session: &AccountSession, encoded_user_id: &[u8]) -> bool {
    encoded_user_id == session.user().id().encode()
}

pub(in crate::ipc_wire) fn dispatch_account_manager_for_application(
    _manager: &AccountManagerForApplicationSession,
    request: CmifRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    unsupported_service_command("IManagerForApplication", request.command_id)
}

fn account_response(
    session: &AccountSession,
    token: u32,
    result: HorizonIpcResult,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if session.is_domain() {
        Ok((encode_domain_response(token, result, &[], &[], &[])?, None))
    } else {
        Ok((encode_response(token, result, &[], None)?, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_101_targets_only_the_configured_local_user() {
        let session = AccountSession::new();

        assert_eq!(
            AccountCommand::decode(101),
            Some(AccountCommand::GetBaasAccountManagerForApplication)
        );
        assert!(is_configured_user(&session, &1_u128.to_le_bytes()));
        assert!(!is_configured_user(&session, &0_u128.to_le_bytes()));
        assert!(!is_configured_user(&session, &[1]));
    }
}

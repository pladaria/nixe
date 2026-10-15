use ash::vk;
use std::{
    ffi::{CStr, c_void},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ValidationMode {
    #[default]
    Disabled,
    Enabled,
}

#[derive(Default)]
pub(crate) struct ValidationState {
    pub errors: Mutex<Vec<String>>,
}

/// Retain this across device destruction to verify initialization and teardown.
#[derive(Clone)]
pub struct ValidationMonitor(pub(crate) Arc<ValidationState>);
impl ValidationMonitor {
    pub fn errors(&self) -> Vec<String> {
        self.0
            .errors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    pub fn check(&self) -> Result<(), crate::VulkanError> {
        let errors = self.errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(crate::VulkanError::Validation(errors))
        }
    }
}

pub(crate) unsafe extern "system" fn callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: Vulkan supplies callback data for this call. InstanceOwner retains
    // the Arc allocation used as user data until its messenger and instance die.
    let message = unsafe {
        if data.is_null() || (*data).p_message.is_null() {
            return vk::FALSE;
        }
        CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("Vulkan validation: {message}");
        // No panic may cross this FFI callback. Bound diagnostics; validation is
        // explicitly disabled in the normal renderer path.
        let state = unsafe { &*user.cast::<ValidationState>() };
        let mut errors = state
            .errors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if errors.len() < 64 {
            errors.push(message.into_owned());
        }
    } else {
        log::warn!("Vulkan validation: {message}");
    }
    vk::FALSE
}

pub(crate) fn messenger_info(
    state: &Arc<ValidationState>,
) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(callback))
        .user_data(Arc::as_ptr(state) as *mut c_void)
}

//! Host-independent touch-screen state shared by window and emulation threads.

use std::sync::{Arc, Mutex};

pub const MAX_TOUCH_CONTACTS: usize = 16;
pub const TOUCH_SCREEN_WIDTH: u32 = 1280;
pub const TOUCH_SCREEN_HEIGHT: u32 = 720;
pub const TOUCH_ATTRIBUTE_START: u32 = 1;
pub const TOUCH_ATTRIBUTE_END: u32 = 2;

/// Geometry and stable identity of one emulated touch-screen contact.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmulatedTouchContact {
    pub finger_id: u32,
    pub x: u32,
    pub y: u32,
    pub diameter_x: u32,
    pub diameter_y: u32,
    pub rotation_angle: u32,
    pub attributes: u32,
}

/// One complete Switch touch-screen sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmulatedTouchScreenState {
    contacts: [EmulatedTouchContact; MAX_TOUCH_CONTACTS],
    count: u8,
}

impl Default for EmulatedTouchScreenState {
    fn default() -> Self {
        Self {
            contacts: [EmulatedTouchContact::default(); MAX_TOUCH_CONTACTS],
            count: 0,
        }
    }
}

impl EmulatedTouchScreenState {
    #[must_use]
    pub fn contacts(&self) -> &[EmulatedTouchContact] {
        &self.contacts[..usize::from(self.count)]
    }
}

#[derive(Debug, Default)]
struct TouchAccumulator {
    contacts: [Option<EmulatedTouchContact>; MAX_TOUCH_CONTACTS],
}

impl TouchAccumulator {
    fn begin(&mut self, mut contact: EmulatedTouchContact) -> bool {
        if self
            .contacts
            .iter()
            .flatten()
            .any(|current| current.finger_id == contact.finger_id)
        {
            return false;
        }
        let Some(slot) = self.contacts.iter_mut().find(|slot| slot.is_none()) else {
            return false;
        };
        contact.attributes = TOUCH_ATTRIBUTE_START;
        *slot = Some(contact);
        true
    }

    fn update(&mut self, contact: EmulatedTouchContact) -> bool {
        let Some(current) = self
            .contacts
            .iter_mut()
            .flatten()
            .find(|current| current.finger_id == contact.finger_id)
        else {
            return false;
        };
        let attributes = current.attributes;
        *current = contact;
        current.attributes = attributes;
        true
    }

    fn end(&mut self, contact: EmulatedTouchContact) -> bool {
        if !self.update(contact) {
            return false;
        }
        let current = self
            .contacts
            .iter_mut()
            .flatten()
            .find(|current| current.finger_id == contact.finger_id)
            .expect("updated touch contact remains present");
        current.attributes |= TOUCH_ATTRIBUTE_END;
        true
    }

    fn cancel_all(&mut self) {
        for contact in self.contacts.iter_mut().flatten() {
            contact.attributes |= TOUCH_ATTRIBUTE_END;
        }
    }

    fn sample(&mut self) -> EmulatedTouchScreenState {
        let mut state = EmulatedTouchScreenState::default();
        for contact in self.contacts.iter().flatten() {
            let index = usize::from(state.count);
            state.contacts[index] = *contact;
            state.count += 1;
        }
        for slot in &mut self.contacts {
            if slot.is_some_and(|contact| contact.attributes & TOUCH_ATTRIBUTE_END != 0) {
                *slot = None;
            } else if let Some(contact) = slot {
                contact.attributes &= !TOUCH_ATTRIBUTE_START;
            }
        }
        state
    }
}

/// Main-thread producer for touch transitions.
#[derive(Clone, Debug)]
pub struct TouchScreenWriter {
    shared: Arc<Mutex<TouchAccumulator>>,
}

impl TouchScreenWriter {
    pub fn begin(&self, contact: EmulatedTouchContact) -> bool {
        self.shared.lock().unwrap().begin(contact)
    }

    pub fn update(&self, contact: EmulatedTouchContact) -> bool {
        self.shared.lock().unwrap().update(contact)
    }

    pub fn end(&self, contact: EmulatedTouchContact) -> bool {
        self.shared.lock().unwrap().end(contact)
    }

    pub fn cancel_all(&self) {
        self.shared.lock().unwrap().cancel_all();
    }
}

/// Emulation-thread consumer for complete touch-screen samples.
#[derive(Debug)]
pub struct TouchScreenReader {
    shared: Arc<Mutex<TouchAccumulator>>,
}

impl TouchScreenReader {
    pub fn sample(&mut self) -> EmulatedTouchScreenState {
        self.shared.lock().unwrap().sample()
    }
}

#[must_use]
pub fn touch_screen_channel() -> (TouchScreenWriter, TouchScreenReader) {
    let shared = Arc::new(Mutex::new(TouchAccumulator::default()));
    (
        TouchScreenWriter {
            shared: Arc::clone(&shared),
        },
        TouchScreenReader { shared },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contact(finger_id: u32, x: u32) -> EmulatedTouchContact {
        EmulatedTouchContact {
            finger_id,
            x,
            y: 200,
            diameter_x: 1,
            diameter_y: 1,
            ..EmulatedTouchContact::default()
        }
    }

    #[test]
    fn samples_preserve_start_move_and_end_transitions() {
        let (writer, mut reader) = touch_screen_channel();
        assert!(writer.begin(contact(7, 100)));
        assert!(writer.update(contact(7, 110)));
        assert_eq!(
            reader.sample().contacts(),
            &[EmulatedTouchContact {
                attributes: TOUCH_ATTRIBUTE_START,
                ..contact(7, 110)
            }]
        );
        assert_eq!(reader.sample().contacts(), &[contact(7, 110)]);

        assert!(writer.end(contact(7, 120)));
        assert_eq!(
            reader.sample().contacts(),
            &[EmulatedTouchContact {
                attributes: TOUCH_ATTRIBUTE_END,
                ..contact(7, 120)
            }]
        );
        assert!(reader.sample().contacts().is_empty());
    }

    #[test]
    fn supports_sixteen_simultaneous_contacts_and_cancels_them_together() {
        let (writer, mut reader) = touch_screen_channel();
        for finger_id in 0..MAX_TOUCH_CONTACTS as u32 {
            assert!(writer.begin(contact(finger_id, finger_id)));
        }
        assert!(!writer.begin(contact(99, 99)));
        assert_eq!(reader.sample().contacts().len(), MAX_TOUCH_CONTACTS);

        writer.cancel_all();
        let ended = reader.sample();
        assert_eq!(ended.contacts().len(), MAX_TOUCH_CONTACTS);
        assert!(
            ended
                .contacts()
                .iter()
                .all(|contact| contact.attributes == TOUCH_ATTRIBUTE_END)
        );
        assert!(reader.sample().contacts().is_empty());
    }
}

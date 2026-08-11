use std::fmt::{Display, Formatter};

macro_rules! identity {
    ($name:ident, $raw:ty) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name($raw);

        impl $name {
            #[must_use]
            pub const fn new(value: $raw) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn get(self) -> $raw {
                self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

identity!(ProcessId, u64);
identity!(GuestThreadId, u64);
identity!(VirtualCpuId, u32);
identity!(SchedulerSequence, u64);
identity!(WakeGeneration, u64);
identity!(LeaseGeneration, u64);

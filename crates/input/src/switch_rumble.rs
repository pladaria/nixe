//! Nintendo HIDAPI waveform encoding for SDL_SendGamepadEffect.

use crate::VibrationValue;

// SDL 3.4.12 accepts this ten-byte report and supplies its own packet counter:
// https://github.com/libsdl-org/SDL/blob/release-3.4.12/src/joystick/hidapi/SDL_hidapi_switch.c
// Frequency encoding and the two amplitude bit fields were measured here:
// https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering/blob/master/rumble_data_table.md
// These normalized amplitude thresholds match SDL's EncodeRumbleHighAmplitude
// and EncodeRumbleLowAmplitude tables. Saturation at 1.0 excludes the unsafe
// overdrive entries in the hardware table.
const AMPLITUDES: [u16; 101] = [
    0, 514, 775, 921, 1096, 1303, 1550, 1843, 2192, 2606, 3100, 3686, 4383, 5213, 6199, 7372, 7698,
    8039, 8395, 8767, 9155, 9560, 9984, 10426, 10887, 11369, 11873, 12398, 12947, 13520, 14119,
    14744, 15067, 15397, 15734, 16079, 16431, 16790, 17158, 17534, 17918, 18310, 18711, 19121,
    19540, 19967, 20405, 20851, 21308, 21775, 22251, 22739, 23236, 23745, 24265, 24797, 25340,
    25894, 26462, 27041, 27633, 28238, 28856, 29488, 30134, 30794, 31468, 32157, 32861, 33581,
    34316, 35068, 35836, 36620, 37422, 38242, 39079, 39935, 40809, 41703, 42616, 43549, 44503,
    45477, 46473, 47491, 48531, 49593, 50679, 51789, 52923, 54082, 55266, 56476, 57713, 58977,
    60268, 61588, 62936, 64315, 65535,
];

fn amplitude(value: f32) -> u8 {
    let normalized = (value.min(1.0) * f32::from(u16::MAX)).round() as u16;
    AMPLITUDES.partition_point(|&threshold| threshold < normalized) as u8
}

fn frequency(value: f32, offset: u16) -> u16 {
    // Quantize to the actuator's representable band, including requests at 0 Hz.
    let encoded = ((value.max(1.0) / 10.0).log2() * 32.0).round();
    (encoded.clamp(f32::from(offset + 1), f32::from(offset + 127)) as u16) - offset
}

fn actuator(value: VibrationValue) -> [u8; 4] {
    if value.is_stopped() {
        return [0x00, 0x01, 0x40, 0x40];
    }
    let high_frequency = frequency(value.high_frequency, 0x60) * 4;
    let low_frequency = frequency(value.low_frequency, 0x40) as u8;
    let high_amplitude = amplitude(value.high_amplitude);
    let low_amplitude = amplitude(value.low_amplitude);
    [
        high_frequency as u8,
        (high_amplitude << 1) | (high_frequency >> 8) as u8,
        low_frequency | ((low_amplitude & 1) << 7),
        0x40 + (low_amplitude >> 1),
    ]
}

pub(crate) fn packet(values: [VibrationValue; 2]) -> [u8; 10] {
    let mut packet = [0; 10];
    packet[0] = 0x10;
    packet[2..6].copy_from_slice(&actuator(values[0]));
    packet[6..10].copy_from_slice(&actuator(values[1]));
    packet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_preserves_independent_sides_bands_and_neutral_stop() {
        let wave = VibrationValue {
            low_amplitude: 0.5,
            low_frequency: 160.0,
            high_amplitude: 0.5,
            high_frequency: 320.0,
        };
        // Table entry 0x88 / 0x0062 is 0.50117 normalized force.
        assert_eq!(
            packet([wave, VibrationValue::default()]),
            [0x10, 0, 0, 0x89, 0x40, 0x62, 0, 1, 0x40, 0x40]
        );
        let wave = VibrationValue {
            low_amplitude: 514.0 / 65535.0,
            high_amplitude: 0.0,
            ..wave
        };
        assert_eq!(actuator(wave), [0, 1, 0xc0, 0x40]);
        assert_eq!(
            actuator(VibrationValue {
                low_frequency: 0.0,
                high_frequency: f32::MAX,
                low_amplitude: 20.0,
                high_amplitude: 20.0
            }),
            [0xfc, 0xc9, 1, 0x72]
        );
        assert_eq!(
            actuator(VibrationValue {
                low_frequency: 123.0,
                high_frequency: 456.0,
                ..Default::default()
            }),
            [0, 1, 0x40, 0x40]
        );
    }
}

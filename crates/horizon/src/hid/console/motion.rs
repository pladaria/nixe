//! The console's motion filter follows Eden's default MotionInput path.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/frontend/motion_input.cpp
use super::*;

#[derive(Debug)]
pub(in crate::hid) struct ConsoleMotion {
    pub acceleration: [f32; 3],
    pub gyroscope: [f32; 3],
    pub bias: [f32; 3],
    pub quaternion: [f32; 4],
    error: [f32; 3],
    integral: [f32; 3],
    accelerometer_only: bool,
    reset_counter: u32,
}

impl Default for ConsoleMotion {
    fn default() -> Self {
        Self {
            acceleration: [0.0; 3],
            gyroscope: [0.0; 3],
            bias: [0.0; 3],
            quaternion: [0.0, 0.0, -1.0, 0.0],
            error: [0.0; 3],
            integral: [0.0; 3],
            accelerometer_only: true,
            reset_counter: 0,
        }
    }
}

fn length(v: [f32; 3]) -> f32 {
    v.into_iter().map(|x| x * x).sum::<f32>().sqrt()
}

impl ConsoleMotion {
    pub fn moving(&self, sensitivity: f32) -> bool {
        let acceleration = length(self.acceleration);
        length(self.gyroscope) >= sensitivity || acceleration <= 0.9 || acceleration >= 1.1
    }

    pub fn update(&mut self, state: Option<&EmulatedControllerState>, delta: Duration) {
        let Some(state) = state else {
            return;
        };
        if state.accelerometer.is_none() && state.gyroscope.is_none() {
            return;
        }
        // The console uses the first player's source with the same SDL units
        // and axis conversion as Eden's SDLDriver::UpdateMotion.
        if let Some(a) = state.accelerometer {
            self.acceleration =
                [-a.x, a.z, -a.y].map(|value| (value / STANDARD_GRAVITY).clamp(-7.0, 7.0));
        }
        if let Some(g) = state.gyroscope {
            let raw = [g.x, -g.z, g.y].map(|value| value / std::f32::consts::TAU);
            self.gyroscope = std::array::from_fn(|i| (raw[i] - self.bias[i]).clamp(-5.0, 5.0));
            if !self.moving(0.05) {
                self.bias = std::array::from_fn(|i| self.bias[i] * 0.9999 + raw[i] * 0.0001);
            }
            // The console leaves the user gyro threshold at its default zero.
            self.accelerometer_only = false;
        }
        if length(self.error) >= 0.1 && !self.accelerometer_only {
            if !self.moving(0.05) && self.acceleration[2] <= -0.9 {
                self.reset_counter += 1;
                if self.reset_counter > 900 {
                    self.quaternion = [0.0, 0.0, -1.0, 0.0];
                    for _ in 1..100 {
                        if length(self.error) < 0.01 {
                            break;
                        }
                        self.align_to_accelerometer();
                    }
                    self.integral = [0.0; 3];
                    self.reset_counter = 0;
                }
            } else {
                self.reset_counter = 0;
            }
        }
        let dt = delta.as_secs_f32();
        if dt > 0.1 {
            return;
        }
        let [gx, gy, gz] = self.gyroscope.map(|value| value * std::f32::consts::TAU);
        let mut angular = if self.accelerometer_only {
            [0.0; 3]
        } else {
            [gy, -gx, -gz]
        };
        if (0.75..=1.25).contains(&length(self.acceleration)) {
            self.measure_error();
            if length(self.error) >= 0.05 {
                self.integral = std::array::from_fn(|i| self.integral[i] + self.error[i]);
            } else {
                self.integral = [0.0; 3];
            }
            let (kp, ki) = if self.accelerometer_only {
                (35.0 * 0.3, 10.0 * 0.005)
            } else {
                (0.3, 0.005)
            };
            angular =
                std::array::from_fn(|i| angular[i] + kp * self.error[i] + ki * self.integral[i]);
            if self.accelerometer_only {
                self.gyroscope = [-angular[1], angular[0], -angular[2]];
            }
        }
        self.integrate(angular, dt);
    }

    fn measure_error(&mut self) {
        let magnitude = length(self.acceleration);
        let [ax, ay, az] = self.acceleration.map(|value| value / magnitude);
        let [x, y, z, w] = self.quaternion;
        let vx = 2.0 * (x * z - w * y);
        let vy = 2.0 * (w * x + y * z);
        let vz = w * w - x * x - y * y + z * z;
        self.error = [-az * vx + ax * vz, ay * vz + az * vy, -ax * vy - ay * vx];
    }

    fn integrate(&mut self, angular: [f32; 3], dt: f32) {
        let [gy, gx, gz] = angular;
        let [x, y, z, w] = self.quaternion;
        // Eden updates w before the three vector components.
        let w = w + (-x * gx - y * gy - z * gz) * (0.5 * dt);
        let next = [
            x + (w * gx + y * gz - z * gy) * (0.5 * dt),
            y + (w * gy - x * gz + z * gx) * (0.5 * dt),
            z + (w * gz + x * gy - y * gx) * (0.5 * dt),
            w,
        ];
        let magnitude = next.into_iter().map(|x| x * x).sum::<f32>().sqrt();
        self.quaternion = next.map(|value| value / magnitude);
    }

    fn align_to_accelerometer(&mut self) {
        self.measure_error();
        let angular =
            std::array::from_fn(|i| 10.0 * 0.3 * self.error[i] + 5.0 * 0.005 * self.integral[i]);
        self.integrate(angular, 0.015);
    }
}

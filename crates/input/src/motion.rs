//! Console-independent IMU integration in SDL's right-handed sensor frame.

use std::time::Duration;

use crate::MotionVector;

const GRAVITY: f64 = 9.806_65;

/// Integrated rotation in radians and body basis vectors expressed in the
/// reference frame (X right, Y up, Z towards the player). Heading is relative:
/// a six-axis IMU has no absolute yaw reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionEstimate {
    pub angle: MotionVector,
    pub orientation: [[f32; 3]; 3],
}

pub(crate) struct MotionIntegrator {
    quaternion: [f64; 4],
    angle: [f64; 3],
    initialized: bool,
}

impl Default for MotionIntegrator {
    fn default() -> Self {
        Self {
            quaternion: [1.0, 0.0, 0.0, 0.0],
            angle: [0.0; 3],
            initialized: false,
        }
    }
}

impl MotionIntegrator {
    pub fn update(
        &mut self,
        gyro: Option<MotionVector>,
        accel: Option<MotionVector>,
        delta: Duration,
    ) -> Option<MotionEstimate> {
        let (Some(gyro), Some(accel)) = (gyro, accel) else {
            *self = Self::default();
            return None;
        };
        let angular_velocity = [f64::from(gyro.x), f64::from(gyro.y), f64::from(gyro.z)];
        let acceleration = [f64::from(accel.x), f64::from(accel.y), f64::from(accel.z)];
        // SDL reports rad/s and m/s², including gravity:
        // https://wiki.libsdl.org/SDL3/SDL_SensorType
        let magnitude = norm(acceleration);
        let gravity = ((0.8 * GRAVITY..=1.2 * GRAVITY).contains(&magnitude))
            .then(|| acceleration.map(|value| value / magnitude));
        if !self.initialized {
            // Align measured gravity to +Y without inventing absolute heading.
            if let Some([x, y, z]) = gravity {
                self.quaternion = if y < -0.999_999 {
                    [0.0, 1.0, 0.0, 0.0]
                } else {
                    normalize([1.0 + y, -z, 0.0, x])
                };
            }
            self.initialized = true;
        } else {
            let dt = delta.as_secs_f64();
            for (angle, velocity) in self.angle.iter_mut().zip(angular_velocity) {
                *angle += velocity * dt;
            }
            let mut corrected = angular_velocity;
            if let Some(measured) = gravity {
                let [w, x, y, z] = self.quaternion;
                let predicted = [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ];
                // Proportional gravity feedback (Mahony-style complementary
                // filter). Reject clearly non-gravitational acceleration;
                // gravity corrects tilt, never an unobservable yaw heading.
                // https://doi.org/10.1109/TAC.2008.923738
                let error = cross(measured, predicted);
                for (velocity, correction) in corrected.iter_mut().zip(error) {
                    *velocity += 2.0 * correction;
                }
            }
            let speed = norm(corrected);
            if speed > f64::EPSILON && dt > 0.0 {
                let (sin, cos) = (speed * dt * 0.5).sin_cos();
                let scale = sin / speed;
                let rotation = [
                    cos,
                    corrected[0] * scale,
                    corrected[1] * scale,
                    corrected[2] * scale,
                ];
                self.quaternion = normalize(multiply(self.quaternion, rotation));
            }
        }
        let [w, x, y, z] = self.quaternion;
        Some(MotionEstimate {
            angle: MotionVector {
                x: self.angle[0] as f32,
                y: self.angle[1] as f32,
                z: self.angle[2] as f32,
            },
            orientation: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + w * z),
                    2.0 * (x * z - w * y),
                ],
                [
                    2.0 * (x * y - w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z + w * x),
                ],
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
            .map(|axis| axis.map(|value| value as f32)),
        })
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(q: [f64; 4]) -> [f64; 4] {
    let length = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    q.map(|value| value / length)
}

fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [w, x, y, z] = a;
    let [v, i, j, k] = b;
    [
        w * v - x * i - y * j - z * k,
        w * i + x * v + y * k - z * j,
        w * j - x * k + y * v + z * i,
        w * k + x * j - y * i + z * v,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(x: f32, y: f32, z: f32) -> Option<MotionVector> {
        Some(MotionVector { x, y, z })
    }
    fn near(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.0001, "{actual} != {expected}");
    }

    #[test]
    fn stationary_and_missing_sensors() {
        let mut integrator = MotionIntegrator::default();
        for _ in 0..1000 {
            let state = integrator
                .update(
                    vector(0.0, 0.0, 0.0),
                    vector(0.0, GRAVITY as f32, 0.0),
                    Duration::from_millis(5),
                )
                .unwrap();
            assert_eq!(state.angle, MotionVector::default());
            assert_eq!(
                state.orientation,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            );
        }
        assert!(
            integrator
                .update(None, vector(0.0, 9.8, 0.0), Duration::ZERO)
                .is_none()
        );
    }

    #[test]
    fn yaw_integrates_every_sample_and_survives_a_stop() {
        let mut integrator = MotionIntegrator::default();
        let accel = vector(0.0, GRAVITY as f32, 0.0);
        integrator.update(vector(0.0, 0.0, 0.0), accel, Duration::ZERO);
        // Variable intervals totaling one second; the mailbox may discard all
        // intermediate results, but must not discard their integrated motion.
        for _ in 0..100 {
            integrator.update(
                vector(0.0, std::f32::consts::FRAC_PI_2, 0.0),
                accel,
                Duration::from_millis(3),
            );
            integrator.update(
                vector(0.0, std::f32::consts::FRAC_PI_2, 0.0),
                accel,
                Duration::from_millis(7),
            );
        }
        let state = integrator
            .update(vector(0.0, 0.0, 0.0), accel, Duration::from_secs(1))
            .unwrap();
        near(state.angle.y, std::f32::consts::FRAC_PI_2);
        near(state.orientation[0][2], -1.0);
        near(state.orientation[2][0], 1.0);
        near(state.orientation[1][1], 1.0);
    }

    #[test]
    fn gravity_initializes_tilt_and_rejects_linear_acceleration() {
        let mut integrator = MotionIntegrator::default();
        let state = integrator
            .update(
                vector(0.0, 0.0, 0.0),
                vector(GRAVITY as f32, 0.0, 0.0),
                Duration::ZERO,
            )
            .unwrap();
        near(state.orientation[0][1], 1.0);
        let accelerated = integrator
            .update(
                vector(0.0, 0.0, 0.0),
                vector(0.0, 30.0, 0.0),
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(state.orientation, accelerated.orientation);
        integrator.update(None, None, Duration::ZERO);
        let reset = integrator
            .update(
                vector(0.0, 0.0, 0.0),
                vector(0.0, GRAVITY as f32, 0.0),
                Duration::ZERO,
            )
            .unwrap();
        near(reset.orientation[0][0], 1.0);
        assert_eq!(reset.angle, MotionVector::default());
    }

    #[test]
    fn each_gyro_axis_has_the_right_sign_and_orientation_is_orthonormal() {
        for axis in 0..3 {
            let mut integrator = MotionIntegrator::default();
            // Zero acceleration models free fall: no gravity feedback.
            integrator.update(vector(0.0, 0.0, 0.0), vector(0.0, 0.0, 0.0), Duration::ZERO);
            let mut gyro = [0.0; 3];
            gyro[axis] = std::f32::consts::FRAC_PI_2;
            let state = integrator
                .update(
                    vector(gyro[0], gyro[1], gyro[2]),
                    vector(0.0, 0.0, 0.0),
                    Duration::from_secs(1),
                )
                .unwrap();
            let next = (axis + 1) % 3;
            let last = (axis + 2) % 3;
            near(state.orientation[next][last], 1.0);
            near(state.orientation[last][next], -1.0);
            for row in 0..3 {
                for column in 0..3 {
                    let dot = state.orientation[row]
                        .iter()
                        .zip(state.orientation[column])
                        .map(|(a, b)| a * b)
                        .sum();
                    near(dot, if row == column { 1.0 } else { 0.0 });
                }
            }
        }
    }

    #[test]
    fn gravity_feedback_corrects_tilt_without_changing_integrated_angle() {
        let mut integrator = MotionIntegrator::default();
        integrator.update(
            vector(0.0, 0.0, 0.0),
            vector(0.0, GRAVITY as f32, 0.0),
            Duration::ZERO,
        );
        // Model an orientation error, not a measured angular rotation.
        integrator.quaternion = normalize([1.0, 0.2, 0.0, 0.0]);
        let mut state = None;
        for _ in 0..2000 {
            state = integrator.update(
                vector(0.0, 0.0, 0.0),
                vector(0.0, GRAVITY as f32, 0.0),
                Duration::from_millis(5),
            );
        }
        let state = state.unwrap();
        near(state.orientation[1][1], 1.0);
        near(state.orientation[1][2], 0.0);
        assert_eq!(state.angle, MotionVector::default());
    }
}

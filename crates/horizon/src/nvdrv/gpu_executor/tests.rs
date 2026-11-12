use super::*;

#[test]
fn admission_accounts_owned_bytes_and_releases_them_before_device_completion() {
    let budget = Arc::new(GpuWorkBudget::new(&crate::test_memory::kernel()));
    let Some(GpuSubmissionAdmission::Ready(first)) = budget.reserve(MAX_GPU_INGRESS_BYTES - 8)
    else {
        panic!("initial capacity")
    };
    assert!(matches!(
        budget.reserve(9),
        Some(GpuSubmissionAdmission::Pending(_))
    ));
    let Some(GpuSubmissionAdmission::Ready(last)) = budget.reserve(8) else {
        panic!("exact remaining capacity")
    };
    assert_eq!(budget.state.lock().unwrap().bytes, MAX_GPU_INGRESS_BYTES);
    drop(first);
    assert_eq!(budget.state.lock().unwrap().bytes, 8);
    let Some(GpuSubmissionAdmission::Ready(replacement)) =
        budget.reserve(MAX_GPU_INGRESS_BYTES - 8)
    else {
        panic!("consumed descriptor storage reusable")
    };
    drop(last);
    drop(replacement);
    assert_eq!(budget.state.lock().unwrap().bytes, 0);
}

use super::*;

#[test]
fn canceled_or_rejected_host_packets_dispose_unapplied_ui_deltas() {
    fn packet() -> HostFrame {
        let mut textures = egui::TexturesDelta::default();
        textures
            .set
            .entry(egui::TextureId::Managed(0))
            .or_default()
            .push(egui::epaint::ImageDelta::full(
                egui::ColorImage::filled([1, 1], egui::Color32::WHITE),
                egui::TextureOptions::LINEAR,
            ));
        HostFrame {
            frame: None,
            size: [8, 8],
            ui: Some(UiFrame {
                primitives: vec![],
                textures,
                pixels_per_point: 1.,
            }),
            screenshot: None,
        }
    }
    let client = PresentationClient::default();
    client.submit(packet()).unwrap();
    assert!(!client.ready().unwrap());
    assert!(client.submit(packet()).is_err());
    client.close();
    assert!(client.stopped());
    assert!(client.submit(packet()).is_err());
    let failed = PresentationClient::default();
    failed.fail("presentation failed".into());
    assert!(failed.ready().is_err());
    assert_eq!(failed.submit(packet()).unwrap_err(), "presentation failed");
}

use super::*;
pub(crate) fn view(c: &Call, texture: u64, view: u64) -> Result<Texture> {
    crate::view::apply(
        c.read(texture)?,
        (view != 0).then(|| c.read(view)).transpose()?,
    )
    .map_err(Into::into)
}

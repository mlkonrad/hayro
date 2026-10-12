use crate::{GlobalState, Renderer, convert_blend_mode, derive_settings};
use hayro_interpret::{BlendMode, CacheKey, MaskType, SoftMask};
use kurbo::Rect;
use rustc_hash::FxHashMap;
use vello_cpu::color::palette::css::BLACK;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::{Mask, Pixmap, RenderSettings};

/// A cache of rendered soft masks that is shared across the pages of a document.
///
/// Rendered masks have the size of the whole page, so the cache evicts the least recently
/// used masks once their total size exceeds a budget.
#[derive(Default)]
pub(crate) struct SoftMaskCache {
    entries: FxHashMap<SharedMaskKey, (Mask, u64)>,
    size: usize,
    clock: u64,
}

impl SoftMaskCache {
    /// The maximum total number of mask pixels (one byte each) kept in the cache.
    const BUDGET: usize = 64 * 1024 * 1024;

    fn get(&mut self, key: SharedMaskKey) -> Option<Mask> {
        self.clock += 1;
        let (mask, last_used) = self.entries.get_mut(&key)?;
        *last_used = self.clock;

        Some(mask.clone())
    }

    fn insert(&mut self, key: SharedMaskKey, mask: Mask) {
        let size = mask_size(&mask);

        if size > Self::BUDGET {
            return;
        }

        while self.size + size > Self::BUDGET {
            let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, (_, used))| *used) else {
                break;
            };
            let (evicted, _) = self.entries.remove(&oldest).unwrap();
            self.size -= mask_size(&evicted);
        }

        self.clock += 1;
        self.size += size;

        if let Some((old, _)) = self.entries.insert(key, (mask, self.clock)) {
            self.size -= mask_size(&old);
        }
    }
}

/// The mask's cache key, the size of the page and whether image interpolation is forced.
type SharedMaskKey = (u128, u16, u16, bool);

fn mask_size(mask: &Mask) -> usize {
    usize::from(mask.width()) * usize::from(mask.height())
}

impl Renderer<'_> {
    /// Get the rendered version of the soft mask, using the per-page cache and, if the mask
    /// does not depend on the page's resources, the cache shared across pages.
    fn soft_mask(&mut self, m: &SoftMask<'_>) -> Mask {
        let key = m.cache_key();

        if let Some(mask) = self.soft_mask_cache.get(&key) {
            return mask.clone();
        }

        let settings = *self.ctx.render_settings();
        let global = self.global;
        let width = self.ctx.width();
        let height = self.ctx.height();

        let shared_key = (self.share_soft_masks && m.has_own_resources()).then_some((
            key,
            width,
            height,
            global.force_image_interpolation,
        ));

        let cached = shared_key.and_then(|k| global.soft_mask_cache.borrow_mut().get(k));

        let mask = cached.unwrap_or_else(|| {
            let mask = draw_soft_mask(m, settings, width, height, global);

            if let Some(k) = shared_key {
                global.soft_mask_cache.borrow_mut().insert(k, mask.clone());
            }

            mask
        });

        self.soft_mask_cache.insert(key, mask.clone());

        mask
    }

    pub(super) fn apply_soft_mask(&mut self, mask: Option<&SoftMask<'_>>) {
        let mask = mask.map(|m| self.soft_mask(m));

        if let Some(mask) = mask {
            self.ctx.set_mask(mask);
        } else {
            self.ctx.reset_mask();
        }
    }

    pub(super) fn push_transparency_group(
        &mut self,
        opacity: f32,
        mask: Option<SoftMask<'_>>,
        blend_mode: BlendMode,
    ) {
        let mask = mask.map(|m| self.soft_mask(&m));

        self.ctx.push_layer(
            None,
            Some(convert_blend_mode(blend_mode)),
            Some(opacity),
            mask,
            None,
        );
    }

    pub(super) fn pop_transparency_group(&mut self) {
        self.ctx.pop_layer();
    }
}

fn draw_soft_mask(
    mask: &SoftMask<'_>,
    settings: RenderSettings,
    width: u16,
    height: u16,
    global: &GlobalState,
) -> Mask {
    let mut ctx = vello_cpu::RenderContext::new_with(width, height, derive_settings(&settings));
    let mut renderer = Renderer::new(&mut ctx, global);

    let bg_color = mask.background_color().to_rgba();
    let apply_bg = bg_color.to_rgba8() != BLACK.to_rgba8().to_u8_array();

    if apply_bg {
        renderer
            .ctx
            .set_paint(AlphaColor::<Srgb>::new(bg_color.components()));
        renderer
            .ctx
            .fill_rect(&Rect::new(0.0, 0.0, width as f64, height as f64));
        renderer.ctx.push_layer(None, None, None, None, None);
    }

    mask.interpret(&mut renderer);

    if apply_bg {
        renderer.ctx.pop_layer();
    }

    // Reuse the pixmap of a previous mask if possible. Since the scene covers the whole
    // pixmap, rendering overwrites all of its pixels.
    let mut pix = global
        .mask_pixmap
        .take()
        .filter(|p| p.width() == width && p.height() == height)
        .unwrap_or_else(|| Pixmap::new(width, height));
    renderer.ctx.flush();
    let mut resources = vello_cpu::Resources::default();
    renderer.ctx.render(&mut pix, &mut resources);

    let mut data = match mask.mask_type() {
        MaskType::Luminosity => luminance_mask_data(&pix),
        MaskType::Alpha => pix.data().iter().map(|p| p.a).collect(),
    };
    global.mask_pixmap.set(Some(pix));

    if let Some(transfer_function) = mask.transfer_function() {
        transfer_function.apply_to(&mut data);
    }

    Mask::from_parts(data, width, height)
}

/// Compute the same values as `Mask::new_luminance`, but faster.
///
/// Pixels are processed in chunks: chunks of black (or fully transparent) pixels are skipped,
/// chunks of gray pixels are looked up in a table, and the remaining chunks are processed in
/// fixed-size batches that the compiler can vectorize. The floating-point operations are
/// exactly the same as in Vello, so the result is bit-identical.
pub(crate) fn luminance_mask_data(pix: &Pixmap) -> Vec<u8> {
    const CHUNK: usize = 16;
    // The red, green and blue bytes of a pixel stored in a little-endian `u32`.
    const RGB_BITS: u32 = 0x00FF_FFFF;

    let (pixels, _) = pix.data_as_u8_slice().as_chunks::<4>();
    let mut data = vec![0; pixels.len()];

    // Masks are often drawn in shades of gray, for which the result only depends on a single
    // channel and can be looked up.
    let gray_table: [u8; 256] = std::array::from_fn(|v| {
        let v = v as u8;
        to_u8(luminance(&[v, v, v, 255]))
    });

    let (src_chunks, src_rest) = pixels.as_chunks::<CHUNK>();
    let (dst_chunks, dst_rest) = data.as_chunks_mut::<CHUNK>();

    for (src, dst) in src_chunks.iter().zip(dst_chunks) {
        let rgb = src.iter().fold(0, |acc, p| acc | u32::from_le_bytes(*p));

        if rgb & RGB_BITS == 0 {
            continue;
        }

        if src.iter().all(|p| p[0] == p[1] && p[1] == p[2]) {
            for (out, p) in dst.iter_mut().zip(src) {
                *out = gray_table[usize::from(p[0])];
            }

            continue;
        }

        let luma: [f32; CHUNK] = std::array::from_fn(|i| luminance(&src[i]));

        for (out, l) in dst.iter_mut().zip(luma) {
            *out = to_u8(l);
        }
    }

    for (p, out) in src_rest.iter().zip(dst_rest) {
        *out = to_u8(luminance(p));
    }

    data
}

/// Equivalent to `(luma * 255.0 + 0.5) as u8` for luminance values in `[0, 1]` (with slight
/// overshoot from rounding), but can be vectorized by the compiler.
#[inline(always)]
fn to_u8(luma: f32) -> u8 {
    (luma * 255.0 + 0.5).min(255.0) as i32 as u8
}

/// The luminance of a premultiplied RGBA pixel, as computed by Vello.
#[inline(always)]
fn luminance(p: &[u8; 4]) -> f32 {
    let r = f32::from(p[0]) / 255.;
    let g = f32::from(p[1]) / 255.;
    let b = f32::from(p[2]) / 255.;

    r * 0.2126 + g * 0.7152 + b * 0.0722
}

#[cfg(test)]
mod tests {
    use super::luminance_mask_data;
    use vello_cpu::color::PremulRgba8;
    use vello_cpu::{Mask, Pixmap};

    fn assert_matches_vello(pix: &Pixmap) {
        let expected = Mask::new_luminance(pix);
        let actual = Mask::from_parts(luminance_mask_data(pix), pix.width(), pix.height());

        assert!(expected == actual);
    }

    #[test]
    fn luminance_mask_matches_vello() {
        // Every possible RGB combination, laid out as a 4096 x 4096 pixmap.
        let mut pix = Pixmap::new(4096, 4096);
        for (i, p) in pix.data_mut().iter_mut().enumerate() {
            *p = PremulRgba8 {
                r: (i >> 16) as u8,
                g: (i >> 8) as u8,
                b: i as u8,
                a: 255,
            };
        }

        assert_matches_vello(&pix);
    }

    #[test]
    fn luminance_mask_matches_vello_gray() {
        // Runs of gray pixels (which take the lookup table path), mixed with colored and
        // black runs, with a size that is not a multiple of the chunk size.
        let mut pix = Pixmap::new(257, 61);
        for (i, p) in pix.data_mut().iter_mut().enumerate() {
            let v = (i / 3) as u8;
            *p = match (i / 16) % 4 {
                0 | 1 => PremulRgba8 {
                    r: v,
                    g: v,
                    b: v,
                    a: 255,
                },
                2 => PremulRgba8 {
                    r: v,
                    g: v / 2,
                    b: v / 3,
                    a: 255,
                },
                _ => PremulRgba8 {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: v,
                },
            };
        }

        assert_matches_vello(&pix);
    }
}

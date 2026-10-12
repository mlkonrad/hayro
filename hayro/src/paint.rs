use crate::Renderer;
use hayro_interpret::encode::{EncodedShadingType, texture_dimensions};
use hayro_interpret::gradient::SvgGradientKind;
use hayro_interpret::pattern::Pattern;
use hayro_interpret::util::x_y_advances;
use hayro_interpret::{CacheKey, Paint};
use kurbo::{Affine, BezPath, Rect, Shape};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use vello_cpu::color::{AlphaColor, DynamicColor, PremulRgba8, Srgb};
use vello_cpu::peniko::{ColorStop, Gradient, ImageQuality, ImageSampler};
use vello_cpu::{Image, ImageSource, PaintType, PixelMetadata, Pixmap, peniko};

/// Identifies a texture rendered from a pattern.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum TextureKey {
    Shading {
        shading: u128,
        opacity: u32,
        bbox: [u64; 4],
    },
    Tiling {
        pattern: u128,
        scale: [u32; 2],
        size: [u16; 2],
    },
}

/// A cache of textures rendered from patterns that is shared across the pages of a
/// document, evicting the least recently used textures once their total size exceeds a
/// budget.
#[derive(Default)]
pub(crate) struct TextureCache {
    entries: FxHashMap<TextureKey, (Arc<Pixmap>, Affine, u64)>,
    size: usize,
    clock: u64,
}

impl TextureCache {
    /// The maximum total size of the cached textures, in bytes.
    const BUDGET: usize = 64 * 1024 * 1024;

    fn get(&mut self, key: &TextureKey) -> Option<(Arc<Pixmap>, Affine)> {
        self.clock += 1;
        let (pixmap, transform, last_used) = self.entries.get_mut(key)?;
        *last_used = self.clock;

        Some((pixmap.clone(), *transform))
    }

    fn insert(&mut self, key: TextureKey, (pixmap, transform): (Arc<Pixmap>, Affine)) {
        let size = pixmap_size(&pixmap);

        if size > Self::BUDGET {
            return;
        }

        while self.size + size > Self::BUDGET {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, _, used))| *used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            let (evicted, _, _) = self.entries.remove(&oldest).unwrap();
            self.size -= pixmap_size(&evicted);
        }

        self.clock += 1;
        self.size += size;

        if let Some((old, _, _)) = self.entries.insert(key, (pixmap, transform, self.clock)) {
            self.size -= pixmap_size(&old);
        }
    }
}

fn pixmap_size(pixmap: &Pixmap) -> usize {
    usize::from(pixmap.width()) * usize::from(pixmap.height()) * 4
}

impl Renderer<'_> {
    #[must_use]
    pub(super) fn set_paint(
        &mut self,
        paint: &Paint<'_>,
        path_bbox: impl Fn() -> Rect,
        is_stroke: bool,
    ) -> Option<BezPath> {
        let mut paint_transform = Affine::IDENTITY;
        let mut clip_path = None;

        let paint: PaintType = match paint.clone() {
            Paint::Color(c) => {
                let c = c.to_rgba().to_rgba8();
                AlphaColor::from_rgba8(c[0], c[1], c[2], c[3]).into()
            }
            Paint::Pattern(p) => {
                let path_transform = self.ctx.transform();

                match *p {
                    Pattern::Shading(s) => {
                        const NATIVE_GRADIENT_TOLERANCE: f32 = 0.01;

                        clip_path = s.shading.clip_path.clone();
                        let encoded = s.encode_cached(&mut self.global.shading_cache.borrow_mut());
                        let mut bbox = (*path_transform * path_bbox().to_path(0.0)).bounding_box();

                        if is_stroke {
                            // Try to account for stroke in bbox.
                            let (a1, a2) = x_y_advances(path_transform);
                            let factor = a1.length().max(a2.length()) * self.ctx.stroke().width;
                            bbox = bbox.inflate(factor, factor);
                        }

                        bbox = bbox.intersect(Rect::new(
                            0.0,
                            0.0,
                            self.ctx.width() as f64,
                            self.ctx.height() as f64,
                        ));

                        if let EncodedShadingType::RadialAxial(gradient) = &encoded.shading_type
                            && let Some(native) =
                                gradient.as_svg_gradient(&encoded, bbox, NATIVE_GRADIENT_TOLERANCE)
                        {
                            paint_transform = path_transform.inverse()
                                * Affine::translate((-0.5, -0.5))
                                * native.transform;

                            let stops = native
                                .stops
                                .iter()
                                .map(|stop| ColorStop {
                                    offset: stop.offset,
                                    color: DynamicColor::from_alpha_color(AlphaColor::<Srgb>::new(
                                        stop.color,
                                    )),
                                })
                                .collect::<Vec<_>>();

                            let gradient = match native.kind {
                                SvgGradientKind::Linear { start, end } => {
                                    Gradient::new_linear(start, end)
                                }
                                SvgGradientKind::Radial {
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                } => Gradient::new_two_point_radial(
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                ),
                            }
                            .with_extend(peniko::Extend::Pad)
                            .with_stops(stops.as_slice());

                            PaintType::Gradient(gradient)
                        } else {
                            // The texture only depends on the encoded shading and the bounding
                            // box, so it can be reused (for example across pages).
                            let key = s.transfer_function.is_none().then(|| TextureKey::Shading {
                                shading: s.cache_key(),
                                opacity: s.opacity.to_bits(),
                                bbox: [bbox.x0, bbox.y0, bbox.x1, bbox.y1].map(f64::to_bits),
                            });
                            let cached = key
                                .as_ref()
                                .and_then(|k| self.global.texture_cache.borrow_mut().get(k));

                            let (pixmap, transform) = cached.unwrap_or_else(|| {
                                let (width, height) = texture_dimensions(bbox, 1.0);
                                let mut image =
                                    Vec::with_capacity(width as usize * height as usize);
                                let mut may_have_transparency = false;
                                // Neighboring pixels often have the same color, so remember the
                                // last conversion.
                                let mut last: Option<([u32; 4], PremulRgba8)> = None;
                                let (width, height, transform) =
                                    encoded.sample_texture_runs(bbox, 1.0, |sample, count| {
                                        let bits = sample.map(f32::to_bits);
                                        let pixel = match last {
                                            Some((last_bits, pixel)) if last_bits == bits => pixel,
                                            _ => {
                                                let pixel = AlphaColor::<Srgb>::new(sample)
                                                    .premultiply()
                                                    .to_rgba8();
                                                last = Some((bits, pixel));
                                                pixel
                                            }
                                        };
                                        may_have_transparency |= pixel.a != 255;
                                        image.extend(std::iter::repeat_n(pixel, count as usize));
                                    });

                                let pixmap = Arc::new(Pixmap::from_parts(
                                    bytemuck::cast_vec(image),
                                    width as u16,
                                    height as u16,
                                    PixelMetadata {
                                        may_have_transparency,
                                        ..Default::default()
                                    },
                                ));

                                if let Some(key) = key {
                                    self.global
                                        .texture_cache
                                        .borrow_mut()
                                        .insert(key, (pixmap.clone(), transform));
                                }

                                (pixmap, transform)
                            });

                            paint_transform = path_transform.inverse() * transform;

                            let image = Image {
                                image: ImageSource::Pixmap(pixmap),
                                sampler: ImageSampler {
                                    x_extend: peniko::Extend::Repeat,
                                    y_extend: peniko::Extend::Repeat,
                                    quality: ImageQuality::Medium,
                                    alpha: 1.0,
                                },
                            };

                            PaintType::Image(image)
                        }
                    }
                    Pattern::Tiling(t) => {
                        const MAX_PIXMAP_SIZE: f32 = 3000.0;
                        // TODO: Raise this limit and perform downsampling if reached
                        // (see pdftc_100k_0138.pdf).
                        const MIN_PIXMAP_SIZE: f32 = 1.0;

                        let bbox = t.bbox;
                        let max_x_scale = MAX_PIXMAP_SIZE / bbox.width() as f32;
                        let min_x_scale = MIN_PIXMAP_SIZE / bbox.width() as f32;
                        let max_y_scale = MAX_PIXMAP_SIZE / bbox.height() as f32;
                        let min_y_scale = MIN_PIXMAP_SIZE / bbox.height() as f32;

                        let (mut xs, mut ys) = {
                            let (x, y) = x_y_advances(&(t.matrix));
                            (x.length() as f32, y.length() as f32)
                        };
                        xs = xs.max(min_x_scale).min(max_x_scale);
                        ys = ys.max(min_y_scale).min(max_y_scale);

                        let x_step = xs * t.x_step;
                        let y_step = ys * t.y_step;

                        let scaled_width = bbox.width() as f32 * xs;
                        let scaled_height = bbox.height() as f32 * ys;
                        let pix_width = x_step.abs().round() as u16;
                        let pix_height = y_step.abs().round() as u16;

                        let mut initial_transform = Affine::scale_non_uniform(xs as f64, ys as f64)
                            * Affine::translate((-bbox.x0, -bbox.y0));

                        // The rendered pattern only depends on the pattern, the scale and the
                        // size, so it can be reused (for example across pages).
                        let key = t.render_cache_key().map(|pattern| TextureKey::Tiling {
                            pattern,
                            scale: [xs.to_bits(), ys.to_bits()],
                            size: [pix_width, pix_height],
                        });
                        let cached = key
                            .as_ref()
                            .and_then(|k| self.global.texture_cache.borrow_mut().get(k))
                            .map(|(pix, _)| pix);

                        let pix = cached.unwrap_or_else(|| {
                            let mut ctx = self.child_context(pix_width, pix_height);
                            let mut renderer = Renderer::new(&mut ctx, self.global);
                            renderer.inside_pattern = true;
                            t.interpret(&mut renderer, initial_transform, is_stroke);
                            let mut pix = Pixmap::new(pix_width, pix_height);
                            renderer.ctx.flush();
                            let mut resources = vello_cpu::Resources::default();
                            renderer.ctx.render(&mut pix, &mut resources);
                            let pix = Arc::new(pix);

                            if let Some(key) = key {
                                self.global
                                    .texture_cache
                                    .borrow_mut()
                                    .insert(key, (pix.clone(), Affine::IDENTITY));
                            }

                            pix
                        });

                        // TODO: Fix these
                        if x_step < 0.0 {
                            initial_transform *=
                                Affine::new([-1.0, 0.0, 0.0, 1.0, scaled_width as f64, 0.0]);
                        }

                        if y_step < 0.0 {
                            initial_transform *=
                                Affine::new([1.0, 0.0, 0.0, -1.0, 0.0, scaled_height as f64]);
                        }

                        paint_transform =
                            path_transform.inverse() * t.matrix * initial_transform.inverse();

                        let image = Image {
                            image: ImageSource::Pixmap(pix),
                            sampler: ImageSampler {
                                x_extend: peniko::Extend::Repeat,
                                y_extend: peniko::Extend::Repeat,
                                quality: ImageQuality::Medium,
                                alpha: 1.0,
                            },
                        };

                        PaintType::Image(image)
                    }
                }
            }
        };

        self.ctx.set_paint_transform(paint_transform);
        self.ctx.set_paint(paint);

        clip_path
    }
}

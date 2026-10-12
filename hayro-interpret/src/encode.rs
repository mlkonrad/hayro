//! Encoding shading patterns for easy sampling.

use crate::CacheKey;
use crate::color::{AlphaColor, ColorComponents, ColorSpace};
use crate::interpret::state::ActiveTransferFunction;
use crate::pattern::ShadingPattern;
use crate::shading::{
    PATCH_GRID_SIZE, PatchGrid, ShadingFunction, ShadingType, Triangle, inflate_triangle,
    patch_cell_triangles, patch_grid_point,
};
use crate::util::x_y_advances;
use kurbo::{Affine, Point, Rect, Shape};
use rustc_hash::FxHashMap;
use smallvec::{ToSmallVec, smallvec};
use std::sync::Arc;

/// A shading pattern that was encoded so it can be sampled.
#[derive(Clone, Debug)]
pub struct EncodedShadingPattern {
    /// The base transform of the shading pattern.
    pub base_transform: Affine,
    pub(crate) color_space: ColorSpace,
    pub(crate) background_color: AlphaColor,
    /// The encoded shading type.
    pub shading_type: EncodedShadingType,
    pub(crate) opacity: f32,
    pub(crate) transfer_function: Option<ActiveTransferFunction>,
}

/// The encoded shading type.
#[derive(Clone, Debug)]
pub enum EncodedShadingType {
    /// Function-based shading.
    FunctionBased(EncodedFunctionBasedShading),
    /// Axial or radial shading.
    RadialAxial(EncodedRadialAxialShading),
    /// Mesh-based shading sampled into device pixels.
    Sampled(EncodedSampledShading),
    /// Empty dummy shading.
    Dummy(EncodedDummyShading),
}

/// Encoded function-based shading.
#[derive(Clone, Debug)]
pub struct EncodedFunctionBasedShading {
    pub(crate) domain: Rect,
    pub(crate) function: ShadingFunction,
}

/// Encoded axial or radial shading.
#[derive(Clone, Debug)]
pub struct EncodedRadialAxialShading {
    pub(crate) function: ShadingFunction,
    pub(crate) params: RadialAxialParams,
    pub(crate) domain: [f32; 2],
    pub(crate) extend: [bool; 2],
}

/// Encoded sampled mesh shading.
#[derive(Clone, Debug)]
pub struct EncodedSampledShading {
    pub(crate) samples: Arc<Samples>,
    pub(crate) function: Option<ShadingFunction>,
}

/// The colors of a mesh shading, sampled at integer device pixel positions.
pub(crate) enum Samples {
    /// A map from pixel positions to colors, for samples that are scattered or very sparse.
    Sparse(FxHashMap<(u16, u16), ColorComponents>),
    /// A grid of colors, for samples that densely cover an area.
    Dense(DenseSamples),
}

impl Samples {
    #[inline]
    fn get(&self, x: u16, y: u16) -> Option<&[f32]> {
        match self {
            Self::Sparse(map) => map.get(&(x, y)).map(|c| c.as_slice()),
            Self::Dense(dense) => dense.get(x, y),
        }
    }

    /// The bounds `[x0, y0, x1, y1]` (with exclusive maximum) of the positions that have a
    /// sample, or `None` if there are no samples.
    fn bounds(&self) -> Option<[u32; 4]> {
        match self {
            Self::Sparse(map) => map.keys().fold(None, |bounds, &(x, y)| {
                let (x, y) = (u32::from(x), u32::from(y));
                let [x0, y0, x1, y1] = bounds.unwrap_or([x, y, x + 1, y + 1]);
                Some([x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)])
            }),
            Self::Dense(dense) => Some([
                u32::from(dense.x0),
                u32::from(dense.y0),
                u32::from(dense.x0) + dense.width as u32,
                u32::from(dense.y0) + dense.height as u32,
            ]),
        }
    }

    /// An estimate of the memory used by the samples, in bytes.
    fn size(&self) -> usize {
        match self {
            // The key, a `SmallVec<[f32; 4]>` and the overhead of the hash table.
            Self::Sparse(map) => map.len() * 48,
            Self::Dense(dense) => dense.colors.len() * 4 + dense.lengths.len(),
        }
    }
}

impl std::fmt::Debug for Samples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sparse(map) => write!(f, "Samples::Sparse({} samples)", map.len()),
            Self::Dense(d) => write!(f, "Samples::Dense({}x{})", d.width, d.height),
        }
    }
}

/// Colors sampled on the pixel grid `[x0, x0 + width) x [y0, y0 + height)`.
pub(crate) struct DenseSamples {
    x0: u16,
    y0: u16,
    width: usize,
    height: usize,
    /// The maximum number of color components of a sample.
    stride: usize,
    /// The color components of each pixel, `stride` values per pixel.
    colors: Vec<f32>,
    /// The number of color components of each pixel, or 0 if it has no sample.
    lengths: Vec<u8>,
}

impl DenseSamples {
    #[inline]
    fn get(&self, x: u16, y: u16) -> Option<&[f32]> {
        let x = usize::from(x.checked_sub(self.x0)?);
        let y = usize::from(y.checked_sub(self.y0)?);

        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = y * self.width + x;
        let len = usize::from(self.lengths[idx]);

        (len != 0).then(|| &self.colors[idx * self.stride..][..len])
    }
}

/// Encoded dummy shading.
#[derive(Clone, Debug)]
pub struct EncodedDummyShading;

/// A shading that was rasterized into a texture.
pub struct ShadingTexture {
    /// The straight (non-premultiplied) RGBA8 pixel data, row-major.
    pub data: Vec<u8>,
    /// The width of the texture in pixels.
    pub width: u32,
    /// The height of the texture in pixels.
    pub height: u32,
    /// The transform from texture space to the space of the rasterized
    /// bounding box.
    pub transform: Affine,
    /// Whether any pixel of the texture is not fully opaque.
    pub has_transparency: bool,
}

/// Return the texture dimensions [`EncodedShadingPattern::sample_texture`]
/// uses for the given bounding box and scale factor.
///
/// Exposed so that callers can bound the size of the resulting texture
/// before sampling.
pub fn texture_dimensions(bbox: Rect, scale: f32) -> (u32, u32) {
    (
        (bbox.width() as f32 * scale).max(1.0).ceil() as u32,
        (bbox.height() as f32 * scale).max(1.0).ceil() as u32,
    )
}

impl EncodedShadingPattern {
    /// Sample the shading at the given position.
    #[inline]
    pub fn sample(&self, pos: Point) -> [f32; 4] {
        self.finish_sample(
            self.shading_type
                .eval(pos, self.background_color, &self.color_space),
        )
    }

    /// Apply the opacity and transfer function to an evaluated color.
    #[inline]
    fn finish_sample(&self, color: Option<AlphaColor>) -> [f32; 4] {
        color
            .map(|v| {
                let mut components = v.components();
                components[3] *= self.opacity;

                if let Some(tf) = &self.transfer_function {
                    return tf.apply(&AlphaColor::new(components)).components();
                }

                components
            })
            .unwrap_or([0.0, 0.0, 0.0, 0.0])
    }

    /// Sample the shading on a pixel grid covering `bbox`, with `scale`
    /// pixels per unit of `bbox`.
    ///
    /// The callback is invoked once per pixel in row-major order with the
    /// straight (non-premultiplied) RGBA sample, letting callers apply their
    /// own conversion. Returns the dimensions of the grid (see
    /// [`texture_dimensions`]) and the transform from grid space to the space
    /// of `bbox`. No size cap is applied — callers that need to bound the
    /// texture size should check [`texture_dimensions`] first.
    pub fn sample_texture(
        &self,
        bbox: Rect,
        scale: f32,
        mut f: impl FnMut([f32; 4]),
    ) -> (u32, u32, Affine) {
        self.sample_texture_runs(bbox, scale, |sample, count| {
            for _ in 0..count {
                f(sample);
            }
        })
    }

    /// Like [`sample_texture`](Self::sample_texture), but the callback is invoked with runs of
    /// consecutive pixels (in row-major order) that have the same sample.
    pub fn sample_texture_runs(
        &self,
        bbox: Rect,
        scale: f32,
        mut f: impl FnMut([f32; 4], u32),
    ) -> (u32, u32, Affine) {
        let (width, height) = texture_dimensions(bbox, scale);
        let inv_scale = 1.0 / scale as f64;

        let (x_advance, y_advance) =
            x_y_advances(&(Affine::scale(inv_scale) * self.base_transform));

        let mut start_point = self.base_transform
            * Affine::translate((0.5 * inv_scale, 0.5 * inv_scale))
            * Point::new(bbox.x0, bbox.y0);

        // Sampled (mesh) shadings often only cover a small part of the area. Outside of the
        // area that contains samples, every position evaluates to the background color, so
        // it only needs to be computed once.
        let sampled = match &self.shading_type {
            EncodedShadingType::Sampled(sampled) => Some((
                sampled.samples.bounds(),
                self.finish_sample(Some(self.background_color)),
            )),
            _ => None,
        };

        for _ in 0..height {
            let mut point = start_point;

            match &sampled {
                Some((bounds, background)) => {
                    // Same conversion as when looking up a sample.
                    let pos = |p: Point| (u32::from(p.x as u16), u32::from(p.y as u16));
                    let inside = |(x, y): (u32, u32)| {
                        bounds
                            .is_some_and(|[x0, y0, x1, y1]| x >= x0 && x < x1 && y >= y0 && y < y1)
                    };
                    let row_outside = x_advance.y == 0.0
                        && !bounds.is_some_and(|[_, y0, _, y1]| {
                            let y = pos(point).1;
                            y >= y0 && y < y1
                        });

                    if row_outside {
                        f(*background, width);
                    } else {
                        let mut run = 0;

                        for _ in 0..width {
                            if inside(pos(point)) {
                                if run > 0 {
                                    f(*background, run);
                                    run = 0;
                                }
                                f(self.sample(point), 1);
                            } else {
                                run += 1;
                            }

                            point += x_advance;
                        }

                        if run > 0 {
                            f(*background, run);
                        }
                    }
                }
                None => {
                    for _ in 0..width {
                        f(self.sample(point), 1);
                        point += x_advance;
                    }
                }
            }

            start_point += y_advance;
        }

        (
            width,
            height,
            Affine::translate((bbox.x0, bbox.y0)) * Affine::scale(inv_scale),
        )
    }

    /// Rasterize the shading into a straight (non-premultiplied) RGBA8
    /// texture covering `bbox`, with `scale` pixels per unit of `bbox`.
    ///
    /// See [`sample_texture`](Self::sample_texture) for the grid semantics.
    pub fn render_texture(&self, bbox: Rect, scale: f32) -> ShadingTexture {
        let (width, height) = texture_dimensions(bbox, scale);
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        let mut has_transparency = false;

        let (_, _, transform) = self.sample_texture(bbox, scale, |sample| {
            let converted = [
                (sample[0] * 255.0 + 0.5) as u8,
                (sample[1] * 255.0 + 0.5) as u8,
                (sample[2] * 255.0 + 0.5) as u8,
                (sample[3] * 255.0 + 0.5) as u8,
            ];

            has_transparency |= converted[3] != 255;
            data.extend_from_slice(&converted);
        });

        ShadingTexture {
            data,
            width,
            height,
            transform,
            has_transparency,
        }
    }
}

/// A cache for the expensive part of encoding mesh shadings, which can be shared across
/// multiple encodings of the same shading pattern (for example across pages).
///
/// The cache evicts the least recently used entries once the total number of cached samples
/// exceeds a budget.
#[derive(Default)]
pub struct SampledShadingCache {
    entries: FxHashMap<u128, (Arc<Samples>, u64)>,
    size: usize,
    clock: u64,
}

impl SampledShadingCache {
    /// The maximum total size of the cached samples, in bytes.
    const BUDGET: usize = 64 * 1024 * 1024;

    fn get_or_insert_with(&mut self, key: u128, compute: impl FnOnce() -> Samples) -> Arc<Samples> {
        self.clock += 1;

        if let Some((samples, last_used)) = self.entries.get_mut(&key) {
            *last_used = self.clock;
            return samples.clone();
        }

        let samples = Arc::new(compute());
        let size = samples.size();

        if size <= Self::BUDGET {
            while self.size + size > Self::BUDGET {
                let Some((&oldest, _)) = self.entries.iter().min_by_key(|(_, (_, used))| *used)
                else {
                    break;
                };
                let (evicted, _) = self.entries.remove(&oldest).unwrap();
                self.size -= evicted.size();
            }

            self.size += size;
            self.entries.insert(key, (samples.clone(), self.clock));
        }

        samples
    }
}

impl ShadingPattern {
    /// Encode the shading pattern.
    pub fn encode(&self) -> EncodedShadingPattern {
        self.encode_impl(None)
    }

    /// Encode the shading pattern, reusing the samples of mesh shadings from the cache.
    ///
    /// The result is the same as for [`ShadingPattern::encode`].
    pub fn encode_cached(&self, cache: &mut SampledShadingCache) -> EncodedShadingPattern {
        self.encode_impl(Some(cache))
    }

    fn encode_impl(&self, mut cache: Option<&mut SampledShadingCache>) -> EncodedShadingPattern {
        let key = self.cache_key();
        let mut samples = |compute: &dyn Fn() -> Samples| match cache.as_mut() {
            Some(cache) => cache.get_or_insert_with(key, compute),
            None => Arc::new(compute()),
        };

        let base_transform;

        let shading_type = match self.shading.shading_type.as_ref() {
            ShadingType::FunctionBased {
                domain,
                matrix,
                function,
            } => {
                base_transform = (self.matrix * *matrix).inverse();
                encode_function_shading(domain, function)
            }
            ShadingType::RadialAxial {
                coords,
                domain,
                function,
                extend,
                axial,
            } => {
                let (encoded, initial_transform) =
                    encode_axial_shading(*coords, *domain, function, *extend, *axial);

                base_transform = initial_transform * self.matrix.inverse();

                encoded
            }
            ShadingType::TriangleMesh {
                triangles,
                function,
            } => {
                let full_transform = self.matrix;
                let samples = samples(&|| sample_triangles(triangles, full_transform));

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples,
                    function: function.clone(),
                })
            }
            ShadingType::CoonsPatchMesh { patches, function } => {
                let samples = samples(&|| {
                    let mut samples = FxHashMap::default();
                    for patch in patches {
                        sample_patch(
                            &patch.grid(),
                            |p| patch.interpolate(p),
                            self.matrix,
                            &mut samples,
                        );
                    }

                    Samples::Sparse(samples)
                });

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples,
                    function: function.clone(),
                })
            }
            ShadingType::TensorProductPatchMesh { patches, function } => {
                let samples = samples(&|| {
                    let mut samples = FxHashMap::default();
                    for patch in patches {
                        sample_patch(
                            &patch.grid(),
                            |p| patch.interpolate(p),
                            self.matrix,
                            &mut samples,
                        );
                    }

                    Samples::Sparse(samples)
                });

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples,
                    function: function.clone(),
                })
            }
            ShadingType::Dummy => {
                base_transform = Affine::IDENTITY;

                EncodedShadingType::Dummy(EncodedDummyShading)
            }
        };

        let color_space = self.shading.color_space.clone();

        let background_color = self
            .shading
            .background
            .as_ref()
            .map(|b| color_space.to_rgba(b, 1.0))
            .unwrap_or(AlphaColor::TRANSPARENT);

        EncodedShadingPattern {
            color_space,
            background_color,
            shading_type,
            base_transform,
            opacity: self.opacity,
            transfer_function: self.transfer_function.clone(),
        }
    }
}

fn encode_axial_shading(
    coords: [f32; 6],
    domain: [f32; 2],
    function: &ShadingFunction,
    extend: [bool; 2],
    is_axial: bool,
) -> (EncodedShadingType, Affine) {
    let initial_transform;

    let params = if is_axial {
        let [x_0, y_0, x_1, y_1, _, _] = coords;

        initial_transform = ts_from_line_to_line(
            Point::new(x_0 as f64, y_0 as f64),
            Point::new(x_1 as f64, y_1 as f64),
            Point::ZERO,
            Point::new(1.0, 0.0),
        );

        RadialAxialParams::Axial
    } else {
        let [x_0, y_0, r0, x_1, y_1, r_1] = coords;

        initial_transform = Affine::translate((-x_0 as f64, -y_0 as f64));
        let new_x1 = x_1 - x_0;
        let new_y1 = y_1 - y_0;

        let p1 = Point::new(new_x1 as f64, new_y1 as f64);
        let r = Point::new(r0 as f64, r_1 as f64);

        RadialAxialParams::Radial { p1, r }
    };

    (
        EncodedShadingType::RadialAxial(EncodedRadialAxialShading {
            function: function.clone(),
            params,
            domain,
            extend,
        }),
        initial_transform,
    )
}

/// The maximum number of pixels for which samples are stored densely.
const MAX_DENSE_SAMPLES: usize = 1 << 24;

fn sample_triangles(triangles: &[Triangle], transform: Affine) -> Samples {
    let triangles: Vec<Triangle> = triangles
        .iter()
        .map(|t| {
            let p0 = transform * t.p0.point;
            let p1 = transform * t.p1.point;
            let p2 = transform * t.p2.point;

            let mut v0 = t.p0.clone();
            v0.point = p0;
            let mut v1 = t.p1.clone();
            v1.point = p1;
            let mut v2 = t.p2.clone();
            v2.point = p2;

            Triangle::new(v0, v1, v2)
        })
        .collect();

    // The pixel positions each triangle is tested at.
    let ranges = |t: &Triangle| {
        let bbox = t.bounding_box();
        (
            (bbox.x0.floor() as u16)..(bbox.x1.ceil() as u16),
            (bbox.y0.floor() as u16)..(bbox.y1.ceil() as u16),
        )
    };

    let (mut x0, mut y0, mut x1, mut y1) = (u16::MAX, u16::MAX, 0, 0);
    let mut stride = 0;

    for t in &triangles {
        let (xs, ys) = ranges(t);

        if !xs.is_empty() && !ys.is_empty() {
            x0 = x0.min(xs.start);
            x1 = x1.max(xs.end);
            y0 = y0.min(ys.start);
            y1 = y1.max(ys.end);
            stride = stride.max(t.p0.colors.len());
        }
    }

    let width = usize::from(x1.saturating_sub(x0));
    let height = usize::from(y1.saturating_sub(y0));

    // Each triangle overwrites the samples of previous ones, in both representations.
    // Samples without color components can't be told apart from missing ones in the dense
    // representation (this doesn't happen for valid shadings).
    let min_len = triangles
        .iter()
        .map(|t| t.p0.colors.len())
        .min()
        .unwrap_or(0);

    if width * height > MAX_DENSE_SAMPLES || stride > usize::from(u8::MAX) || min_len == 0 {
        let mut map = FxHashMap::default();

        for t in &triangles {
            let (xs, ys) = ranges(t);

            for y in ys {
                for x in xs.clone() {
                    let point = Point::new(x as f64, y as f64);
                    if t.contains_point(point) {
                        map.insert((x, y), t.interpolate(point));
                    }
                }
            }
        }

        return Samples::Sparse(map);
    }

    let mut colors = vec![0.0; width * height * stride];
    let mut lengths = vec![0_u8; width * height];

    for t in &triangles {
        let (xs, ys) = ranges(t);

        for y in ys {
            let row = usize::from(y - y0) * width;

            for x in xs.clone() {
                let point = Point::new(x as f64, y as f64);
                if t.contains_point(point) {
                    let color = t.interpolate(point);
                    let idx = row + usize::from(x - x0);
                    colors[idx * stride..][..color.len()].copy_from_slice(&color);
                    lengths[idx] = color.len() as u8;
                }
            }
        }
    }

    Samples::Dense(DenseSamples {
        x0,
        y0,
        width,
        height,
        stride,
        colors,
        lengths,
    })
}

/// Sample the triangles that a patch is converted into (see `CoonsPatch::to_triangles` and
/// `TensorProductPatch::to_triangles`) at integer device pixel positions.
///
/// The result is exactly the same as converting the patch into triangles and passing them to
/// [`sample_triangles`]: the same floating-point operations are performed as in
/// `Triangle::new` and `Triangle::interpolate`, in the same order. However, the triangles
/// are sampled directly instead of being collected first, and the colors of grid points are
/// only computed for triangles that actually contain a pixel position, which is rare at
/// typical resolutions.
fn sample_patch(
    grid: &PatchGrid,
    interpolate: impl Fn(Point) -> ColorComponents,
    transform: Affine,
    out: &mut FxHashMap<(u16, u16), ColorComponents>,
) {
    // The colors of the grid points, computed when they are first needed.
    let mut colors: Vec<Option<ColorComponents>> = Vec::new();
    let mut device_grid = [Point::ZERO; PATCH_GRID_SIZE * PATCH_GRID_SIZE];
    let (mut max_coord, mut max_device_coord) = (0.0_f64, 0.0_f64);

    for (device, p) in device_grid.iter_mut().zip(grid) {
        *device = transform * *p;
        max_coord = max_coord.max(p.x.abs().max(p.y.abs()));
        max_device_coord = max_device_coord.max(device.x.abs().max(device.y.abs()));
    }

    // An upper bound for the distance between a computed (inflated and) transformed grid
    // point and its exact value. The rounding error of these few operations is in the order
    // of 1e-16 relative to the magnitudes involved, so this leaves a wide safety margin.
    let [a, b, c, d, e, f] = transform.as_coeffs();
    let epsilon = 1e-12
        * (1.0 + (a.abs() + b.abs() + c.abs() + d.abs()) * (1.0 + max_coord) + e.abs() + f.abs());

    // Only skip cells if all values are in a range where the reasoning in
    // `cell_has_no_samples` applies.
    let can_skip = epsilon.is_finite() && max_device_coord < 65536.0;

    for i in 0..(PATCH_GRID_SIZE - 1) {
        for j in 0..(PATCH_GRID_SIZE - 1) {
            let triangles = patch_cell_triangles(i, j);

            if can_skip && cell_has_no_samples(&device_grid, triangles, epsilon) {
                continue;
            }

            for corners in triangles {
                let points = inflate_triangle(corners.map(|idx| grid[idx]));

                for_each_sample(points.map(|p| transform * p), |x, y, (u, v, w)| {
                    if colors.is_empty() {
                        colors.resize(grid.len(), None);
                    }

                    for &idx in &corners {
                        if colors[idx].is_none() {
                            colors[idx] = Some(interpolate(patch_grid_point(
                                idx / PATCH_GRID_SIZE,
                                idx % PATCH_GRID_SIZE,
                            )));
                        }
                    }

                    let [c0, c1, c2] = corners.map(|idx| colors[idx].as_deref().unwrap());
                    let mut result = ColorComponents::new();
                    for i in 0..c0.len() {
                        result.push(u * c0[i] + v * c1[i] + w * c2[i]);
                    }

                    out.insert((x, y), result);
                });
            }
        }
    }
}

/// Return true if it is guaranteed that none of the two triangles of a grid cell contains
/// any of the pixel positions that `for_each_sample` tests, so that the cell can be skipped
/// without changing the result.
///
/// `device_grid` contains the transformed (but not inflated) grid points, which must all be
/// less than 65536 in magnitude, and `epsilon` must be finite. The check is
/// conservative: it may return false for cells without samples, but never true for cells
/// with samples, even when accounting for floating-point rounding.
///
/// The reasoning is as follows. `for_each_sample` tests the integer positions in
/// `[floor(x0), ceil(x1)) x [floor(y0), ceil(y1))`, where `(x0, y0, x1, y1)` is the bounding
/// box of the triangle. If there is no integer within `DELTA` of `[x0, x1]`, all tested
/// positions are at least `DELTA` to the left or right of the triangle (and analogously for
/// the y direction). For such a position, at least one barycentric coordinate is at most
/// `-DELTA / (2 * width)`, so the corresponding cross product computed by `winding` is at
/// least `DELTA * area / width` in magnitude and has the opposite sign of the triangle's
/// orientation, while another one is at least `area` in magnitude and has the same sign as
/// the orientation. If both values exceed the possible rounding error of the computation,
/// `winding` returns 0. This doesn't hold for (nearly) degenerate triangles, which are
/// therefore never skipped.
#[inline(always)]
fn cell_has_no_samples(device_grid: &[Point], triangles: [[usize; 3]; 2], epsilon: f64) -> bool {
    const DELTA: f64 = 1e-3;
    // `PATCH_INFLATION_FACTOR - 1`, rounded up.
    const INFLATION: f64 = 0.026;

    let corners = [
        device_grid[triangles[0][0]],
        device_grid[triangles[0][1]],
        device_grid[triangles[0][2]],
        device_grid[triangles[1][1]],
    ];

    let (mut x0, mut y0, mut x1, mut y1) = (corners[0].x, corners[0].y, corners[0].x, corners[0].y);
    for p in &corners[1..] {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }

    let (width, height) = (x1 - x0, y1 - y0);

    if !(width + height).is_finite() {
        return false;
    }

    // Inflating moves each vertex away from the triangle's centroid (which lies within the
    // cell) by at most `INFLATION` times its distance to the centroid.
    let margin = INFLATION * (width + height) + epsilon;
    let x_clear = !contains_integer(x0 - margin - DELTA, x1 + margin + DELTA);

    if !x_clear && contains_integer(y0 - margin - DELTA, y1 + margin + DELTA) {
        return false;
    }

    // The bounding box of each (inflated) triangle is at most this large.
    let (max_width, max_height) = (width + 2.0 * margin, height + 2.0 * margin);
    let extent = if x_clear { max_width } else { max_height };

    // An upper bound for the rounding error of a cross product in `winding`, whose inputs
    // are differences of at most `max_width + max_height + 2` in magnitude.
    let size = max_width + max_height + 2.0;
    let cross_error = 1e-13 * size * size;
    // An upper bound for the difference between the area computed below and the area of the
    // exact triangle: moving each vertex by at most `epsilon` changes twice the area by at
    // most `8 * epsilon * (width + height + epsilon)`, plus rounding.
    let area_error = 8.0 * epsilon * (max_width + max_height + epsilon) + cross_error;

    triangles.iter().all(|t| {
        let [p0, p1, p2] = t.map(|idx| device_grid[idx]);
        // Twice the area of the triangle. Inflating can only make it larger.
        let doubled_area = (p1 - p0).cross(p2 - p0).abs() - area_error;

        doubled_area > 0.0
            && doubled_area * DELTA > 2.0 * extent * cross_error
            && doubled_area > 2.0 * cross_error
    })
}

/// Return whether there is an integer in `[min, max]`.
///
/// Both values must be finite and less than 2^51 in magnitude.
#[inline(always)]
fn contains_integer(min: f64, max: f64) -> bool {
    // Adding and subtracting this constant rounds to the nearest integer, which is much
    // cheaper than a saturating conversion to an integer type.
    const ROUND: f64 = 6_755_399_441_055_744.0;

    let rounded = (max + ROUND) - ROUND;
    let floor = if rounded > max {
        rounded - 1.0
    } else {
        rounded
    };

    floor >= min
}

/// Equivalent to `x.floor() as u16`, but without calling into libm in the common case.
#[inline]
fn floor_u16(x: f64) -> u16 {
    // `as` conversions saturate, so only the in-range case needs to be exact.
    if x.abs() < 65536.0 {
        let truncated = x as i32;
        let floor = if (truncated as f64) > x {
            truncated - 1
        } else {
            truncated
        };
        floor.clamp(0, i32::from(u16::MAX)) as u16
    } else {
        x.floor() as u16
    }
}

/// Equivalent to `x.ceil() as u16`, but without calling into libm in the common case.
#[inline]
fn ceil_u16(x: f64) -> u16 {
    if x.abs() < 65536.0 {
        let truncated = x as i32;
        let ceil = if (truncated as f64) < x {
            truncated + 1
        } else {
            truncated
        };
        ceil.clamp(0, i32::from(u16::MAX)) as u16
    } else {
        x.ceil() as u16
    }
}

/// Call `f` with the barycentric coordinates of each integer pixel position that the
/// triangle with the given (device space) points contains, exactly like `sample_triangles`
/// does for such a triangle.
#[inline]
fn for_each_sample(points: [Point; 3], mut f: impl FnMut(u16, u16, (f32, f32, f32))) {
    let [p0, p1, p2] = points;

    let v0 = p1 - p0;
    let v1 = p2 - p0;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);

    let triangle = kurbo::Triangle::new(p0, p1, p2);
    let bbox = triangle.bounding_box();

    for y in floor_u16(bbox.y0)..ceil_u16(bbox.y1) {
        for x in floor_u16(bbox.x0)..ceil_u16(bbox.x1) {
            let pos = Point::new(x as f64, y as f64);

            if triangle.winding(pos) == 0 {
                continue;
            }

            let v2 = pos - p0;
            let d20 = v2.dot(v0);
            let d21 = v2.dot(v1);

            let denom = d00 * d11 - d01 * d01;
            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = (1.0 - v - w) as f32;

            f(x, y, (u, v as f32, w as f32));
        }
    }
}

fn encode_function_shading(domain: &[f32; 4], function: &ShadingFunction) -> EncodedShadingType {
    let domain = Rect::new(
        domain[0] as f64,
        domain[2] as f64,
        domain[1] as f64,
        domain[3] as f64,
    );

    EncodedShadingType::FunctionBased(EncodedFunctionBasedShading {
        domain,
        function: function.clone(),
    })
}

#[derive(Clone, Debug)]
pub(crate) enum RadialAxialParams {
    Axial,
    Radial { p1: Point, r: Point },
}

impl EncodedShadingType {
    pub(crate) fn eval(
        &self,
        pos: Point,
        bg_color: AlphaColor,
        color_space: &ColorSpace,
    ) -> Option<AlphaColor> {
        match self {
            Self::FunctionBased(EncodedFunctionBasedShading { domain, function }) => {
                if !domain.contains(pos) {
                    Some(bg_color)
                } else {
                    let out = function.eval(&smallvec![pos.x as f32, pos.y as f32])?;
                    // TODO: Clamp out-of-range values.
                    Some(color_space.to_rgba(&out, 1.0))
                }
            }
            Self::RadialAxial(EncodedRadialAxialShading {
                function,
                params,
                domain,
                extend,
            }) => {
                let (t0, t1) = (domain[0], domain[1]);

                let mut t = match params {
                    RadialAxialParams::Axial => pos.x as f32,
                    RadialAxialParams::Radial { p1, r } => {
                        radial_pos(&pos, p1, *r, extend[0], extend[1]).unwrap_or(f32::MIN)
                    }
                };

                if t == f32::MIN {
                    return Some(bg_color);
                }

                if t < 0.0 {
                    if extend[0] {
                        t = 0.0;
                    } else {
                        return Some(bg_color);
                    }
                } else if t > 1.0 {
                    if extend[1] {
                        t = 1.0;
                    } else {
                        return Some(bg_color);
                    }
                }

                let t = t0 + (t1 - t0) * t;

                let val = function.eval(&smallvec![t])?;

                Some(color_space.to_rgba(&val, 1.0))
            }
            Self::Sampled(EncodedSampledShading { samples, function }) => {
                if let Some(color) = samples.get(pos.x as u16, pos.y as u16) {
                    if let Some(function) = function {
                        let val = function.eval(&color.to_smallvec())?;
                        Some(color_space.to_rgba(&val, 1.0))
                    } else {
                        Some(color_space.to_rgba(color, 1.0))
                    }
                } else {
                    Some(bg_color)
                }
            }
            Self::Dummy(_) => Some(AlphaColor::TRANSPARENT),
        }
    }
}

fn ts_from_line_to_line(src1: Point, src2: Point, dst1: Point, dst2: Point) -> Affine {
    let unit_to_line1 = unit_to_line(src1, src2);
    let line1_to_unit = unit_to_line1.inverse();
    let unit_to_line2 = unit_to_line(dst1, dst2);

    unit_to_line2 * line1_to_unit
}

fn unit_to_line(p0: Point, p1: Point) -> Affine {
    Affine::new([
        p1.y - p0.y,
        p0.x - p1.x,
        p1.x - p0.x,
        p1.y - p0.y,
        p0.x,
        p0.y,
    ])
}

fn radial_pos(
    pos: &Point,
    p1: &Point,
    r: Point,
    min_extend: bool,
    max_extend: bool,
) -> Option<f32> {
    let r0 = r.x as f32;
    let dx = p1.x as f32;
    let dy = p1.y as f32;
    let dr = r.y as f32 - r0;

    let px = pos.x as f32;
    let py = pos.y as f32;

    let a = dx * dx + dy * dy - dr * dr;
    let b = -2.0 * (px * dx + py * dy + r0 * dr);
    let c = px * px + py * py - r0 * r0;

    let discriminant = b * b - 4.0 * a * c;

    // No solution available.
    if discriminant < 0.0 {
        return None;
    }

    if a.abs() < 1e-6 {
        if b.abs() < 1e-6 {
            return None;
        }

        let t = -c / b;

        if (!min_extend && t < 0.0) || (!max_extend && t > 1.0) {
            return None;
        }

        let r_t = r0 + dr * t;
        if r_t < 0.0 {
            return None;
        }

        return Some(t);
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    let max = t1.max(t2);
    let mut take_max = Some(max);
    let min = t1.min(t2);
    let mut take_min = Some(min);

    if (!min_extend && min < 0.0) || r0 + dr * min < 0.0 {
        take_min = None;
    }

    if (!max_extend && max > 1.0) || r0 + dr * max < 0.0 {
        take_max = None;
    }

    match (take_min, take_max) {
        (Some(_), Some(max)) => Some(max),
        (Some(min), None) => Some(min),
        (None, Some(max)) => Some(max),
        (None, None) => None,
    }
}

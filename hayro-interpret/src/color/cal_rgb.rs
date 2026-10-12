use super::ToRgb;
use hayro_syntax::object::Dict;
use hayro_syntax::object::dict::keys::{BLACK_POINT, GAMMA, MATRIX, WHITE_POINT};

#[derive(Debug, Clone)]
pub(crate) struct CalRgb {
    white_point: [f32; 3],
    black_point: [f32; 3],
    matrix: [f32; 9],
    gamma: [f32; 3],
}

// See <https://github.com/mozilla/pdf.js/blob/06f44916c8936b92f464d337fe3a0a6b2b78d5b4/src/core/colorspace.js#L846>
// Completely copied from there without really understanding the logic, but we get the same results as Firefox
// which should be good enough (and by viewing the `calrgb.pdf` test file in different viewers you will
// see that in many cases each viewer does whatever it wants, even Acrobat), so this is good enough for us.
impl CalRgb {
    pub(super) fn new(dict: &Dict<'_>) -> Option<Self> {
        let white_point = dict.get::<[f32; 3]>(WHITE_POINT).unwrap_or([1.0, 1.0, 1.0]);
        let black_point = dict.get::<[f32; 3]>(BLACK_POINT).unwrap_or([0.0, 0.0, 0.0]);
        let matrix = dict
            .get::<[f32; 9]>(MATRIX)
            .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let gamma = dict.get::<[f32; 3]>(GAMMA).unwrap_or([1.0, 1.0, 1.0]);

        Some(Self {
            white_point,
            black_point,
            matrix,
            gamma,
        })
    }

    const BRADFORD_SCALE_MATRIX: [f32; 9] = [
        0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296,
    ];

    const BRADFORD_SCALE_INVERSE_MATRIX: [f32; 9] = [
        0.9869929, -0.1470543, 0.1599627, 0.4323053, 0.5183603, 0.0492912, -0.0085287, 0.0400428,
        0.9684867,
    ];

    const SRGB_D65_XYZ_TO_RGB_MATRIX: [f32; 9] = [
        3.2404542, -1.5371385, -0.4985314, -0.969_266, 1.8760108, 0.0415560, 0.0556434, -0.2040259,
        1.0572252,
    ];

    const FLAT_WHITEPOINT: [f32; 3] = [1.0, 1.0, 1.0];
    const D65_WHITEPOINT: [f32; 3] = [0.95047, 1.0, 1.08883];

    fn decode_l_constant() -> f32 {
        ((8.0_f32 + 16.0) / 116.0).powi(3) / 8.0
    }

    fn srgb_transfer_function(color: f32) -> f32 {
        if color <= 0.0031308 {
            (12.92 * color).clamp(0.0, 1.0)
        } else if color >= 0.99554525 {
            1.0
        } else {
            ((1.0 + 0.055) * color.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)
        }
    }

    fn matrix_product(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
        [
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
            a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
            a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
        ]
    }

    fn to_flat(source_white_point: &[f32; 3], lms: &[f32; 3]) -> [f32; 3] {
        [
            lms[0] / source_white_point[0],
            lms[1] / source_white_point[1],
            lms[2] / source_white_point[2],
        ]
    }

    fn to_d65(source_white_point: &[f32; 3], lms: &[f32; 3]) -> [f32; 3] {
        [
            lms[0] * Self::D65_WHITEPOINT[0] / source_white_point[0],
            lms[1] * Self::D65_WHITEPOINT[1] / source_white_point[1],
            lms[2] * Self::D65_WHITEPOINT[2] / source_white_point[2],
        ]
    }

    fn decode_l(l: f32) -> f32 {
        if l < 0.0 {
            -Self::decode_l(-l)
        } else if l > 8.0 {
            ((l + 16.0) / 116.0).powi(3)
        } else {
            l * Self::decode_l_constant()
        }
    }

    fn compensate_black_point(source_bp: &[f32; 3], xyz_flat: &[f32; 3]) -> [f32; 3] {
        if source_bp == &[0.0, 0.0, 0.0] {
            return *xyz_flat;
        }

        let zero_decode_l = Self::decode_l(0.0);

        let mut out = [0.0; 3];
        for i in 0..3 {
            let src = Self::decode_l(source_bp[i]);
            let scale = (1.0 - zero_decode_l) / (1.0 - src);
            let offset = 1.0 - scale;
            out[i] = xyz_flat[i] * scale + offset;
        }

        out
    }

    fn normalize_white_point_to_flat(
        &self,
        source_white_point: &[f32; 3],
        xyz: &[f32; 3],
    ) -> [f32; 3] {
        if source_white_point[0] == 1.0 && source_white_point[2] == 1.0 {
            return *xyz;
        }
        let lms = Self::matrix_product(&Self::BRADFORD_SCALE_MATRIX, xyz);
        let lms_flat = Self::to_flat(source_white_point, &lms);
        Self::matrix_product(&Self::BRADFORD_SCALE_INVERSE_MATRIX, &lms_flat)
    }

    fn normalize_white_point_to_d65(
        &self,
        source_white_point: &[f32; 3],
        xyz: &[f32; 3],
    ) -> [f32; 3] {
        let lms = Self::matrix_product(&Self::BRADFORD_SCALE_MATRIX, xyz);
        let lms_d65 = Self::to_d65(source_white_point, &lms);
        Self::matrix_product(&Self::BRADFORD_SCALE_INVERSE_MATRIX, &lms_d65)
    }

    /// Apply the gamma of the given channel to a component.
    #[inline]
    fn apply_gamma(&self, channel: usize, value: u8) -> f32 {
        let value = value as f32 / 255.0;
        if value == 1.0 {
            1.0
        } else {
            value.powf(self.gamma[channel])
        }
    }

    /// The results of `apply_gamma` for all possible input values.
    fn gamma_tables(&self) -> [[f32; 256]; 3] {
        std::array::from_fn(|channel| std::array::from_fn(|i| self.apply_gamma(channel, i as u8)))
    }

    fn convert_pixel(&self, input: [u8; 3], gamma_tables: &[[f32; 256]; 3]) -> [u8; 3] {
        let [agr, bgg, cgb] = [
            gamma_tables[0][usize::from(input[0])],
            gamma_tables[1][usize::from(input[1])],
            gamma_tables[2][usize::from(input[2])],
        ];

        let m = &self.matrix;
        let x = m[0] * agr + m[3] * bgg + m[6] * cgb;
        let y = m[1] * agr + m[4] * bgg + m[7] * cgb;
        let z = m[2] * agr + m[5] * bgg + m[8] * cgb;
        let xyz = [x, y, z];

        let xyz_flat = self.normalize_white_point_to_flat(&self.white_point, &xyz);
        let xyz_black = Self::compensate_black_point(&self.black_point, &xyz_flat);
        let xyz_d65 = self.normalize_white_point_to_d65(&Self::FLAT_WHITEPOINT, &xyz_black);
        let srgb_xyz = Self::matrix_product(&Self::SRGB_D65_XYZ_TO_RGB_MATRIX, &xyz_d65);

        [
            (Self::srgb_transfer_function(srgb_xyz[0]) * 255.0 + 0.5) as u8,
            (Self::srgb_transfer_function(srgb_xyz[1]) * 255.0 + 0.5) as u8,
            (Self::srgb_transfer_function(srgb_xyz[2]) * 255.0 + 0.5) as u8,
        ]
    }
}

impl CalRgb {
    /// Convert all pixels, reusing the results of recently seen colors (the result only
    /// depends on the input color).
    fn convert_all<'a>(&self, pixels: impl Iterator<Item = ([u8; 3], &'a mut [u8; 3])>) {
        const MEMO_SIZE: usize = 4096;

        let gamma_tables = self.gamma_tables();
        let mut memo = vec![(u32::MAX, [0_u8; 3]); MEMO_SIZE];

        for (input, output) in pixels {
            let key = u32::from_le_bytes([input[0], input[1], input[2], 0]);
            let slot = &mut memo[(key.wrapping_mul(0x9E37_79B1) >> 20) as usize % MEMO_SIZE];

            if slot.0 != key {
                *slot = (key, self.convert_pixel(input, &gamma_tables));
            }

            *output = slot.1;
        }
    }
}

impl ToRgb for CalRgb {
    fn convert(&self, input: &[u8], output: &mut [u8]) -> Option<()> {
        self.convert_all(
            input
                .as_chunks::<3>()
                .0
                .iter()
                .copied()
                .zip(output.as_chunks_mut::<3>().0),
        );

        Some(())
    }

    fn convert_in_place(&self, input: &mut [u8]) -> Option<()> {
        let gamma_tables = self.gamma_tables();
        for input in input.as_chunks_mut::<3>().0 {
            *input = self.convert_pixel(*input, &gamma_tables);
        }

        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::CalRgb;
    use crate::color::ToRgb;

    /// The previous implementation, for comparison.
    fn convert_pixel_reference(cs: &CalRgb, input: [u8; 3]) -> [u8; 3] {
        let tables = cs.gamma_tables();
        let [r, g, b] = input.map(|v| v as f32 / 255.0);
        let [gr, gg, gb] = cs.gamma;
        let direct = [
            if r == 1.0 { 1.0 } else { r.powf(gr) },
            if g == 1.0 { 1.0 } else { g.powf(gg) },
            if b == 1.0 { 1.0 } else { b.powf(gb) },
        ];
        // The tables must contain exactly the same values.
        for (c, v) in input.iter().enumerate() {
            assert_eq!(tables[c][usize::from(*v)].to_bits(), direct[c].to_bits());
        }
        cs.convert_pixel(input, &tables)
    }

    #[test]
    fn memoized_conversion_matches() {
        let cs = CalRgb {
            white_point: [0.9505, 1.0, 1.089],
            black_point: [0.0, 0.0, 0.0],
            matrix: [
                0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505,
            ],
            gamma: [2.2, 1.8, 2.4],
        };

        // Many pixels, with repeated colors and memo collisions.
        let input: Vec<u8> = (0..300_000_u32)
            .flat_map(|i| {
                let v = i.wrapping_mul(2_654_435_761) % 5000;
                [(v % 256) as u8, (v / 7 % 256) as u8, (i % 256) as u8]
            })
            .collect();
        let mut output = vec![0; input.len()];
        cs.convert(&input, &mut output).unwrap();

        for (i, o) in input.chunks_exact(3).zip(output.chunks_exact(3)) {
            assert_eq!(
                convert_pixel_reference(&cs, [i[0], i[1], i[2]]),
                [o[0], o[1], o[2]]
            );
        }
    }
}

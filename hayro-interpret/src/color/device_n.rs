use super::{ColorSpace, ToRgb, U8Lookup};
use crate::cache::Cache;
use crate::function::Function;
use hayro_syntax::object::{Array, Name, Object};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub(crate) struct DeviceN {
    alternate_space: ColorSpace,
    pub(super) num_components: u8,
    tint_transform: Function,
    is_none: bool,
    lookup: U8Lookup<[u8; 3]>,
}

impl DeviceN {
    pub(super) fn new(array: &Array<'_>, cache: &Cache) -> Option<Self> {
        let mut iter = array.flex_iter();
        // Skip `/DeviceN`
        let _ = iter.next::<Name<'_>>()?;
        // Skip `Name`.
        let names = iter
            .next::<Array<'_>>()?
            .iter::<Name<'_>>()
            .collect::<Vec<_>>();
        let num_components = u8::try_from(names.len()).ok()?;
        let all_none = names.iter().all(|n| n.as_str() == "None");
        let alternate_space = ColorSpace::new(iter.next::<Object<'_>>()?, cache)?;
        let tint_transform = Function::new(&iter.next::<Object<'_>>()?)?;

        if num_components == 0 {
            return None;
        }

        Some(Self {
            alternate_space,
            num_components,
            tint_transform,
            is_none: all_none,
            lookup: U8Lookup::default(),
        })
    }

    fn evaluate(&self, input: &[u8]) -> Vec<u8> {
        /// The maximum number of remembered colors per call.
        const MAX_ENTRIES: usize = 1 << 16;

        let num_components = self.num_components as usize;
        let evaluate_one = |n: &[u8]| {
            let input = n.iter().map(|value| *value as f32 / 255.0).collect();
            let values = self
                .tint_transform
                .eval(input)
                .unwrap_or(self.alternate_space.initial_color());
            self.alternate_space.encode_values(&values)
        };

        if num_components > 8 {
            return input
                .chunks_exact(num_components)
                .flat_map(evaluate_one)
                .collect();
        }

        // Images usually contain many pixels with the same color, and the result only depends
        // on the input components, so remember the results of colors that were already seen.
        let mut seen: FxHashMap<u64, SmallVec<[u8; 4]>> = FxHashMap::default();
        let mut out = Vec::with_capacity(input.len() / num_components * 4);

        for n in input.chunks_exact(num_components) {
            let key = n.iter().fold(0_u64, |key, b| (key << 8) | u64::from(*b));

            if let Some(values) = seen.get(&key) {
                out.extend_from_slice(values);
            } else {
                let values = evaluate_one(n);
                out.extend_from_slice(&values);

                if seen.len() < MAX_ENTRIES {
                    seen.insert(key, values);
                }
            }
        }

        out
    }

    fn convert_inner(&self, input: &[u8], output: &mut [u8]) -> Option<()> {
        let evaluated = self.evaluate(input);
        self.alternate_space.convert(&evaluated, output)
    }

    fn u8_lookup(&self) -> Option<&[[u8; 3]; 256]> {
        self.lookup
            .get_or_init(|input, output| self.convert_inner(input, output))
    }
}

impl ToRgb for DeviceN {
    fn convert(&self, input: &[u8], output: &mut [u8]) -> Option<()> {
        if self.num_components == 1 {
            let lookup = self.u8_lookup()?;
            for (input, output) in input.iter().zip(output.as_chunks_mut::<3>().0) {
                output.copy_from_slice(&lookup[*input as usize]);
            }

            Some(())
        } else {
            self.convert_inner(input, output)
        }
    }

    fn convert_in_place(&self, input: &mut [u8]) -> Option<()> {
        if self.num_components != 3 {
            return None;
        }

        let evaluated = self.evaluate(input);
        self.alternate_space.convert(&evaluated, input)
    }

    fn is_none(&self) -> bool {
        self.is_none
    }
}

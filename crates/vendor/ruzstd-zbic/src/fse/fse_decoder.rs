use crate::bit_io::BitReaderReversed;
use crate::decoding::errors::{FSEDecoderError, FSETableError};
use alloc::vec::Vec;
use core::convert::TryFrom;

pub struct FSEDecoder<'table> {
    /// An FSE state value represents an index in the FSE table.
    pub state: Entry,
    /// A reference to the table used for decoding.
    table: &'table FSETable,
}

impl<'t> FSEDecoder<'t> {
    /// Initialize a new Finite State Entropy decoder.
    pub fn new(table: &'t FSETable) -> FSEDecoder<'t> {
        FSEDecoder {
            state: table.decode.first().copied().unwrap_or(Entry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            }),
            table,
        }
    }

    /// Returns the byte associated with the symbol the internal cursor is pointing at.
    pub fn decode_symbol(&self) -> u8 {
        self.state.symbol
    }

    /// Initialize internal state and prepare for decoding. After this, `decode_symbol` can be called
    /// to read the first symbol and `update_state` can be called to prepare to read the next symbol.
    pub fn init_state(&mut self, bits: &mut BitReaderReversed<'_>) -> Result<(), FSEDecoderError> {
        if self.table.accuracy_log == 0 {
            return Err(FSEDecoderError::TableIsUninitialized);
        }
        let new_state = bits.get_bits(self.table.accuracy_log);
        self.state = self.table.decode[new_state as usize];

        Ok(())
    }

    /// Advance the internal state to decode the next symbol in the bitstream.
    pub fn update_state(&mut self, bits: &mut BitReaderReversed<'_>) {
        let num_bits = self.state.num_bits;
        let add = bits.get_bits(num_bits);
        let base_line = self.state.base_line;
        let new_state = base_line + add as u32;
        self.state = self.table.decode[new_state as usize];

        //println!("Update: {}, {} -> {}", base_line, add,  self.state);
    }
}

/// FSE decoding involves a decoding table that describes the probabilities of
/// all literals from 0 to the highest present one
///
/// <https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md#fse-table-description>
#[derive(Debug, Clone)]
pub struct FSETable {
    /// The maximum symbol in the table (inclusive). Limits the probabilities length to max_symbol + 1.
    max_symbol: u8,
    /// The actual table containing the decoded symbol and the compression data
    /// connected to that symbol.
    pub decode: Vec<Entry>, //used to decode symbols, and calculate the next state
    /// The size of the table is stored in logarithm base 2 format,
    /// with the **size of the table** being equal to `(1 << accuracy_log)`.
    /// This value is used so that the decoder knows how many bits to read from the bitstream.
    pub accuracy_log: u8,
    /// In this context, probability refers to the likelihood that a symbol occurs in the given data.
    /// Given this info, the encoder can assign shorter codes to symbols that appear more often,
    /// and longer codes that appear less often, then the decoder can use the probability
    /// to determine what code was assigned to what symbol.
    ///
    /// The probability of a single symbol is a value representing the proportion of times the symbol
    /// would fall within the data.
    ///
    /// If a symbol probability is set to `-1`, it means that the probability of a symbol
    /// occurring in the data is less than one.
    pub symbol_probabilities: Vec<i32>, //used while building the decode Vector
    /// The number of times each symbol occurs (The first entry being 0x0, the second being 0x1) and so on
    /// up until the highest possible symbol (255).
    symbol_counter: Vec<u32>,
}

impl FSETable {
    /// Initialize a new empty Finite State Entropy decoding table.
    pub fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            symbol_probabilities: Vec::with_capacity(256), //will never be more than 256 symbols because u8
            symbol_counter: Vec::with_capacity(256), //will never be more than 256 symbols because u8
            decode: Vec::new(),                      //depending on acc_log.
            accuracy_log: 0,
        }
    }

    /// Reset `self` and update `self`'s state to mirror the provided table.
    pub fn reinit_from(&mut self, other: &Self) {
        self.reset();
        self.symbol_counter.extend_from_slice(&other.symbol_counter);
        self.symbol_probabilities
            .extend_from_slice(&other.symbol_probabilities);
        self.decode.extend_from_slice(&other.decode);
        self.accuracy_log = other.accuracy_log;
    }

    /// Empty the table and clear all internal state.
    pub fn reset(&mut self) {
        self.symbol_counter.clear();
        self.symbol_probabilities.clear();
        self.decode.clear();
        self.accuracy_log = 0;
    }

    /// returns how many BYTEs (not bits) were read while building the decoder
    pub fn build_decoder(&mut self, source: &[u8], max_log: u8) -> Result<usize, FSETableError> {
        self.accuracy_log = 0;

        let bytes_read = self.read_probabilities(source, max_log)?;
        self.build_decoding_table()?;

        Ok(bytes_read)
    }

    /// Given the provided accuracy log, build a decoding table from that log.
    pub fn build_from_probabilities(
        &mut self,
        acc_log: u8,
        probs: &[i32],
    ) -> Result<(), FSETableError> {
        if acc_log == 0 {
            return Err(FSETableError::AccLogIsZero);
        }
        self.symbol_probabilities = probs.to_vec();
        self.accuracy_log = acc_log;
        self.build_decoding_table()
    }

    /// Build the actual decoding table after probabilities have been read into the table.
    /// After this function is called, the decoding process can begin.
    fn build_decoding_table(&mut self) -> Result<(), FSETableError> {
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(FSETableError::TooManySymbols {
                got: self.symbol_probabilities.len(),
            });
        }

        self.decode.clear();

        let table_size = 1 << self.accuracy_log;
        if self.decode.len() < table_size {
            self.decode.reserve(table_size - self.decode.len());
        }
        //fill with dummy entries
        self.decode.resize(
            table_size,
            Entry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            },
        );

        let mut negative_idx = table_size; //will point to the highest index with is already occupied by a negative-probability-symbol

        //first scan for all -1 probabilities and place them at the top of the table
        for symbol in 0..self.symbol_probabilities.len() {
            if self.symbol_probabilities[symbol] == -1 {
                negative_idx -= 1;
                let entry = &mut self.decode[negative_idx];
                entry.symbol = symbol as u8;
                entry.base_line = 0;
                entry.num_bits = self.accuracy_log;
            }
        }

        //then place in a semi-random order all of the other symbols
        let mut position = 0;
        for idx in 0..self.symbol_probabilities.len() {
            let symbol = idx as u8;
            if self.symbol_probabilities[idx] <= 0 {
                continue;
            }

            //for each probability point the symbol gets on slot
            let prob = self.symbol_probabilities[idx];
            for _ in 0..prob {
                let entry = &mut self.decode[position];
                entry.symbol = symbol;

                position = next_position(position, table_size);
                while position >= negative_idx {
                    position = next_position(position, table_size);
                    //everything above negative_idx is already taken
                }
            }
        }

        // baselines and num_bits can only be calculated when all symbols have been spread
        self.symbol_counter.clear();
        self.symbol_counter
            .resize(self.symbol_probabilities.len(), 0);
        for idx in 0..negative_idx {
            let entry = &mut self.decode[idx];
            let symbol = entry.symbol;
            let prob = self.symbol_probabilities[symbol as usize];

            let symbol_count = self.symbol_counter[symbol as usize];
            let (bl, nb) = calc_baseline_and_numbits(table_size as u32, prob as u32, symbol_count);

            //println!("symbol: {:2}, table: {}, prob: {:3}, count: {:3}, bl: {:3}, nb: {:2}", symbol, table_size, prob, symbol_count, bl, nb);

            assert!(nb <= self.accuracy_log);
            self.symbol_counter[symbol as usize] += 1;

            entry.base_line = bl;
            entry.num_bits = nb;
        }
        Ok(())
    }

    /// Read Nintendo's binary-interpolative probability table.
    ///
    /// ZBIC keeps the Zstandard frame and block layout, but replaces the standard FSE normalized
    /// count representation with this cumulative binary-interpolative form. This is a safe Rust
    /// translation of the format implemented by the Horizon/Atmosphere loader, with every index,
    /// arithmetic operation, and cumulative count checked before it is used.
    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, FSETableError> {
        self.symbol_probabilities.clear();
        let first = *source.first().ok_or(FSETableError::InvalidBicData)?;
        let mut raw_bytes_left = usize::from(first & 0x7f);
        let low_probability = u32::from(first >> 7);
        let table_bytes = raw_bytes_left
            .checked_add(1)
            .ok_or(FSETableError::InvalidBicData)?;
        if table_bytes >= source.len() {
            return Err(FSETableError::InvalidBicData);
        }

        let refill = |mut value: u64, bytes_left: &mut usize| {
            while *bytes_left != 0 && value >> 32 < 0x100 {
                value = value
                    .wrapping_shl(8)
                    .wrapping_add(u64::from(source[*bytes_left]));
                *bytes_left -= 1;
            }
            value
        };

        let packed = refill(0, &mut raw_bytes_left);
        let mut encoded_symbols = packed / 0x34;
        let max_symbol_index =
            u32::try_from(packed % 0x34).map_err(|_| FSETableError::InvalidBicData)? + 1;
        encoded_symbols = refill(encoded_symbols, &mut raw_bytes_left);
        if max_symbol_index > u32::from(self.max_symbol) {
            return Err(FSETableError::TooManySymbols {
                got: max_symbol_index as usize + 1,
            });
        }

        let mut cumulative_table = encoded_symbols >> 3;
        self.accuracy_log = u8::try_from(encoded_symbols & 7)
            .map_err(|_| FSETableError::InvalidBicData)?
            + ACC_LOG_OFFSET;
        if self.accuracy_log > max_log {
            return Err(FSETableError::AccLogTooBig {
                got: self.accuracy_log,
                max: max_log,
            });
        }
        cumulative_table = refill(cumulative_table, &mut raw_bytes_left);

        let probability_sum = 1_u32
            .checked_shl(u32::from(self.accuracy_log))
            .ok_or(FSETableError::InvalidBicData)?;
        let mut interpolation_code = cumulative_table / u64::from(probability_sum);
        interpolation_code = refill(interpolation_code, &mut raw_bytes_left);
        let final_count = if low_probability == 0 {
            u32::try_from(cumulative_table % u64::from(probability_sum))
                .map_err(|_| FSETableError::InvalidBicData)?
                + 1
        } else {
            max_symbol_index
                .checked_add(
                    u32::try_from(cumulative_table % u64::from(probability_sum))
                        .map_err(|_| FSETableError::InvalidBicData)?,
                )
                .and_then(|value| value.checked_add(2))
                .ok_or(FSETableError::InvalidBicData)?
        };

        let interpolation_size = (if max_symbol_index.is_power_of_two() {
            max_symbol_index.checked_mul(2)
        } else {
            max_symbol_index.checked_next_power_of_two()
        })
        .filter(|size| *size <= 0xff)
        .ok_or(FSETableError::InvalidBicData)? as usize;
        let mut cumulative = [0_u32; 257];
        cumulative[interpolation_size] = final_count;

        let mut intervals = Vec::with_capacity(interpolation_size);
        intervals.push((0_usize, interpolation_size));
        while let Some((lower, upper)) = intervals.pop() {
            if upper - lower <= 1 {
                continue;
            }
            let middle = lower + (upper - lower) / 2;
            let first_count = cumulative[lower];
            let last_count = cumulative[upper];
            if first_count > last_count {
                return Err(FSETableError::InvalidBicData);
            }
            if first_count == last_count {
                cumulative[lower + 1..upper].fill(first_count);
            } else {
                let radix = u64::from(last_count - first_count + 1);
                cumulative[middle] = u32::try_from(interpolation_code % radix)
                    .map_err(|_| FSETableError::InvalidBicData)?
                    .checked_add(first_count)
                    .ok_or(FSETableError::InvalidBicData)?;
                interpolation_code /= radix;
                interpolation_code = refill(interpolation_code, &mut raw_bytes_left);
            }
            // LIFO ordering reproduces the format's right-first depth-first traversal.
            intervals.push((lower, middle));
            intervals.push((middle, upper));
        }
        // The encoding includes one final degenerate [0, 1] interval.
        if cumulative[0] != cumulative[1] {
            let radix = u64::from(cumulative[1] - cumulative[0] + 1);
            cumulative[0] = u32::try_from(interpolation_code % radix)
                .map_err(|_| FSETableError::InvalidBicData)?;
            interpolation_code /= radix;
            let _ = refill(interpolation_code, &mut raw_bytes_left);
        }

        let mut accumulated = 0_u32;
        let mut remaining = probability_sum;
        for &count in &cumulative[1..=max_symbol_index as usize + 1] {
            let distance = count
                .checked_sub(accumulated)
                .ok_or(FSETableError::InvalidBicData)?;
            accumulated = count;
            let probability = i32::try_from(distance).map_err(|_| FSETableError::InvalidBicData)?
                - i32::try_from(low_probability).map_err(|_| FSETableError::InvalidBicData)?;
            remaining = remaining
                .checked_sub(probability.unsigned_abs())
                .ok_or(FSETableError::InvalidBicData)?;
            self.symbol_probabilities.push(probability);
        }
        if remaining != 0 || raw_bytes_left != 0 {
            return Err(FSETableError::InvalidBicData);
        }
        Ok(table_bytes)
    }
}

/// A single entry in an FSE table.
#[derive(Copy, Clone, Debug)]
pub struct Entry {
    /// This value is used as an offset value, and it is added
    /// to a value read from the stream to determine the next state value.
    pub base_line: u32,
    /// How many bits should be read from the stream when decoding this entry.
    pub num_bits: u8,
    /// The byte that should be put in the decode output when encountering this state.
    pub symbol: u8,
}

/// This value is added to the first 4 bits of the stream to determine the
/// `Accuracy_Log`
const ACC_LOG_OFFSET: u8 = 5;

fn highest_bit_set(x: u32) -> u32 {
    assert!(x > 0);
    u32::BITS - x.leading_zeros()
}

//utility functions for building the decoding table from probabilities
/// Calculate the position of the next entry of the table given the current
/// position and size of the table.
fn next_position(mut p: usize, table_size: usize) -> usize {
    p += (table_size >> 1) + (table_size >> 3) + 3;
    p &= table_size - 1;
    p
}

fn calc_baseline_and_numbits(
    num_states_total: u32,
    num_states_symbol: u32,
    state_number: u32,
) -> (u32, u8) {
    if num_states_symbol == 0 {
        return (0, 0);
    }
    let num_state_slices = if 1 << (highest_bit_set(num_states_symbol) - 1) == num_states_symbol {
        num_states_symbol
    } else {
        1 << (highest_bit_set(num_states_symbol))
    }; //always power of two

    let num_double_width_state_slices = num_state_slices - num_states_symbol; //leftovers to the power of two need to be distributed
    let num_single_width_state_slices = num_states_symbol - num_double_width_state_slices; //these will not receive a double width slice of states
    let slice_width = num_states_total / num_state_slices; //size of a single width slice of states
    let num_bits = highest_bit_set(slice_width) - 1; //number of bits needed to read for one slice

    if state_number < num_double_width_state_slices {
        let baseline = num_single_width_state_slices * slice_width + state_number * slice_width * 2;
        (baseline, num_bits as u8 + 1)
    } else {
        let index_shifted = state_number - num_double_width_state_slices;
        ((index_shifted * slice_width), num_bits as u8)
    }
}

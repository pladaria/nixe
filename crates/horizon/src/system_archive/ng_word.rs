//! Nixe-owned word lists compiled into the SDK's succinct Aho-Corasick format.
//! The archive paths and gzip envelope follow the generated resource documented by:
//! https://github.com/JamesDoesGaming/yuzu/blob/master/src/core/file_sys/system_archive/ng_word.cpp
//! The nonempty format was checked against the guest SDK's Import/Match routines.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use flate2::{Compression, write::GzEncoder};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dictionary {
    categories: BTreeMap<String, Category>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Category {
    words: Vec<String>,
}

pub(super) fn files(source: &str) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let dictionary: Dictionary = toml::from_str(source).map_err(|e| e.to_string())?;
    let mut words = BTreeSet::new();
    for (category, entry) in dictionary.categories {
        for word in entry.words {
            if word.is_empty() || !word.bytes().all(|c| c.is_ascii_lowercase()) {
                return Err(format!(
                    "dictionary category {category}: word {word:?} must contain lowercase ASCII letters"
                ));
            }
            words.insert(word.into_bytes());
        }
    }
    let common = gzip(&encode(&words)?)?;
    let empty = gzip(&encode(&BTreeSet::new())?)?;
    let mut files = BTreeMap::new();
    // Pattern-list bits identify 16 regional vocabularies. Our examples are common
    // to all regions; empty regional/boundary sets are intentional, not a bypass.
    for region in (0..16).map(|i| i.to_string()).chain(["common".into()]) {
        for category in ["b1", "b2", "not_b"] {
            let bytes = if region == "common" && category == "not_b" {
                &common
            } else {
                &empty
            };
            files.insert(format!("ac_{region}_{category}_nx"), bytes.clone());
        }
    }
    // NgWord2 uses a big-endian content revision, independent of little-endian tables.
    files.insert("version.dat".into(), 1_u32.to_be_bytes().to_vec());
    Ok(files)
}
fn gzip(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(bytes).map_err(|e| e.to_string())?;
    encoder.finish().map_err(|e| e.to_string())
}

#[derive(Default)]
struct Writer(Vec<u8>);
impl Writer {
    fn u32(&mut self, value: usize) -> Result<(), String> {
        self.0.extend(
            u32::try_from(value)
                .map_err(|_| "dictionary exceeds the SDK's 32-bit table format")?
                .to_le_bytes(),
        );
        Ok(())
    }
    fn u64(&mut self, value: usize) {
        self.0.extend((value as u64).to_le_bytes());
    }
    fn array(&mut self, values: &[usize]) -> Result<(), String> {
        self.u32(values.len())?;
        for &value in values {
            self.u32(value)?;
        }
        Ok(())
    }
    fn bit_vector(&mut self, bits: &[bool]) -> Result<(), String> {
        self.u32(bits.len())?;
        for chunk in bits.chunks(32) {
            let value = chunk
                .iter()
                .enumerate()
                .fold(0, |v, (i, bit)| v | (usize::from(*bit) << i));
            self.u32(value)?;
        }
        Ok(())
    }
    fn rank(&mut self, bits: &[bool]) -> Result<(), String> {
        if bits.is_empty() {
            return Ok(());
        }
        let mut total = 0;
        let mut block = 0;
        let mut relative = Vec::new();
        for i in (0..=bits.len()).step_by(32) {
            if i % 256 == 0 {
                self.u32(total)?;
                block = total;
            }
            relative.push((total - block) as u8);
            total += bits[i..(i + 32).min(bits.len())]
                .iter()
                .filter(|b| **b)
                .count();
        }
        self.0.extend(relative);
        Ok(())
    }
    fn set(&mut self, bits: &[bool]) -> Result<(), String> {
        self.bit_vector(bits)?;
        self.rank(bits)
    }
    fn select(&mut self, bits: &[bool]) -> Result<(), String> {
        // Two rank vectors map a set-bit index to its nonempty 32-bit word.
        let words: Vec<_> = bits
            .chunks(32)
            .enumerate()
            .filter_map(|(i, chunk)| {
                let count = chunk.iter().filter(|b| **b).count();
                (count != 0).then_some((i, count))
            })
            .collect();
        self.array(
            &words
                .iter()
                .enumerate()
                .map(|(j, (i, _))| i - j)
                .collect::<Vec<_>>(),
        )?;
        let flags = vec![true; words.len()];
        let mut unary = Vec::new();
        for (_, count) in words {
            unary.push(true);
            unary.extend(std::iter::repeat_n(false, count - 1));
        }
        self.bit_vector(&flags)?;
        self.bit_vector(&unary)?;
        self.rank(&flags)?;
        self.rank(&unary)
    }
    fn sparse(&mut self, values: &[usize], universe: usize) -> Result<(), String> {
        // Elias-Fano: packed low bits and a unary high-bit vector with rank/select.
        let ratio = universe / values.len().max(1);
        let width = (usize::BITS - ratio.max(1).leading_zeros() - 1).max(1) as usize;
        self.u64(universe);
        self.u64(values.first().copied().unwrap_or(u32::MAX as usize));
        self.u64(values.last().copied().unwrap_or(0));
        self.u32(values.len())?;
        self.u32(width)?;
        let mut low = vec![0_usize; (values.len() * width).div_ceil(32)];
        let mut high = vec![false; values.last().map_or(0, |v| (v >> width) + values.len())];
        for (i, &value) in values.iter().enumerate() {
            for j in 0..width {
                low[(i * width + j) / 32] |= ((value >> j) & 1) << ((i * width + j) % 32);
            }
            high[(value >> width) + i] = true;
        }
        self.array(&low)?;
        self.set(&high)?;
        self.select(&high)
    }
    fn compressed_array(&mut self, values: &[usize]) -> Result<(), String> {
        // 64-value blocks: signed fixed-width deltas from a base, followed by an
        // inline tail. Full-width fields are valid for every SDK integer value.
        let blocks = values.len() / 64;
        let mut metadata = Vec::new();
        for i in 0..blocks {
            let offset = i * 64;
            if offset >= 1 << 27 {
                return Err("dictionary exceeds the SDK's 27-bit compressed-array offsets".into());
            }
            metadata.extend([(31 << 27) | offset, 0]);
        }
        self.array(&metadata)?;
        self.array(&values[..blocks * 64])?;
        let tail = &values[blocks * 64..];
        self.0.push(tail.len() as u8);
        for &value in tail {
            self.u32(value)?;
        }
        Ok(())
    }
    fn parentheses(&mut self, bits: &[bool]) -> Result<(), String> {
        self.set(bits)?;
        let mut stack = Vec::new();
        let mut pairs = Vec::new();
        for (i, bit) in bits.iter().enumerate() {
            if *bit {
                stack.push(i);
            } else {
                let open = stack.pop().expect("balanced failure tree");
                if open / 32 != i / 32 {
                    pairs.push((open, i));
                }
            }
        }
        if pairs.is_empty() {
            self.0.push(0);
            return Ok(());
        }
        // Pioneer pairs connect blocks. Retain the first opening per opening
        // block and the last closing per closing block, then recurse on their
        // balanced subsequence. This supports enclosing nodes across word boundaries.
        let mut marks = BTreeSet::new();
        pairs.sort_unstable();
        let mut last = None;
        for &(open, close) in &pairs {
            if last != Some(open / 32) {
                marks.extend([open, close]);
                last = Some(open / 32);
            }
        }
        pairs.sort_unstable_by_key(|(_, close)| *close);
        last = None;
        for &(open, close) in pairs.iter().rev() {
            if last != Some(close / 32) {
                marks.extend([open, close]);
                last = Some(close / 32);
            }
        }
        self.0.push(1);
        let positions: Vec<_> = marks.into_iter().collect();
        self.sparse(&positions, bits.len())?;
        self.parentheses(&positions.iter().map(|i| bits[*i]).collect::<Vec<_>>())
    }
}

fn encode(words: &BTreeSet<Vec<u8>>) -> Result<Vec<u8>, String> {
    let mut prefixes = BTreeSet::from([Vec::new()]);
    for word in words {
        for i in 1..=word.len() {
            prefixes.insert(word[..i].to_vec());
        }
    }
    // Colexicographic state IDs make char-major transition ranks equal their
    // destination IDs, and order the failure tree in depth-first traversal order.
    let mut nodes: Vec<_> = prefixes.into_iter().collect();
    nodes.sort_unstable_by(|a, b| a.iter().rev().cmp(b.iter().rev()));
    let ids: BTreeMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.as_slice(), i))
        .collect();
    let n = nodes.len();
    let mut failure = vec![0; n];
    let mut children = vec![Vec::new(); n];
    let mut transitions = Vec::new();
    let mut terminals = Vec::new();
    let mut lengths = Vec::new();
    for (i, node) in nodes.iter().enumerate().skip(1) {
        failure[i] = (1..node.len())
            .find_map(|j| ids.get(&node[j..]).copied())
            .unwrap_or(0);
        children[failure[i]].push(i);
        transitions.push(usize::from(*node.last().unwrap()) * n + ids[&node[..node.len() - 1]]);
        if words.contains(node) {
            terminals.push(i);
            lengths.push(node.len());
        }
    }
    transitions.sort_unstable();
    let mut suffixes = Vec::new();
    let mut links = Vec::new();
    for i in 1..n {
        let mut parent = failure[i];
        while parent != 0 {
            if words.contains(&nodes[parent]) {
                suffixes.push(i);
                links.push(parent);
                break;
            }
            parent = failure[parent];
        }
    }
    let mut bits = Vec::new();
    // Iterative traversal avoids a host stack proportional to pattern length.
    let mut traversal = vec![(0, false)];
    while let Some((i, close)) = traversal.pop() {
        bits.push(!close);
        if !close {
            traversal.push((i, true));
            for &child in children[i].iter().rev() {
                traversal.push((child, false));
            }
        }
    }
    let mut writer = Writer::default();
    writer.compressed_array(&lengths)?;
    writer.sparse(&terminals, n)?;
    writer.compressed_array(&links)?;
    writer.sparse(&suffixes, n)?;
    writer.u32(n)?;
    writer.sparse(
        &transitions,
        n.checked_mul(256)
            .ok_or("dictionary transition universe overflows")?,
    )?;
    writer.parentheses(&bits)?;
    writer.select(&bits)?;
    Ok(writer.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    #[test]
    fn categories_contribute_to_one_real_common_filter_and_invalid_lists_fail() {
        let source = "[categories.insults]\nwords = [\"idiot\", \"idiot\"]\n[categories.spam]\nwords = [\"spam\"]";
        let archive = files(source).unwrap();
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(archive["ac_common_not_b_nx"].as_slice())
            .read_to_end(&mut raw)
            .unwrap();
        assert_eq!(
            raw,
            encode(&BTreeSet::from([b"idiot".to_vec(), b"spam".to_vec()])).unwrap()
        );
        assert_ne!(archive["ac_common_not_b_nx"], archive["ac_common_b1_nx"]);
        for invalid in [
            "[categories.a]\nwords = [\"\"]",
            "[categories.a]\nwords = [\"UPPER\"]",
            "[categories.a]\nwords = [\"café\"]",
            "[categories.a]\nwords = []\nunknown = 1",
        ] {
            assert!(files(invalid).is_err(), "{invalid}");
        }
    }
}

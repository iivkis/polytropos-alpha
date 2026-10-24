//! Deterministic character encoding and 90/10 train/validation split.
//! Source path is explicit; no network downloads, silent corpus replacement,
//! or evaluation on training tokens. Follows research handoff's sorted charset.
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct CharCorpus {
    pub alphabet: Vec<char>,
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
}
impl CharCorpus {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        Self::from_text(&std::fs::read_to_string(path)?)
    }
    pub fn from_text(text: &str) -> Result<Self, Box<dyn Error>> {
        let chars = text.chars().collect::<Vec<_>>();
        if chars.len() < 32 {
            return Err("corpus needs >=32 Unicode scalar values".into());
        }
        let alphabet = chars.iter().copied().collect::<std::collections::BTreeSet<_>>()
            .into_iter().collect::<Vec<_>>();
        if alphabet.len() < 2 || alphabet.len() > 4096 {
            return Err("charset length must be 2..=4096".into());
        }
        let ids = alphabet.iter().enumerate()
            .map(|(i,&c)|(c,i)).collect::<BTreeMap<_,_>>();
        let data = chars.iter().map(|c| ids[c]).collect::<Vec<_>>();
        // Match handoff: first 90% chronological character positions are
        // training; final 10% are holdout. No overlapping target position.
        let boundary = data.len()*9/10;
        if boundary < 2 || data.len()-boundary < 2 {
            return Err("corpus split has insufficient train/validation characters".into());
        }
        Ok(Self {
            alphabet,
            train: data[..boundary].to_vec(),
            validation: data[boundary..].to_vec(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_charset_and_disjoint_split() {
        let text = "заbβ🙂".repeat(16);
        let corpus=CharCorpus::from_text(&text).unwrap();
        assert_eq!(corpus.alphabet,vec!['b','β','а','з','🙂']);
        assert_eq!(corpus.train.len()+corpus.validation.len(),text.chars().count());
        assert_eq!(corpus.train.len(),text.chars().count()*9/10);
        assert_eq!(corpus.train[0],corpus.alphabet.iter().position(|&c|c=='з').unwrap());
        assert!(CharCorpus::from_text("x").is_err());
    }
}

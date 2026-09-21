//! Measure how honest a backend's probabilities are.
//!
//! Feed `(predicted probability, what happened)` pairs and get Brier score,
//! log loss, expected calibration error and a reliability table you can
//! print or plot. Works for `noul` directly; for `choice` / `score` pass
//! one-vs-rest pairs for the option you care about, or the probability
//! assigned to the realised option (see [`CalibrationReport::from_records`]).

use serde::{Deserialize, Serialize};

use crate::answer::RawAnswer;
use crate::record::Record;

/// One reliability bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
    /// Mean predicted probability in the bin.
    pub mean_p: f64,
    /// Empirical frequency of the positive outcome in the bin.
    pub frac_pos: f64,
}

/// Summary statistics of a set of probabilistic predictions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub n: usize,
    /// Mean squared error between probability and outcome. 0 = perfect,
    /// 0.25 = always saying 0.5.
    pub brier: f64,
    /// Mean negative log-likelihood (natural log). Lower is better.
    pub log_loss: f64,
    /// Expected calibration error: count-weighted mean |mean_p − frac_pos|.
    pub ece: f64,
    /// Accuracy at a 0.5 threshold.
    pub accuracy: f64,
    /// Base rate of positives.
    pub base_rate: f64,
    pub bins: Vec<Bin>,
}

impl CalibrationReport {
    /// Build from `(p, outcome)` pairs with `n_bins` equal-width bins.
    pub fn from_pairs<I: IntoIterator<Item = (f64, bool)>>(pairs: I, n_bins: usize) -> Self {
        let n_bins = n_bins.max(1);
        let pairs: Vec<(f64, bool)> = pairs.into_iter().map(|(p, y)| (p.clamp(0.0, 1.0), y)).collect();
        let n = pairs.len();
        if n == 0 {
            return Self { n: 0, brier: f64::NAN, log_loss: f64::NAN, ece: f64::NAN, accuracy: f64::NAN, base_rate: f64::NAN, bins: vec![] };
        }
        let eps = 1e-12;
        let mut brier = 0.0;
        let mut ll = 0.0;
        let mut correct = 0usize;
        let mut pos = 0usize;
        let mut sums = vec![(0usize, 0.0f64, 0usize); n_bins]; // (count, sum_p, sum_pos)
        for (p, y) in &pairs {
            let yf = if *y { 1.0 } else { 0.0 };
            brier += (p - yf).powi(2);
            ll -= if *y { (p + eps).ln() } else { (1.0 - p + eps).ln() };
            if (*p >= 0.5) == *y {
                correct += 1;
            }
            if *y {
                pos += 1;
            }
            let idx = ((p * n_bins as f64) as usize).min(n_bins - 1);
            sums[idx].0 += 1;
            sums[idx].1 += p;
            sums[idx].2 += usize::from(*y);
        }
        let mut bins = Vec::with_capacity(n_bins);
        let mut ece = 0.0;
        for (i, (c, sp, spos)) in sums.iter().enumerate() {
            let lo = i as f64 / n_bins as f64;
            let hi = (i + 1) as f64 / n_bins as f64;
            let (mean_p, frac_pos) = if *c > 0 { (sp / *c as f64, *spos as f64 / *c as f64) } else { (f64::NAN, f64::NAN) };
            if *c > 0 {
                ece += (*c as f64 / n as f64) * (mean_p - frac_pos).abs();
            }
            bins.push(Bin { lo, hi, count: *c, mean_p, frac_pos });
        }
        Self {
            n,
            brier: brier / n as f64,
            log_loss: ll / n as f64,
            ece,
            accuracy: correct as f64 / n as f64,
            base_rate: pos as f64 / n as f64,
            bins,
        }
    }

    /// Build from recorded decisions for one question.
    ///
    /// `outcome[question]` must be present in each record and be:
    /// * a bool for `noul`;
    /// * the realised option key (string) for `choice`;
    /// * the realised level index (integer) for `score`.
    ///
    /// For `choice`/`score` the pair is `(P(realised option), true)` plus
    /// `(P(other option), false)` for every other option, i.e. one-vs-rest
    /// over all options — the standard multiclass reliability view.
    pub fn from_records<'a, I: IntoIterator<Item = &'a Record>>(records: I, question: &str, n_bins: usize) -> Self {
        let mut pairs = Vec::new();
        for r in records {
            let Some(outcome) = r.outcome.as_ref().and_then(|o| o.get(question)) else { continue };
            let Some(ans) = r.answers.answers.get(question) else { continue };
            match ans {
                RawAnswer::Noul { noul, .. } => {
                    if let Some(y) = outcome.as_bool() {
                        pairs.push((*noul, y));
                    }
                }
                RawAnswer::Choice { probabilities, .. } => {
                    if let Some(key) = outcome.as_str() {
                        for (k, p) in probabilities {
                            pairs.push((*p, k == key));
                        }
                    }
                }
                RawAnswer::Score { probabilities, .. } => {
                    if let Some(level) = outcome.as_u64() {
                        for (k, p) in probabilities {
                            pairs.push((*p, k == &level.to_string()));
                        }
                    }
                }
            }
        }
        Self::from_pairs(pairs, n_bins)
    }

    /// A compact text rendering — reliability table plus headline numbers.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "n={}  brier={:.4}  log_loss={:.4}  ece={:.4}  acc@0.5={:.3}  base_rate={:.3}\n",
            self.n, self.brier, self.log_loss, self.ece, self.accuracy, self.base_rate
        ));
        s.push_str("  bin          count   mean_p   frac_pos   |gap|\n");
        for b in &self.bins {
            if b.count == 0 {
                s.push_str(&format!("  [{:.2},{:.2})      0        -          -       -\n", b.lo, b.hi));
            } else {
                let bar = "#".repeat(((b.frac_pos * 20.0).round() as usize).min(20));
                s.push_str(&format!(
                    "  [{:.2},{:.2}) {:6}   {:.3}    {:.3}   {:.3}  {}\n",
                    b.lo, b.hi, b.count, b.mean_p, b.frac_pos, (b.mean_p - b.frac_pos).abs(), bar
                ));
            }
        }
        s
    }
}

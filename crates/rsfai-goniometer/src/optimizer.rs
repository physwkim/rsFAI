//! The iterative-fit layer — ISOLATED from the bit-exact transformation / chi²
//! core. This module is the ONLY place `argmin` is touched.
//!
//! pyFAI refines the goniometer parameter vector with `scipy.optimize.minimize`
//! (SLSQP, `GoniometerRefinement.refine2`); here a Nelder-Mead simplex (`argmin`)
//! minimizes the cost the core computes. The optimizer takes a plain
//! `Fn(&[f64]) -> f64` cost closure (which internally calls the bit-exact
//! [`crate::refinement::GoniometerRefinement::residu2`]) and returns the
//! converged parameter vector plus its cost. It never sees the formula evaluator
//! or the geometry math directly, so the converged-parameter *tolerance* gate and
//! the residual/chi² *bit-exact* gate stay structurally separate.
//!
//! The converged parameters are explicitly NOT bit-exact vs pyFAI: the
//! Nelder-Mead trajectory differs from scipy's SLSQP by construction (different
//! step rules, different libm). The verifier asserts a recorded relative
//! tolerance on the parameters and `cost_rust <= cost_pyfai`.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

/// Adapter wrapping a cost closure as an `argmin::CostFunction` over `Vec<f64>`
/// (Nelder-Mead's parameter type), so the bit-exact core needs no `argmin`
/// dependency.
struct CostAdapter<F> {
    f: F,
}

impl<F> CostFunction for CostAdapter<F>
where
    F: Fn(&[f64]) -> f64,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok((self.f)(param))
    }
}

/// Standard-deviation tolerance for the simplex (Nelder-Mead stops when the
/// spread of vertex costs falls below this), matching the tight setting used by
/// the calibration refinement.
const SD_TOLERANCE: f64 = 1e-18;

/// Maximum simplex iterations — generous, mirroring pyFAI's large `maxiter`; the
/// well-conditioned goniometer fit converges in far fewer.
const MAX_ITERS: u64 = 100_000;

/// scipy `fmin` initial-simplex steps: 0.05 relative per coordinate, with a
/// 0.00025 absolute floor for a near-zero coordinate.
const REL_STEP: f64 = 0.05;
const ABS_STEP: f64 = 0.00025;

/// Minimize `cost` from `start` with Nelder-Mead over the full parameter vector,
/// returning `(best_param, best_cost)`.
///
/// The initial simplex is `start` plus one vertex per coordinate, each perturbed
/// by a scale-aware step — scipy's default `fmin` construction. Weakly-constrained
/// directions (cost null directions) settle anywhere along the flat valley floor,
/// which is why the converged-parameter comparison is gated by a recorded
/// tolerance with the cost asserted `<=` pyFAI's, not by a bit-exact value.
pub(crate) fn minimize<F>(start: &[f64], cost: F) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = start.len();
    assert!(n > 0, "at least one parameter is required");

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(start.to_vec());
    for k in 0..n {
        let mut v = start.to_vec();
        if v[k] != 0.0 {
            v[k] += v[k].abs() * REL_STEP;
        } else {
            v[k] = ABS_STEP;
        }
        simplex.push(v);
    }

    let problem = CostAdapter { f: cost };
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(SD_TOLERANCE)
        .expect("sd_tolerance is non-negative");

    let result = Executor::new(problem, solver)
        .configure(|state| state.max_iters(MAX_ITERS))
        .run()
        .expect("Nelder-Mead minimization");

    let state = result.state();
    let best_cost = state.get_best_cost();
    let best = state
        .get_best_param()
        .expect("Nelder-Mead produced a best parameter")
        .clone();
    (best, best_cost)
}

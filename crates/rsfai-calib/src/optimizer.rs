//! The iterative-fit layer — ISOLATED from the bit-exact residual/chi2 core.
//!
//! pyFAI refines the geometry with `scipy.optimize` (`leastsq` / `fmin_slsqp` /
//! `curve_fit`); here a Nelder-Mead simplex (`argmin`) minimizes the chi2 the
//! core computes. This module is the ONLY place `argmin` is touched: it takes a
//! plain `Fn(&[f64; 6]) -> f64` cost closure (which internally calls the
//! bit-exact `GeometryRefinement::chi2`) and returns the converged parameter
//! vector plus its cost. The optimizer never sees the geometry/calibrant math
//! directly, so the converged-parameter *tolerance* gate and the residual/chi2
//! *bit-exact* gate stay structurally separate.
//!
//! The converged parameters are explicitly NOT bit-exact vs pyFAI: the
//! Nelder-Mead trajectory differs from scipy's SLSQP by construction (different
//! step rules, different libm). The verifier asserts a recorded relative
//! tolerance on the parameters and `cost_rust <= cost_pyfai`.

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

/// Adapter wrapping a cost closure as an `argmin::CostFunction` over `Vec<f64>`
/// (Nelder-Mead's parameter type). Nelder-Mead varies only the FREE coordinates
/// (`free` is their indices into the full 6-vector); the fixed coordinates are
/// held at `base`. This mirrors pyFAI's `refine3` free/const split
/// (`geometryRefinement.py:531-553`): the optimizer never sees the fixed
/// coordinates, so a fixed `rot3` (a null direction of this dataset's cost
/// surface) cannot wander.
struct ChiCost<F> {
    f: F,
    base: [f64; 6],
    free: Vec<usize>,
}

impl<F> ChiCost<F> {
    /// Expand a free-subset Nelder-Mead vector back to the full 6-parameter
    /// vector (fixed coordinates from `base`).
    fn expand(&self, param: &[f64]) -> [f64; 6] {
        let mut p = self.base;
        for (k, &idx) in self.free.iter().enumerate() {
            p[idx] = param[k];
        }
        p
    }
}

impl<F> CostFunction for ChiCost<F>
where
    F: Fn(&[f64; 6]) -> f64,
{
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok((self.f)(&self.expand(param)))
    }
}

/// Standard-deviation tolerance for the simplex (Nelder-Mead stops when the
/// spread of vertex costs falls below this). Tight enough that the converged
/// geometry matches scipy's to the recorded relative tolerance.
const SD_TOLERANCE: f64 = 1e-18;

/// Maximum simplex iterations — generous, mirroring pyFAI's `maxiter=1e7`; the
/// well-conditioned ring-fit converges in far fewer.
const MAX_ITERS: u64 = 100_000;

/// scipy `fmin` initial-simplex steps: 0.05 relative per coordinate, with a
/// 0.00025 absolute floor for a near-zero coordinate.
const REL_STEP: f64 = 0.05;
const ABS_STEP: f64 = 0.00025;

/// Minimize `cost` from `start` with Nelder-Mead over the FREE coordinates
/// (`free` = their indices into the 6-vector; the rest stay pinned at `start`),
/// returning `(best_full_param, best_cost)`.
///
/// The initial simplex is the free subvector plus one vertex per free
/// coordinate, each perturbed by a scale-aware step — scipy's default `fmin`
/// construction. When every coordinate is free, weakly-constrained directions
/// (a near-zero `rot3` whose data leverage is ~0) settle anywhere along the flat
/// valley floor, which is why the all-free converged-parameter comparison is
/// gated by cost, not per-parameter value; fixing such a direction makes the
/// remaining minimum unique.
pub(crate) fn minimize_chi2<F>(start: &[f64; 6], free: &[usize], cost: F) -> ([f64; 6], f64)
where
    F: Fn(&[f64; 6]) -> f64,
{
    assert!(!free.is_empty(), "at least one free parameter is required");
    let nfree = free.len();

    // Free-subvector simplex (nfree + 1 vertices).
    let start_free: Vec<f64> = free.iter().map(|&i| start[i]).collect();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(nfree + 1);
    simplex.push(start_free.clone());
    for k in 0..nfree {
        let mut v = start_free.clone();
        if v[k] != 0.0 {
            v[k] += v[k].abs() * REL_STEP;
        } else {
            v[k] = ABS_STEP;
        }
        simplex.push(v);
    }

    let problem = ChiCost {
        f: cost,
        base: *start,
        free: free.to_vec(),
    };
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
        .expect("Nelder-Mead produced a best parameter");
    // Expand the free subvector back to the full 6-parameter vector.
    let mut out = *start;
    for (k, &idx) in free.iter().enumerate() {
        out[idx] = best[k];
    }
    (out, best_cost)
}

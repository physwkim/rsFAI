//! `GeometryTransformation`, ported from `pyFAI/goniometer.py`.
//!
//! A `GeometryTransformation` holds one formula string per PONI component
//! (`dist`, `poni1`, `poni2`, `rot1`, `rot2`, `rot3`), the ordered parameter
//! names (`param_names`), the ordered motor-position names (`pos_names`), and an
//! optional table of user constants. Calling it with a parameter vector and a
//! goniometer position binds each name to its value (plus `pi` and the
//! constants) and evaluates the six formulas to the six PONI scalars
//! (`PoniParam`).
//!
//! The binding order mirrors pyFAI exactly (`GeometryTransformation.__call__`):
//! the constants and `pi` seed the variable table, the `param_names` are bound
//! positionally from `param`, then the `pos_names` from `pos` (a single scalar
//! when there is one motor, else positional). Each formula is evaluated in
//! IEEE-754 f64 by [`crate::expr`], which is bit-exact to numexpr (see that
//! module's docs).
//!
//! pyFAI assembles the result as `PoniParam(**res)` where `res` carries only the
//! components whose formula was supplied. Here a supplied formula yields its
//! value and an absent one yields `0.0` — the same value pyFAI's downstream
//! `residu*` reads via `single_param.get(name, 0.0)`. (`get_ai` requires all six,
//! which a real goniometer config always provides.)

use std::collections::BTreeMap;

use crate::expr::{ExprError, Formula};

/// The six PONI scalars a transformation produces, in pyFAI's `PoniParam` order.
/// Lengths are SI (metres for `dist`/`poni1`/`poni2`, radians for the rotations).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoniParam {
    /// Sample–detector distance `dist` (m).
    pub dist: f64,
    /// PONI along the slow (Y) axis `poni1` (m).
    pub poni1: f64,
    /// PONI along the fast (X) axis `poni2` (m).
    pub poni2: f64,
    /// Rotation about axis 1 `rot1` (rad).
    pub rot1: f64,
    /// Rotation about axis 2 `rot2` (rad).
    pub rot2: f64,
    /// Rotation about axis 3 `rot3` (rad).
    pub rot3: f64,
}

impl PoniParam {
    /// The six components in `PoniParam` order, for indexed iteration.
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3,
        ]
    }
}

/// A transformation evaluation failure: a malformed formula or an unbound name.
pub type TransformError = ExprError;

/// The goniometer geometry transformation: six optional PONI-component formulas
/// plus the parameter / position name bindings and the user constants.
///
/// `GeometryTranslation` in pyFAI is an alias of `GeometryTransformation` (same
/// class); there is no behavioural difference, so it is not a separate type here.
/// `ExtendedTransformation` (which adds a `wavelength` formula) is not ported —
/// no in-scope config or golden needs the per-motor wavelength path.
#[derive(Debug, Clone)]
pub struct GeometryTransformation {
    dist: Option<Formula>,
    poni1: Option<Formula>,
    poni2: Option<Formula>,
    rot1: Option<Formula>,
    rot2: Option<Formula>,
    rot3: Option<Formula>,
    param_names: Vec<String>,
    pos_names: Vec<String>,
    constants: BTreeMap<String, f64>,
}

impl GeometryTransformation {
    /// Build a transformation from the six formula strings (each optional, `None`
    /// for a component the config omits), the ordered parameter names, the ordered
    /// position names, and the user constants.
    ///
    /// pyFAI defaults `pos_names` to `("pos",)` (a single motor) when not given;
    /// pass `None` for that behaviour. Constants are bound under their names in
    /// addition to `pi`; a name collision between a constant and a param/pos name
    /// is rejected (pyFAI raises `RuntimeError`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_expr: Option<&str>,
        poni1_expr: Option<&str>,
        poni2_expr: Option<&str>,
        rot1_expr: Option<&str>,
        rot2_expr: Option<&str>,
        rot3_expr: Option<&str>,
        param_names: &[&str],
        pos_names: Option<&[&str]>,
        constants: &[(&str, f64)],
    ) -> Result<GeometryTransformation, TransformError> {
        let parse_opt = |e: Option<&str>| -> Result<Option<Formula>, TransformError> {
            match e {
                Some(s) => Ok(Some(Formula::parse(s)?)),
                None => Ok(None),
            }
        };
        let param_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        let pos_names: Vec<String> = match pos_names {
            Some(p) => p.iter().map(|s| s.to_string()).collect(),
            None => vec!["pos".to_string()],
        };
        let mut const_map: BTreeMap<String, f64> = BTreeMap::new();
        for (k, v) in constants {
            const_map.insert((*k).to_string(), *v);
        }
        // pyFAI rejects a param/pos name that collides with a reserved binding
        // (`pi`, a constant). Mirror that as a parse-time error.
        for name in param_names.iter().chain(pos_names.iter()) {
            if name == "pi" || const_map.contains_key(name) {
                return Err(ExprError::Parse(format!(
                    "the keyword `{name}` is already defined, choose another variable name"
                )));
            }
        }
        Ok(GeometryTransformation {
            dist: parse_opt(dist_expr)?,
            poni1: parse_opt(poni1_expr)?,
            poni2: parse_opt(poni2_expr)?,
            rot1: parse_opt(rot1_expr)?,
            rot2: parse_opt(rot2_expr)?,
            rot3: parse_opt(rot3_expr)?,
            param_names,
            pos_names,
            constants: const_map,
        })
    }

    /// The ordered parameter names.
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// The ordered position (motor) names.
    pub fn pos_names(&self) -> &[String] {
        &self.pos_names
    }

    /// Evaluate the transformation at parameter vector `param` and goniometer
    /// position `pos`, returning the six PONI scalars. `param.len()` must match
    /// the parameter-name count; `pos.len()` must match the position-name count
    /// (a single-motor config takes a one-element slice). A formula referencing an
    /// unbound name is a [`TransformError`].
    ///
    /// This is `GeometryTransformation.__call__`: bind constants + `pi`, then the
    /// params by name, then the positions by name, then evaluate each present
    /// formula. Absent components yield `0.0`.
    pub fn call(&self, param: &[f64], pos: &[f64]) -> Result<PoniParam, TransformError> {
        assert_eq!(
            param.len(),
            self.param_names.len(),
            "param length {} != param_names {}",
            param.len(),
            self.param_names.len()
        );
        assert_eq!(
            pos.len(),
            self.pos_names.len(),
            "pos length {} != pos_names {}",
            pos.len(),
            self.pos_names.len()
        );

        // Variable binding, in pyFAI's order: constants, pi, params, positions.
        let mut vars: BTreeMap<String, f64> = self.constants.clone();
        vars.insert("pi".to_string(), std::f64::consts::PI);
        for (name, value) in self.param_names.iter().zip(param.iter()) {
            vars.insert(name.clone(), *value);
        }
        for (name, value) in self.pos_names.iter().zip(pos.iter()) {
            vars.insert(name.clone(), *value);
        }

        let eval = |f: &Option<Formula>| -> Result<f64, TransformError> {
            match f {
                Some(formula) => formula.eval(&vars),
                None => Ok(0.0),
            }
        };

        Ok(PoniParam {
            dist: eval(&self.dist)?,
            poni1: eval(&self.poni1)?,
            poni2: eval(&self.poni2)?,
            rot1: eval(&self.rot1)?,
            rot2: eval(&self.rot2)?,
            rot3: eval(&self.rot3)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_motor_rot2_affine() {
        // dist/poni constant; rot2 = scale*pos + offset. One param scale, one
        // const offset, one motor.
        let t = GeometryTransformation::new(
            Some("dist"),
            Some("poni1"),
            Some("poni2"),
            Some("0.0"),
            Some("scale * pos + offset"),
            Some("0.0"),
            &["dist", "poni1", "poni2", "scale"],
            None,
            &[("offset", 0.05)],
        )
        .unwrap();
        let p = t.call(&[0.2, 0.1, 0.11, 0.001], &[2.5]).unwrap();
        assert_eq!(p.dist, 0.2);
        assert_eq!(p.poni1, 0.1);
        assert_eq!(p.poni2, 0.11);
        assert_eq!(p.rot1, 0.0);
        assert_eq!(p.rot2, 0.001 * 2.5 + 0.05);
        assert_eq!(p.rot3, 0.0);
    }

    #[test]
    fn two_motor_binding() {
        let t = GeometryTransformation::new(
            Some("d"),
            Some("0"),
            Some("0"),
            Some("0"),
            Some("a * m1"),
            Some("b * m2"),
            &["d", "a", "b"],
            Some(&["m1", "m2"]),
            &[],
        )
        .unwrap();
        let p = t.call(&[0.3, 0.01, 0.02], &[1.5, -2.0]).unwrap();
        assert_eq!(p.dist, 0.3);
        assert_eq!(p.rot2, 0.01 * 1.5);
        assert_eq!(p.rot3, 0.02 * -2.0);
    }

    #[test]
    fn constant_name_collision_rejected() {
        let err = GeometryTransformation::new(
            Some("pi"),
            None,
            None,
            None,
            None,
            None,
            &["pi"],
            None,
            &[],
        );
        assert!(err.is_err());
    }
}

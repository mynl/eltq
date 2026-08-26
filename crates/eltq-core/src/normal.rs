//! Standard normal cdf and quantile.
//!
//! The cdf goes through `libm::erfc` (the fdlibm port, below one ulp; the
//! statrs erfc loses about 1e-11 relative in the tails, which the edge-score
//! round trip exposed). An own branch-free vectorizable erf is the 1.11.0
//! kernel item. The quantile is Wichura's AS241 (PPND16), a rational
//! approximation accurate to about 1e-16 relative over the open unit
//! interval, used once per half-bucket edge when building a unit's edge
//! scores.

use std::f64::consts::SQRT_2;

use libm::erfc;

/// Phi(x), with the infinities mapping to exactly 0 and 1.
#[inline]
pub fn cdf(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        0.0
    } else if x == f64::INFINITY {
        1.0
    } else {
        0.5 * erfc(-x / SQRT_2)
    }
}

/// Phi^{-1}(p) for p in [0, 1] (AS241); the endpoints map to the infinities.
pub fn quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * poly(&A, r) / poly(&B, r);
    }
    let r = (-(p.min(1.0 - p)).ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        poly(&C, r) / poly(&D, r)
    } else {
        let r = r - 5.0;
        poly(&E, r) / poly(&F, r)
    };
    if q < 0.0 { -value } else { value }
}

/// Horner evaluation with coefficients in ascending order.
#[inline]
fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

const A: [f64; 8] = [
    3.387_132_872_796_366_608e0,
    1.331_416_678_917_843_774_5e2,
    1.971_590_950_306_551_442_7e3,
    1.373_169_376_550_946_112_5e4,
    4.592_195_393_154_987_145_7e4,
    6.726_577_092_700_870_085_3e4,
    3.343_057_558_358_812_810_5e4,
    2.509_080_928_730_122_672_7e3,
];
const B: [f64; 8] = [
    1.0,
    4.231_333_070_160_091_125_2e1,
    6.871_870_074_920_579_083e2,
    5.394_196_021_424_751_107_7e3,
    2.121_379_430_158_659_586_7e4,
    3.930_789_580_009_271_061e4,
    2.872_908_573_572_194_267_4e4,
    5.226_495_278_852_854_561e3,
];
const C: [f64; 8] = [
    1.423_437_110_749_683_577_34e0,
    4.630_337_846_156_545_295_9e0,
    5.769_497_221_460_691_405_5e0,
    3.647_848_324_763_204_605_04e0,
    1.270_458_252_452_368_382_58e0,
    2.417_807_251_774_506_117_7e-1,
    2.272_384_498_926_918_458_33e-2,
    7.745_450_142_783_414_076_4e-4,
];
const D: [f64; 8] = [
    1.0,
    2.053_191_626_637_758_821_87e0,
    1.676_384_830_183_803_849_4e0,
    6.897_673_349_851_000_045_5e-1,
    1.481_039_764_274_800_745_9e-1,
    1.519_866_656_361_645_719_66e-2,
    5.475_938_084_995_344_946e-4,
    1.050_750_071_644_416_843_24e-9,
];
const E: [f64; 8] = [
    6.657_904_643_501_103_777_2e0,
    5.463_784_911_164_114_369_9e0,
    1.784_826_539_917_291_335_8e0,
    2.965_605_718_285_048_912_3e-1,
    2.653_218_952_657_612_309_3e-2,
    1.242_660_947_388_078_438_6e-3,
    2.711_555_568_743_487_578_15e-5,
    2.010_334_399_292_288_132_65e-7,
];
const F: [f64; 8] = [
    1.0,
    5.998_322_065_558_879_376_9e-1,
    1.369_298_809_227_358_053_1e-1,
    1.487_536_129_085_061_485_25e-2,
    7.868_691_311_456_132_591e-4,
    1.846_318_317_510_054_681_8e-5,
    1.421_511_758_316_445_888_7e-7,
    2.044_263_103_389_939_785_64e-15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_quantiles() {
        assert_eq!(quantile(0.5), 0.0);
        assert!((quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-14);
        assert!((quantile(0.025) + 1.959_963_984_540_054).abs() < 1e-14);
        assert!((quantile(1e-10) + 6.361_340_902_404_056).abs() < 1e-13);
        // 1 - 1e-10 is not representable exactly (relative slip 8e-8 in the
        // tail probability, ~1e-8 in z), so only a loose upper-tail check.
        assert!((quantile(1.0 - 1e-10) - 6.361_340_902_404_056).abs() < 1e-7);
        assert_eq!(quantile(0.25), -quantile(0.75));
        assert_eq!(quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile(1.0), f64::INFINITY);
    }

    #[test]
    fn cdf_quantile_round_trip() {
        // Lower tail keeps full relative precision in p; the upper tail is
        // limited by 1 - p losing digits (error ~ ulp(1) / phi(x)), which is
        // inherent to any (cdf, quantile) pair, the reference included.
        for i in -60..=60 {
            let x = i as f64 * 0.1;
            let p = cdf(x);
            let back = quantile(p);
            let tolerance = if x <= 0.0 { 1e-13 } else { 4.0 * f64::EPSILON / (0.5 * erfc(x / SQRT_2)).max(1e-300) * 0.4 + 1e-13 };
            assert!((back - x).abs() <= tolerance * x.abs().max(1.0), "x={x} back={back}");
        }
        assert_eq!(cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(cdf(f64::INFINITY), 1.0);
        assert!((cdf(0.0) - 0.5).abs() < 1e-16);
        assert!((cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-15);
    }
}

//! Standard normal cdf and quantile.
//!
//! The cdf goes through an own `erfc` (Cody's rational Chebyshev
//! approximations, the CALERF ranges: |x| <= 0.46875, 0.46875 < |x| <= 4,
//! |x| > 4), a straight polynomial pipeline per range that the compiler can
//! vectorize across a row of edge scores; it is held to `libm::erfc` (the
//! fdlibm port, below one ulp) at 1e-15 by the tests. The quantile is
//! Wichura's AS241 (PPND16), a rational approximation accurate to about
//! 1e-16 relative over the open unit interval, used once per half-bucket
//! edge when building a unit's edge scores.

use std::f64::consts::SQRT_2;

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

// Cody (1969) coefficients as in SPECFUN's CALERF.
const ERF_A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_56e2,
    3.774_852_376_853_020_21e2,
    3.209_377_589_138_469_47e3,
    1.857_777_061_846_031_53e-1,
];
const ERF_B: [f64; 4] = [
    2.360_129_095_234_412_09e1,
    2.440_246_379_344_441_73e2,
    1.282_616_526_077_372_28e3,
    2.844_236_833_439_170_62e3,
];
const ERFC_C: [f64; 9] = [
    5.641_884_969_886_700_89e-1,
    8.883_149_794_388_375_94e0,
    6.611_919_063_714_162_95e1,
    2.986_351_381_974_001_31e2,
    8.819_522_212_417_690_9e2,
    1.712_047_612_634_070_58e3,
    2.051_078_377_826_071_47e3,
    1.230_339_354_797_997_25e3,
    2.153_115_354_744_038_46e-8,
];
const ERFC_D: [f64; 8] = [
    1.574_492_611_070_983_47e1,
    1.176_939_508_913_124_99e2,
    5.371_811_018_620_098_58e2,
    1.621_389_574_566_690_19e3,
    3.290_799_235_733_459_63e3,
    4.362_619_090_143_247_16e3,
    3.439_367_674_143_721_64e3,
    1.230_339_354_803_749_42e3,
];
const ERFC_P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
const ERFC_Q: [f64; 5] = [
    2.568_520_192_289_822_42e0,
    1.872_952_849_923_467_25e0,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];
const INV_SQRT_PI: f64 = 5.641_895_835_477_562_87e-1;
/// Beyond this erfc underflows to zero in f64 (the last subnormals sit
/// around 27.2).
const ERFC_X_MAX: f64 = 28.0;

/// Complementary error function.
#[inline]
pub fn erfc(x: f64) -> f64 {
    let y = x.abs();
    if y <= 0.46875 {
        let ysq = if y > 1.11e-16 { y * y } else { 0.0 };
        let mut num = ERF_A[4] * ysq;
        let mut den = ysq;
        for i in 0..3 {
            num = (num + ERF_A[i]) * ysq;
            den = (den + ERF_B[i]) * ysq;
        }
        return 1.0 - x * (num + ERF_A[3]) / (den + ERF_B[3]);
    }
    let result = if y <= 4.0 {
        let mut num = ERFC_C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + ERFC_C[i]) * y;
            den = (den + ERFC_D[i]) * y;
        }
        let ratio = (num + ERFC_C[7]) / (den + ERFC_D[7]);
        exp_neg_square(y) * ratio
    } else if y >= ERFC_X_MAX {
        0.0
    } else {
        let ysq = 1.0 / (y * y);
        let mut num = ERFC_P[5] * ysq;
        let mut den = ysq;
        for i in 0..4 {
            num = (num + ERFC_P[i]) * ysq;
            den = (den + ERFC_Q[i]) * ysq;
        }
        let tail = ysq * (num + ERFC_P[4]) / (den + ERFC_Q[4]);
        exp_neg_square(y) * (INV_SQRT_PI - tail) / y
    };
    if x < 0.0 { 2.0 - result } else { result }
}

/// exp(-y^2) evaluated as exp(-ysq^2) exp(-(y - ysq)(y + ysq)) with ysq the
/// 1/16 truncation of y, which keeps the argument rounding error small.
#[inline]
fn exp_neg_square(y: f64) -> f64 {
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp()
}

/// Error function.
#[inline]
pub fn erf(x: f64) -> f64 {
    if x.abs() <= 0.46875 {
        1.0 - erfc(x)
    } else if x > 0.0 {
        1.0 - erfc(x)
    } else {
        erfc(-x) - 1.0
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
    fn erfc_matches_libm_to_1e15() {
        // Dense sweep over every range including the switch points; the
        // fdlibm port is the oracle (below one ulp).
        // Subnormal results (x beyond ~26.5) carry no relative precision in
        // either implementation and are compared absolutely.
        let check = |x: f64| {
            let (ours, want) = (erfc(x), libm::erfc(x));
            if want < 1e-300 {
                assert!((ours - want).abs() <= 1e-300, "x={x}: ours {ours} libm {want}");
                return 0.0;
            }
            let rel = ((ours - want) / want).abs();
            assert!(rel <= 1e-15, "x={x}: ours {ours} libm {want} rel {rel}");
            rel
        };
        let mut worst = 0.0_f64;
        let mut x = -6.0;
        while x <= 29.0 {
            worst = worst.max(check(x));
            x += 0.0137;
        }
        for x in [0.0, 0.46875, 0.46875 + 1e-12, 4.0, 4.0 + 1e-12, 26.6, 30.0, -0.5, -4.5, 1e-17] {
            check(x);
        }
        eprintln!("erfc worst relative error {worst:.2e}");
        assert!((erf(0.3) - libm::erf(0.3)).abs() < 1e-16);
        assert!((erf(-2.0) - libm::erf(-2.0)).abs() < 1e-16);
    }

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

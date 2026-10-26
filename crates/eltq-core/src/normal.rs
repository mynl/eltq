//! Standard normal cdf and quantile.
//!
//! The cdf goes through an own `erfc` (Cody's rational Chebyshev
//! approximations, the CALERF ranges: |x| <= 0.46875, 0.46875 < |x| <= 4,
//! |x| > 4), a straight polynomial pipeline per range that the compiler can
//! vectorize across a row of edge scores; it is held to `libm::erfc` (the
//! fdlibm port, below one ulp) at 1e-15 by the tests. The quantile is
//! Wichura's AS241 (PPND16), a rational approximation accurate to about
//! 1e-16 relative over the open unit interval, used once per half-bucket
//! edge when building a unit's edge scores. The bivariate cdf
//! (`bivariate_cdf`, 2.6.0) is Genz's `bvnu`: Drezner and Wesolowsky's
//! integral over the correlation by Gauss-Legendre, with the transformed
//! integrand of Genz 2004 ("Numerical computation of rectangular bivariate
//! and trivariate normal and t probabilities", Statistics and Computing 14)
//! for |rho| >= 0.925; absolute accuracy about 5e-16, held against the
//! reference's Owen's T form by the tests.

use std::f64::consts::{PI, SQRT_2};

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

/// `Phi_2(h, k; rho) = P(X <= h, Y <= k)` for standard normals with
/// correlation `rho` in [-1, 1]. The infinities and `rho = +-1` are exact;
/// otherwise Genz's `bvnu` on the complementary event (module notes).
pub fn bivariate_cdf(h: f64, k: f64, rho: f64) -> f64 {
    bivariate_upper(-h, -k, rho)
}

// Gauss-Legendre nodes and weights on [-1, 1] by |rho|: 6 points below 0.3,
// 12 below 0.75, 20 otherwise (Genz 2004); the symmetric pairs are formed
// in the loops.
const GL6_X: [f64; 3] = [0.932_469_514_203_152_2, 0.661_209_386_466_264_7, 0.238_619_186_083_197_0];
const GL6_W: [f64; 3] = [0.171_324_492_379_170_5, 0.360_761_573_048_138_4, 0.467_913_934_572_690_4];
const GL12_X: [f64; 6] = [
    0.981_560_634_246_719_1,
    0.904_117_256_370_475_0,
    0.769_902_674_194_305_0,
    0.587_317_954_286_617_1,
    0.367_831_498_998_180_2,
    0.125_233_408_511_469_2,
];
const GL12_W: [f64; 6] = [
    0.047_175_336_386_511_77,
    0.106_939_325_995_318_3,
    0.160_078_328_543_346_4,
    0.203_167_426_723_065_9,
    0.233_492_536_538_354_7,
    0.249_147_045_813_402_9,
];
const GL20_X: [f64; 10] = [
    0.993_128_599_185_094_9,
    0.963_971_927_277_913_8,
    0.912_234_428_251_325_9,
    0.839_116_971_822_218_8,
    0.746_331_906_460_150_8,
    0.636_053_680_726_515_0,
    0.510_867_001_950_827_1,
    0.373_706_088_715_419_6,
    0.227_785_851_141_645_1,
    0.076_526_521_133_497_33,
];
const GL20_W: [f64; 10] = [
    0.017_614_007_139_152_12,
    0.040_601_429_800_386_94,
    0.062_672_048_334_109_06,
    0.083_276_741_576_704_75,
    0.101_930_119_817_240_4,
    0.118_194_531_961_518_4,
    0.131_688_638_449_176_6,
    0.142_096_109_318_382_1,
    0.149_172_986_472_603_7,
    0.152_753_387_130_725_9,
];

/// `P(X > dh, Y > dk)`, Genz's `bvnu`: the upper-tail probability computed
/// directly, so small values keep their relative accuracy (the cdf's
/// complement would not).
pub fn bivariate_upper(dh: f64, dk: f64, r: f64) -> f64 {
    BivariateUpper::new(r).eval(dh, dk)
}

/// Genz's `bvnu` at a fixed correlation, with everything that depends on
/// the correlation alone (the Gauss-Legendre nodes transformed through the
/// arcsine or the square root) computed once: a unit's stratum table
/// evaluates it at every edge and stratum boundary with one correlation.
#[derive(Debug, Clone)]
pub struct BivariateUpper {
    r: f64,
    /// Per node and sign: `(sn, 1 / (1 - sn^2), w)` for |r| < 0.925.
    arcs: Vec<(f64, f64, f64)>,
    /// Per node and sign: `(xs, rs, w)` for |r| >= 0.925.
    roots: Vec<(f64, f64, f64)>,
    asr: f64,
    a_sq: f64,
    a_half: f64,
}

impl BivariateUpper {
    pub fn new(r: f64) -> Self {
        let (xs, ws): (&[f64], &[f64]) = if r.abs() < 0.3 {
            (&GL6_X, &GL6_W)
        } else if r.abs() < 0.75 {
            (&GL12_X, &GL12_W)
        } else {
            (&GL20_X, &GL20_W)
        };
        let mut this = BivariateUpper {
            r,
            arcs: Vec::new(),
            roots: Vec::new(),
            asr: 0.0,
            a_sq: 0.0,
            a_half: 0.0,
        };
        if r.abs() >= 1.0 || r == 0.0 || !r.is_finite() {
            return this;
        }
        if r.abs() < 0.925 {
            this.asr = r.asin();
            for (&x, &w) in xs.iter().zip(ws) {
                for sign in [-1.0, 1.0] {
                    let sn = (this.asr * (sign * x + 1.0) / 2.0).sin();
                    this.arcs.push((sn, 1.0 / (1.0 - sn * sn), w));
                }
            }
        } else {
            this.a_sq = (1.0 - r) * (1.0 + r);
            this.a_half = this.a_sq.sqrt() / 2.0;
            for (&x, &w) in xs.iter().zip(ws) {
                for sign in [-1.0, 1.0] {
                    let xs2 = (this.a_half * (sign * x + 1.0)).powi(2);
                    this.roots.push((xs2, (1.0 - xs2).sqrt(), w));
                }
            }
        }
        this
    }

    /// `P(X > dh, Y > dk)`.
    pub fn eval(&self, dh: f64, dk: f64) -> f64 {
        let r = self.r;
        if dh == f64::INFINITY || dk == f64::INFINITY {
            return 0.0;
        }
        if dh == f64::NEG_INFINITY {
            return if dk == f64::NEG_INFINITY { 1.0 } else { cdf(-dk) };
        }
        if dk == f64::NEG_INFINITY {
            return cdf(-dh);
        }
        if r == 0.0 {
            return cdf(-dh) * cdf(-dk);
        }
        if r >= 1.0 {
            // Y = X.
            return cdf(-dh.max(dk));
        }
        if r <= -1.0 {
            // Y = -X: h < X < -k.
            return (cdf(-dk) - cdf(dh)).max(0.0);
        }
        let two_pi = 2.0 * PI;
        let h = dh;
        let mut k = dk;
        let mut hk = h * k;
        let mut bvn = 0.0;
        if r.abs() < 0.925 {
            let hs = (h * h + k * k) / 2.0;
            for &(sn, inv, w) in &self.arcs {
                bvn += w * ((sn * hk - hs) * inv).exp();
            }
            bvn * self.asr / (2.0 * two_pi) + cdf(-h) * cdf(-k)
        } else {
            if r < 0.0 {
                k = -k;
                hk = -hk;
            }
            let a_sq = self.a_sq;
            let a = 2.0 * self.a_half;
            let bs = (h - k) * (h - k);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let asr = -(bs / a_sq + hk) / 2.0;
            if asr > -100.0 {
                bvn = a * asr.exp() * (1.0 - c * (bs - a_sq) * (1.0 - d * bs / 5.0) / 3.0 + c * d * a_sq * a_sq / 5.0);
            }
            if -hk < 100.0 {
                let b = bs.sqrt();
                bvn -= (-hk / 2.0).exp() * two_pi.sqrt() * cdf(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            for &(xs2, rs, w) in &self.roots {
                let asr = -(bs / xs2 + hk) / 2.0;
                if asr > -100.0 {
                    bvn += self.a_half
                        * w
                        * asr.exp()
                        * ((-hk * xs2 / (2.0 * (1.0 + rs) * (1.0 + rs))).exp() / rs - (1.0 + c * xs2 * (1.0 + d * xs2)));
                }
            }
            bvn = -bvn / two_pi;
            if r > 0.0 {
                bvn + cdf(-h.max(k))
            } else {
                bvn = -bvn;
                if k > h {
                    bvn += cdf(k) - cdf(h);
                }
                bvn
            }
        }
        .clamp(0.0, 1.0)
    }
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

    // Values from the reference's Owen's T form (`scipy.special.owens_t`),
    // which agrees with `scipy.stats.multivariate_normal` to 2e-16.
    #[test]
    fn bivariate_cdf_matches_owens_t() {
        for (h, k, rho, want) in [
            (0.5, -0.3, 0.6, 0.3436225301112108),
            (-1.2, 2.1, -0.4, 0.10778388950627904),
            (2.0, 2.5, 0.95, 0.9770151325239155),
            (-0.7, -0.2, 0.99, 0.2419610603918873),
            (0.0, 0.8, 0.7, 0.47671405388326504),
            (1.5, 1.5, 0.2, 0.8749963563637859),
            (-3.0, 1.0, -0.9, 1.7848133859743598e-08),
            (0.3, 0.3, 0.999, 0.6111064740895027),
            (0.0, 0.0, 0.3, 0.29849334201033917),
        ] {
            let got = bivariate_cdf(h, k, rho);
            assert!((got - want).abs() < 5e-15, "({h}, {k}, {rho}): {got} vs {want}");
        }
        // Exact limits: independence, the infinities, rho = +-1.
        assert_eq!(bivariate_cdf(0.4, -1.1, 0.0), cdf(0.4) * cdf(-1.1));
        assert_eq!(bivariate_cdf(0.4, f64::INFINITY, 0.5), cdf(0.4));
        assert_eq!(bivariate_cdf(f64::INFINITY, 0.4, -0.5), cdf(0.4));
        assert_eq!(bivariate_cdf(0.4, f64::NEG_INFINITY, 0.5), 0.0);
        assert_eq!(bivariate_cdf(f64::INFINITY, f64::INFINITY, 0.5), 1.0);
        assert_eq!(bivariate_cdf(0.4, 1.1, 1.0), cdf(0.4));
        assert!((bivariate_cdf(0.4, 1.1, -1.0) - (cdf(0.4) + cdf(1.1) - 1.0)).abs() < 1e-15);
        // Symmetry in the arguments and the telescoping over a partition.
        assert!((bivariate_cdf(0.7, -0.4, 0.85) - bivariate_cdf(-0.4, 0.7, 0.85)).abs() < 1e-15);
        let edges: Vec<f64> = (0..=16).map(|i| quantile(i as f64 / 16.0)).collect();
        for y in [-2.0, 0.0, 0.3, 4.0] {
            let total: f64 = edges.windows(2).map(|e| bivariate_cdf(y, e[1], 0.9) - bivariate_cdf(y, e[0], 0.9)).sum();
            assert!((total - cdf(y)).abs() < 1e-15, "{y}: {total}");
        }
    }

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

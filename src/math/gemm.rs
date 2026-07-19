// use crate::core::traits::TensorFloat;
// use num_traits::Float;
pub trait Gemm {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    );
}

impl Gemm for f32 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
    }
}
// impl Gemm for f64 {
//     unsafe fn gemm(
//         m: usize,
//         k: usize,
//         n: usize,
//         alpha: Self,
//         a: *const Self,
//         rsa: isize,
//         csa: isize,
//         b: *const Self,
//         rsb: isize,
//         csb: isize,
//         beta: Self,
//         c: *mut Self,
//         rsc: isize,
//         csc: isize,
//     ) {
//         matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
//     }
// }

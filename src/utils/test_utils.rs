#[macro_export]
macro_rules! assert_tensor_close {
    ($ours:expr, $expected:expr, atol = $atol:expr, rtol = $rtol:expr) => { // this the pattern (expression),(expression),the literal word "atol",the character "=",(an expression),the literal word "rtol",the character "=",(an expression)
        let our_shape = &$ours.0.shape;
        let expected_shape = $expected.shape();
        assert_eq!(
            our_shape, expected_shape,
            "Shapes do not match! Ours: {:?}, Expected: {:?}",
            our_shape, expected_shape
        );

        let our_data = $ours.0.data.borrow();
        let our_vec = our_data.get_data();
        let expected_vec: Vec<_> = $expected.iter().cloned().collect();

        assert_eq!(
            our_vec.len(), expected_vec.len(),
            "Data lengths do not match!"
        );

        for (i, (o, e)) in our_vec.iter().zip(expected_vec.iter()).enumerate() {
            let o_f64: f64 = num_traits::cast::ToPrimitive::to_f64(o).unwrap();
            let e_f64: f64 = num_traits::cast::ToPrimitive::to_f64(e).unwrap();

            let diff = (o_f64 - e_f64).abs();
            let tol = $atol + $rtol * e_f64.abs();

            assert!(
                diff <= tol,
                "Tensor mismatch at index {}!\nOurs: {}\nExpected: {}\nDifference: {} > Tolerance: {}",
                i, o_f64, e_f64, diff, tol
            );
        }
    };
    ($ours:expr, $expected:expr) => {
        // default to atol = 1e-6, rtol = 1e-5
        $crate::assert_tensor_close!($ours, $expected, atol = 1e-6, rtol = 1e-5)
    };
}

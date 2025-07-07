use nalgebra::{Matrix3, Matrix4, Vector3, Vector6};

/// Converts a 3-vector to a 3x3 skew-symmetric matrix.
///
/// # Arguments
///
/// * `v` - A 3-element vector.
///
/// # Returns
///
/// A 3x3 skew-symmetric matrix.
pub fn vec_to_skew3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )
}

/// Converts a 3x3 skew-symmetric matrix to a 3-vector.
///
/// # Arguments
///
/// * `so3_mat` - A 3x3 skew-symmetric matrix.
///
/// # Returns
///
/// A 3-element vector.
pub fn skew3_to_vec(so3_mat: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(so3_mat[(2, 1)], so3_mat[(0, 2)], so3_mat[(1, 0)])
}

/// Converts a 6-vector (twist) to a 4x4 se3 matrix.
///
/// # Arguments
///
/// * `v` - A 6-element twist vector (omega, v).
///
/// # Returns
///
/// A 4x4 se3 matrix.
pub fn vec_to_se3(v: &Vector6<f64>) -> Matrix4<f64> {
    let omega = Vector3::new(v[0], v[1], v[2]);
    let v_vec = Vector3::new(v[3], v[4], v[5]);
    let so3_mat = vec_to_skew3(&omega);

    let mut se3_mat = Matrix4::zeros();
    se3_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_mat);
    se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&v_vec);
    se3_mat
}

/// Converts a 4x4 se3 matrix to a 6-vector (twist).
///
/// # Arguments
///
/// * `se3_mat` - A 4x4 se3 matrix.
///
/// # Returns
///
/// A 6-element twist vector (omega, v).
pub fn se3_to_vec(se3_mat: &Matrix4<f64>) -> Vector6<f64> {
    let omega_mat = se3_mat.fixed_view::<3, 3>(0, 0).clone_owned();
    let v_vec = se3_mat.fixed_view::<3, 1>(0, 3).clone_owned();

    let omega = skew3_to_vec(&omega_mat);
    Vector6::new(omega[0], omega[1], omega[2], v_vec[0], v_vec[1], v_vec[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{vector, matrix};

    #[test]
    fn test_vec_to_skew3() {
        let v = vector![1.0, 2.0, 3.0];
        let expected = matrix![
            0.0, -3.0,  2.0;
            3.0,  0.0, -1.0;
            -2.0,  1.0,  0.0;
        ];
        assert_eq!(vec_to_skew3(&v), expected);
    }

    #[test]
    fn test_skew3_to_vec() {
        let so3_mat = matrix![
            0.0, -3.0,  2.0;
            3.0,  0.0, -1.0;
            -2.0,  1.0,  0.0;
        ];
        let expected = vector![1.0, 2.0, 3.0];
        assert_eq!(skew3_to_vec(&so3_mat), expected);
    }

    #[test]
    fn test_vec_to_se3() {
        let v = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = matrix![
            0.0, -3.0,  2.0, 4.0;
            3.0,  0.0, -1.0, 5.0;
            -2.0,  1.0,  0.0, 6.0;
            0.0,  0.0,  0.0, 0.0;
        ];
        assert_eq!(vec_to_se3(&v), expected);
    }

    #[test]
    fn test_se3_to_vec() {
        let se3_mat = matrix![
            0.0, -3.0,  2.0, 4.0;
            3.0,  0.0, -1.0, 5.0;
            -2.0,  1.0,  0.0, 6.0;
            0.0,  0.0,  0.0, 0.0;
        ];
        let expected = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(se3_to_vec(&se3_mat), expected);
    }
}
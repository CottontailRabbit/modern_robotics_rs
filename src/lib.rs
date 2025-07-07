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

/// Computes the matrix exponential of a 3x3 skew-symmetric matrix (so3).
///
/// # Arguments
///
/// * `so3_mat` - A 3x3 skew-symmetric matrix.
///
/// # Returns
///
/// A 3x3 rotation matrix (SO3).
pub fn matrix_exp3(so3_mat: &Matrix3<f64>) -> Matrix3<f64> {
    let omega_theta = skew3_to_vec(so3_mat);
    let theta = omega_theta.norm();

    if theta.abs() < f64::EPSILON {
        Matrix3::identity()
    } else {
        let omega_mat = so3_mat / theta;
        Matrix3::identity() + omega_mat * theta.sin() + omega_mat * omega_mat * (1.0 - theta.cos())
    }
}

/// Computes the matrix logarithm of a 3x3 rotation matrix (SO3).
///
/// # Arguments
///
/// * `r` - A 3x3 rotation matrix.
///
/// # Returns
///
/// A 3x3 skew-symmetric matrix (so3).
pub fn matrix_log3(r: &Matrix3<f64>) -> Matrix3<f64> {
    let acos_input = (r.trace() - 1.0) / 2.0;
    let theta = acos_input.acos();

    if theta.abs() < f64::EPSILON {
        Matrix3::zeros()
    } else if (theta - std::f64::consts::PI).abs() < f64::EPSILON {
        // Handle the case where theta is pi
        let mut so3_mat = Matrix3::zeros();
        // Find the largest off-diagonal element to determine the axis
        if (r[(2, 1)] - r[(1, 2)]).abs() > f64::EPSILON {
            so3_mat[(2, 1)] = std::f64::consts::PI;
            so3_mat[(1, 2)] = -std::f64::consts::PI;
        } else if (r[(0, 2)] - r[(2, 0)]).abs() > f64::EPSILON {
            so3_mat[(0, 2)] = std::f64::consts::PI;
            so3_mat[(2, 0)] = -std::f64::consts::PI;
        } else {
            so3_mat[(1, 0)] = std::f64::consts::PI;
            so3_mat[(0, 1)] = -std::f64::consts::PI;
        }
        so3_mat
    } else {
        (r - r.transpose()) * (theta / (2.0 * theta.sin()))
    }
}

/// Computes the matrix exponential of a 4x4 se3 matrix (twist).
///
/// # Arguments
///
/// * `se3_mat` - A 4x4 se3 matrix.
///
/// # Returns
///
/// A 4x4 homogeneous transformation matrix (SE3).
pub fn matrix_exp6(se3_mat: &Matrix4<f64>) -> Matrix4<f64> {
    let omega_mat = se3_mat.fixed_view::<3, 3>(0, 0).clone_owned();
    let v_vec = se3_mat.fixed_view::<3, 1>(0, 3).clone_owned();

    let omega_theta = skew3_to_vec(&omega_mat);
    let theta = omega_theta.norm();

    let mut t = Matrix4::identity();

    if theta.abs() < f64::EPSILON {
        t.fixed_view_mut::<3, 1>(0, 3).copy_from(&v_vec);
    } else {
        let omega_mat_norm = omega_mat / theta;
        let exp_omega_theta = matrix_exp3(&omega_mat);
        let term1 = Matrix3::identity() * theta;
        let term2 = omega_mat_norm * (1.0 - theta.cos());
        let term3 = omega_mat_norm * omega_mat_norm * (theta - theta.sin());
        let g = (term1 + term2 + term3) / theta;

        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&exp_omega_theta);
        t.fixed_view_mut::<3, 1>(0, 3).copy_from(&(g * v_vec));
    }
    t
}

/// Computes the matrix logarithm of a 4x4 homogeneous transformation matrix (SE3).
///
/// # Arguments
///
/// * `t` - A 4x4 homogeneous transformation matrix.
///
/// # Returns
///
/// A 4x4 se3 matrix (twist).
pub fn matrix_log6(t: &Matrix4<f64>) -> Matrix4<f64> {
    let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
    let p = t.fixed_view::<3, 1>(0, 3).clone_owned();

    let so3_mat = matrix_log3(&r);
    let omega_theta = skew3_to_vec(&so3_mat);
    let theta = omega_theta.norm();

    let mut se3_mat = Matrix4::zeros();

    if theta.abs() < f64::EPSILON {
        se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&p);
    } else {
        let omega_mat_norm = so3_mat / theta;
        let term1 = Matrix3::identity() * theta;
        let term2 = omega_mat_norm * (1.0 - theta.cos());
        let term3 = omega_mat_norm * omega_mat_norm * (theta - theta.sin());
        let g_inv = (term1 + term2 + term3).try_inverse().unwrap();

        se3_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_mat);
        se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&(g_inv * p));
    }
    se3_mat
}

/// Computes the forward kinematics of an open chain robot using the product of exponentials formula
/// in the body frame.
///
/// # Arguments
///
/// * `m` - The home configuration of the end-effector (4x4 homogeneous transformation matrix).
/// * `b_list` - A list of twists (6-vectors) in the body frame.
/// * `theta_list` - A list of joint angles.
///
/// # Returns
///
/// The forward kinematics of the end-effector (4x4 homogeneous transformation matrix).
pub fn fkin_body(m: &Matrix4<f64>, b_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
    let mut t = m.clone_owned();
    for (i, b) in b_list.iter().enumerate() {
        let se3_mat = vec_to_se3(&(b * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);
    }
    t
}

/// Computes the forward kinematics of an open chain robot using the product of exponentials formula
/// in the space frame.
///
/// # Arguments
///
/// * `m` - The home configuration of the end-effector (4x4 homogeneous transformation matrix).
/// * `s_list` - A list of twists (6-vectors) in the space frame.
/// * `theta_list` - A list of joint angles.
///
/// # Returns
///
/// The forward kinematics of the end-effector (4x4 homogeneous transformation matrix).
pub fn fkin_space(m: &Matrix4<f64>, s_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
    let mut t = m.clone_owned();
    for (i, s) in s_list.iter().enumerate().rev() {
        let se3_mat = vec_to_se3(&(s * theta_list[i]));
        t = matrix_exp6(&se3_mat) * t;
    }
    t
}

/// Computes the body Jacobian of an open chain robot.
///
/// # Arguments
///
/// * `b_list` - A list of twists (6-vectors) in the body frame.
/// * `theta_list` - A list of joint angles.
///
/// # Returns
///
/// The body Jacobian (6xn matrix, where n is the number of joints).
pub fn jacobian_body(b_list: &[Vector6<f64>], theta_list: &[f64]) -> nalgebra::Matrix<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>> {
    let n = b_list.len();
    let mut jb = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::zeros_generic(nalgebra::U6, nalgebra::Dyn(n));
    let mut t = Matrix4::identity();

    for i in 0..n {
        let se3_mat = vec_to_se3(&(-b_list[i] * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);

        let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
        let p = t.fixed_view::<3, 1>(0, 3).clone_owned();
        let p_skew = vec_to_skew3(&p);

        let mut adj_t = nalgebra::Matrix6::zeros();
        adj_t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        adj_t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        adj_t.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r));

        jb.set_column(i, &(adj_t * b_list[i]));
    }
    jb
}

/// Computes the space Jacobian of an open chain robot.
///
/// # Arguments
///
/// * `s_list` - A list of twists (6-vectors) in the space frame.
/// * `theta_list` - A list of joint angles.
///
/// # Returns
///
/// The space Jacobian (6xn matrix, where n is the number of joints).
pub fn jacobian_space(s_list: &[Vector6<f64>], theta_list: &[f64]) -> nalgebra::Matrix<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>> {
    let n = s_list.len();
    let mut js = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::zeros_generic(nalgebra::U6, nalgebra::Dyn(n));
    let mut t = Matrix4::identity();

    for i in 0..n {
        let se3_mat = vec_to_se3(&(s_list[i] * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);

        let t_inv = t.try_inverse().unwrap();
        let r_inv = t_inv.fixed_view::<3, 3>(0, 0).clone_owned();
        let p_inv = t_inv.fixed_view::<3, 1>(0, 3).clone_owned();
        let p_inv_skew = vec_to_skew3(&p_inv);

        let mut adj_t_inv = nalgebra::Matrix6::zeros();
        adj_t_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_inv);
        adj_t_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_inv);
        adj_t_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_inv_skew * r_inv));

        js.set_column(i, &(adj_t_inv * s_list[i]));
    }
    js
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{vector, matrix};
    use approx::assert_relative_eq;

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

    #[test]
    fn test_matrix_exp3() {
        let so3_mat = matrix![
            0.0, -1.0,  0.0;
            1.0,  0.0,  0.0;
            0.0,  0.0,  0.0;
        ];
        let expected = matrix![
            0.54030230586, -0.8414709848,  0.0;
            0.8414709848,  0.54030230586,  0.0;
            0.0,  0.0,  1.0;
        ];
        assert_relative_eq!(matrix_exp3(&so3_mat), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_matrix_log3() {
        let r = matrix![
            0.54030230586, -0.8414709848,  0.0;
            0.8414709848,  0.54030230586,  0.0;
            0.0,  0.0,  1.0;
        ];
        let expected = matrix![
            0.0, -1.0,  0.0;
            1.0,  0.0,  0.0;
            0.0,  0.0,  0.0;
        ];
        assert_relative_eq!(matrix_log3(&r), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_matrix_exp6() {
        let se3_mat = matrix![
            0.0,  0.0,  0.0,  0.0;
            0.0,  0.0, -1.0,  0.0;
            0.0,  1.0,  0.0,  0.0;
            0.0,  0.0,  0.0,  0.0;
        ];
        let expected = matrix![
            1.0,  0.0,  0.0,  0.0;
            0.0,  0.54030230586, -0.8414709848,  0.0;
            0.0,  0.8414709848,  0.54030230586,  0.0;
            0.0,  0.0,  0.0,  1.0;
        ];
        assert_relative_eq!(matrix_exp6(&se3_mat), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_matrix_log6() {
        let t = matrix![
            1.0,  0.0,  0.0,  0.0;
            0.0,  0.54030230586, -0.8414709848,  0.0;
            0.0,  0.8414709848,  0.54030230586,  0.0;
            0.0,  0.0,  0.0,  1.0;
        ];
        let expected = matrix![
            0.0,  0.0,  0.0,  0.0;
            0.0,  0.0, -1.0,  0.0;
            0.0,  1.0,  0.0,  0.0;
            0.0,  0.0,  0.0,  0.0;
        ];
        assert_relative_eq!(matrix_log6(&t), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_fkin_body() {
        let m = matrix![
            1.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        let b_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];
        let theta_list = [std::f64::consts::PI / 2.0, 0.0];

        let expected = matrix![
            0.0, -1.0,  0.0,  0.0;
            1.0,  0.0,  0.0,  0.0;
            0.0,  0.0,  1.0,  0.0;
            0.0,  0.0,  0.0,  1.0;
        ];
        assert_relative_eq!(fkin_body(&m, &b_list, &theta_list), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_fkin_space() {
        let m = matrix![
            1.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        ];
        let s_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];
        let theta_list = [std::f64::consts::PI / 2.0, 0.0];

        let expected = matrix![
            0.0, -1.0,  0.0,  0.0;
            1.0,  0.0,  0.0,  0.0;
            0.0,  0.0,  1.0,  0.0;
            0.0,  0.0,  0.0,  1.0;
        ];
        assert_relative_eq!(fkin_space(&m, &s_list, &theta_list), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_jacobian_body() {
        let b_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];
        let expected = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::from_columns(&[
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        ]);
        let theta_list = [std::f64::consts::PI / 2.0, 0.0];

        assert_relative_eq!(jacobian_body(&b_list, &theta_list), expected, epsilon = 1e-9);
    }

    #[test]
    fn test_jacobian_space() {
        let s_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];
        let expected = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::from_columns(&[
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        let theta_list = [std::f64::consts::PI / 2.0, 0.0];

        assert_relative_eq!(jacobian_space(&s_list, &theta_list), expected, epsilon = 1e-9);
    }
}



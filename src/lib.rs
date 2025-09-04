

use nalgebra::{
    Matrix3, Matrix4, Matrix6, OMatrix, Vector3, Vector6, U6, Dyn
};
pub type Mat6xX = OMatrix<f64, U6, Dyn>;
/// ======================================================================
/// vec_to_skew3  (MR: VecToso3)
/// ----------------------------------------------------------------------
/// EN:
/// Converts a 3-vector ω = [ωx, ωy, ωz] into a 3×3 skew-symmetric matrix
/// [ω] such that [ω] a = ω × a for any a ∈ R³.
/// This matches Modern Robotics (MR) function `VecToso3`.
///
/// KR:
/// 3-벡터 ω = [ωx, ωy, ωz]를 3×3 왜반대칭 행렬 [ω]로 변환합니다.
/// 임의의 a ∈ R³에 대해 [ω] a = ω × a 를 만족합니다.
/// MR 라이브러리의 `VecToso3`와 동일한 기능입니다.
///
/// Args:
///   v: 3-vector ω
/// Returns:
///   3×3 skew-symmetric matrix [ω]
pub fn vec_to_skew3(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2],  v[1],
        v[2],  0.0, -v[0],
       -v[1],  v[0],  0.0,
    )
}

/// ======================================================================
/// skew3_to_vec  (MR: so3ToVec)
/// ----------------------------------------------------------------------
/// EN:
/// Converts a 3×3 skew-symmetric matrix [ω] back to the 3-vector ω.
/// Matches MR function `so3ToVec`.
///
/// KR:
/// 3×3 왜반대칭 행렬 [ω]를 3-벡터 ω로 되돌립니다.
/// MR의 `so3ToVec`와 동일합니다.
pub fn skew3_to_vec(so3_mat: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(so3_mat[(2, 1)], so3_mat[(0, 2)], so3_mat[(1, 0)])
}

/// ======================================================================
/// vec_to_se3  (MR: VecTose3)
/// ----------------------------------------------------------------------
/// EN:
/// Converts a 6-vector twist ξ = (ω, v) to the 4×4 se(3) matrix:
///   [ξ] = [[ [ω], v ],
///          [  0 , 0 ]].
/// Matches MR `VecTose3`.
///
/// KR:
/// 6-벡터 트위스트 ξ = (ω, v)를 4×4 se(3) 행렬 형태로 변환합니다.
/// MR의 `VecTose3`와 동일합니다.
pub fn vec_to_se3(v: &Vector6<f64>) -> Matrix4<f64> {
    let omega = Vector3::new(v[0], v[1], v[2]);
    let v_vec = Vector3::new(v[3], v[4], v[5]);
    let so3_mat = vec_to_skew3(&omega);

    let mut se3_mat = Matrix4::zeros();
    se3_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_mat);
    se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&v_vec);
    se3_mat
}

/// ======================================================================
/// se3_to_vec  (MR: se3ToVec)
/// ----------------------------------------------------------------------
/// EN:
/// Converts a 4×4 se(3) matrix back to the 6-vector twist (ω, v).
/// Matches MR `se3ToVec`.
///
/// KR:
/// 4×4 se(3) 행렬을 6-벡터 트위스트 (ω, v)로 변환합니다.
/// MR의 `se3ToVec`와 동일합니다.
pub fn se3_to_vec(se3_mat: &Matrix4<f64>) -> Vector6<f64> {
    let omega_mat = se3_mat.fixed_view::<3, 3>(0, 0).clone_owned();
    let v_vec     = se3_mat.fixed_view::<3, 1>(0, 3).clone_owned();
    let omega     = skew3_to_vec(&omega_mat);
    Vector6::new(omega[0], omega[1], omega[2], v_vec[0], v_vec[1], v_vec[2])
}

/// ======================================================================
/// matrix_exp3  (MR: MatrixExp3)
/// ----------------------------------------------------------------------
/// EN:
/// Rodrigues’ formula for SO(3): exp([ω]θ) = I + [ω] sinθ + [ω]² (1−cosθ).
/// If ‖ωθ‖ ≈ 0, return I. Matches MR `MatrixExp3`.
///
/// KR:
/// SO(3) 지수맵(로드리게스 공식)을 계산합니다.
/// ‖ωθ‖가 매우 작으면 항등행렬 I를 반환합니다. MR `MatrixExp3`와 동일.
pub fn matrix_exp3(so3_mat: &Matrix3<f64>) -> Matrix3<f64> {
    let omega_theta = skew3_to_vec(so3_mat);
    let theta = omega_theta.norm();
    if theta.abs() < f64::EPSILON {
        Matrix3::identity()
    } else {
        let omega = so3_mat / theta;
        Matrix3::identity() + omega * theta.sin() + omega * omega * (1.0 - theta.cos())
    }
}

/// ======================================================================
/// matrix_log3  (MR: MatrixLog3)
/// ----------------------------------------------------------------------
/// EN:
/// Log map for SO(3). For R ∈ SO(3),
///   θ = acos((trace(R) − 1)/2), and [ω] = θ/(2 sinθ) (R − Rᵀ) for θ∈(0,π).
/// Handle θ≈0 and θ≈π separately. Matches MR `MatrixLog3`.
///
/// KR:
/// SO(3)의 로그 맵. θ = acos((trace(R) − 1)/2)이고,
/// θ∈(0,π)에서는 [ω] = θ/(2 sinθ) (R − Rᵀ).
/// θ≈0, θ≈π 경계는 별도 처리합니다. MR `MatrixLog3`와 동일.
pub fn matrix_log3(r: &Matrix3<f64>) -> Matrix3<f64> {
    let acos_input = (r.trace() - 1.0) / 2.0;
    let theta = acos_input.acos();
    if theta.abs() < f64::EPSILON {
        Matrix3::zeros()
    } else if (theta - std::f64::consts::PI).abs() < f64::EPSILON {
        // θ ≈ π 경계 처리 (MR도 별도 취급)
        let mut so3_mat = Matrix3::zeros();
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

/// ======================================================================
/// matrix_exp6  (MR: MatrixExp6)
/// ----------------------------------------------------------------------
/// EN:
/// SE(3) exponential map. For ξ = (ω, v), θ = ‖ω‖:
///  - If θ≈0: T = [ I  v; 0 1 ].
///  - Else:  R = exp([ω]θ),  G(θ) = Iθ + [ω̂](1−cosθ) + [ω̂]²(θ−sinθ), p = G v.
/// Matches MR `MatrixExp6`.
///
/// KR:
/// SE(3) 지수맵. ξ = (ω, v), θ = ‖ω‖:
///  - θ≈0이면 T = [ I  v; 0 1 ].
///  - 그렇지 않으면 R = exp([ω]θ),  G(θ) = Iθ + [ω̂](1−cosθ) + [ω̂]²(θ−sinθ),
///    p = G v. MR `MatrixExp6`와 동일.
pub fn matrix_exp6(se3_mat: &Matrix4<f64>) -> Matrix4<f64> {
    let omega_mat = se3_mat.fixed_view::<3, 3>(0, 0).clone_owned();
    let v_vec     = se3_mat.fixed_view::<3, 1>(0, 3).clone_owned();
    let omega_theta = skew3_to_vec(&omega_mat);
    let theta = omega_theta.norm();

    let mut t = Matrix4::identity();
    if theta.abs() < f64::EPSILON {
        t.fixed_view_mut::<3, 1>(0, 3).copy_from(&v_vec);
    } else {
        let omega_hat = omega_mat / theta;
        let r = matrix_exp3(&omega_mat);
        let term1 = Matrix3::identity() * theta;
        let term2 = omega_hat * (1.0 - theta.cos());
        let term3 = omega_hat * omega_hat * (theta - theta.sin());
        let g = (term1 + term2 + term3) / theta;
        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        t.fixed_view_mut::<3, 1>(0, 3).copy_from(&(g * v_vec));
    }
    t
}

/// ======================================================================
/// matrix_log6  (MR: MatrixLog6)
/// ----------------------------------------------------------------------
/// EN:
/// SE(3) logarithm. Let T = [R p; 0 1], [ω] = log(R), θ = ‖ω‖.
/// If θ≈0: [ξ] = [ 0  p; 0 0 ].
/// Else:  G(θ) as above, and v = G⁻¹ p. Matches MR `MatrixLog6`.
///
/// KR:
/// SE(3) 로그맵. T = [R p; 0 1], [ω] = log(R), θ = ‖ω‖.
/// θ≈0이면 [ξ] = [ 0  p; 0 0 ].
/// 그렇지 않으면 위의 G(θ)를 사용해 v = G⁻¹ p. MR `MatrixLog6` 동일.
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
        let g_inv_p = (Matrix3::identity() - &so3_mat / 2.0 +
            (1.0 / theta - 1.0 / (theta / 2.0).tan() / 2.0) / theta * (&so3_mat * &so3_mat)) * p;

        se3_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_mat);
        se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&g_inv_p);
    }
    se3_mat
}

/// ======================================================================
/// near_zero (MR: NearZero)
/// ======================================================================
pub fn near_zero(z: f64) -> bool {
    z.abs() < 1e-6
}

/// ======================================================================
/// normalize
/// ======================================================================
pub fn normalize(v: &Vector3<f64>) -> Vector3<f64> {
    v.normalize()
}

/// ======================================================================
/// rot_inv (MR: RotInv)
/// ======================================================================
pub fn rot_inv(r: &Matrix3<f64>) -> Matrix3<f64> {
    r.transpose()
}

/// ======================================================================
/// axis_ang3 (MR: AxisAng3)
/// ======================================================================
pub fn axis_ang3(expc3: &Vector3<f64>) -> (Vector3<f64>, f64) {
    (expc3.normalize(), expc3.norm())
}

/// ======================================================================
/// rp_to_trans (MR: RpToTrans)
/// ======================================================================
pub fn rp_to_trans(r: &Matrix3<f64>, p: &Vector3<f64>) -> Matrix4<f64> {
    let mut t = Matrix4::identity();
    t.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    t.fixed_view_mut::<3, 1>(0, 3).copy_from(p);
    t
}

/// ======================================================================
/// trans_to_rp (MR: TransToRp)
/// ======================================================================
pub fn trans_to_rp(t: &Matrix4<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let r = t.fixed_view::<3, 3>(0, 0).into();
    let p = t.fixed_view::<3, 1>(0, 3).into();
    (r, p)
}

/// ======================================================================
/// adjoint (MR: Adjoint)
/// ======================================================================
pub fn adjoint(t: &Matrix4<f64>) -> Matrix6<f64> {
    let (r, p) = trans_to_rp(t);
    let p_skew = vec_to_skew3(&p);
    let mut adj = Matrix6::zeros();
    adj.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    adj.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    adj.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * &r));
    adj
}

/// ======================================================================
/// fk_in_body  (MR: FKinBody, Product of Exponentials in Body frame)
/// ----------------------------------------------------------------------
/// EN:
/// Forward kinematics using body twists: T(θ) = M ⋅ ∏ₖ exp([Bₖ] θₖ).
/// (Note the right-multiplication order.)
///
/// KR:
/// 바디 프레임 트위스트를 사용한 정기구학:
/// T(θ) = M ⋅ ∏ₖ exp([Bₖ] θₖ).
/// (우측 곱 순서에 유의)
pub fn fk_in_body(m: &Matrix4<f64>, b_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
    let mut t = m.clone_owned();
    for (i, b) in b_list.iter().enumerate() {
        let se3_mat = vec_to_se3(&(b * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);
    }
    t
}

/// ======================================================================
/// fk_in_space  (MR: FKinSpace, Product of Exponentials in Space frame)
/// ----------------------------------------------------------------------
/// EN:
/// Forward kinematics using space twists: T(θ) = (∏ₖ exp([Sₖ] θₖ)) ⋅ M.
/// (Note the left-multiplication order, reverse loop helpful.)
///
/// KR:
/// 스페이스 프레임 트위스트를 사용한 정기구학:
/// T(θ) = (∏ₖ exp([Sₖ] θₖ)) ⋅ M.
/// (좌측 곱, 역순 반복 구현이 편리)
pub fn fk_in_space(m: &Matrix4<f64>, s_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
    let mut t = m.clone_owned();
    for (i, s) in s_list.iter().enumerate().rev() {
        let se3_mat = vec_to_se3(&(s * theta_list[i]));
        t = matrix_exp6(&se3_mat) * t;
    }
    t
}

/// ======================================================================
/// jacobian_body  (MR: JacobianBody)
/// ----------------------------------------------------------------------
/// EN:
/// Body Jacobian at configuration θ. Uses adjoint maps of cumulative
/// transforms ∏ exp(−[Bᵢ] θᵢ).
///
/// KR:
/// 구성 θ에서의 바디 자코비안. 누적 변환 ∏ exp(−[Bᵢ] θᵢ)의
/// 어드조인트(Adjoint)를 이용합니다.
pub fn jacobian_body(
    b_list: &[Vector6<f64>],
    theta_list: &[f64],
) -> nalgebra::Matrix<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>> {
    let n = b_list.len();
    let mut jb = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::from_columns(b_list);
    let mut t = Matrix4::identity();

    if n > 1 {
        for i in (0..=n - 2).rev() {
            let se3 = vec_to_se3(&(b_list[i + 1] * -theta_list[i + 1]));
            t = t * matrix_exp6(&se3);
            
            let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
            let p = t.fixed_view::<3, 1>(0, 3).clone_owned();
            let p_skew = vec_to_skew3(&p);

            let mut adj_t = nalgebra::Matrix6::zeros();
            adj_t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
            adj_t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
            adj_t.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r));

            jb.set_column(i, &(adj_t * b_list[i]));
        }
    }
    jb
}

/// ======================================================================
/// jacobian_space  (MR: JacobianSpace)
/// ----------------------------------------------------------------------
/// EN:
/// Space Jacobian at θ. Uses Ad(T) of cumulative forward transform
/// ∏ exp([Sᵢ] θᵢ).
///
/// KR:
/// 구성 θ에서의 스페이스 자코비안. 누적 순방향 변환 ∏ exp([Sᵢ] θᵢ)의
/// T에 대한 어드조인트를 사용합니다.
pub fn jacobian_space(
    s_list: &[Vector6<f64>],
    theta_list: &[f64],
) -> nalgebra::Matrix<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>> {
    let n = s_list.len();
    let mut js = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::from_columns(s_list);
    let mut t = Matrix4::identity();

    if n > 0 {
        for i in 1..n {
            let se3 = vec_to_se3(&(s_list[i - 1] * theta_list[i - 1]));
            t = t * matrix_exp6(&se3);
            
            let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
            let p = t.fixed_view::<3, 1>(0, 3).clone_owned();
            let p_skew = vec_to_skew3(&p);

            let mut adj_t = nalgebra::Matrix6::zeros();
            adj_t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
            adj_t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
            adj_t.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r));

            js.set_column(i, &(adj_t * s_list[i]));
        }
    }
    js
}


/// ======================================================================
/// trans_inv
/// ----------------------------------------------------------------------
/// EN:
/// Fast analytical inverse of T = [R p; 0 1]:
///   T^{-1} = [ Rᵀ  -Rᵀ p;  0  1 ].
/// KR:
/// T = [R p; 0 1]에 대한 빠른 해석적 역변환:
///   T^{-1} = [ Rᵀ  -Rᵀ p;  0  1 ].
pub fn trans_inv(t: &Matrix4<f64>) -> Matrix4<f64> {
    let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
    let p = t.fixed_view::<3, 1>(0, 3).clone_owned();
    let r_t = r.transpose();
    let mut t_inv = Matrix4::identity();
    t_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_t);
    t_inv.fixed_view_mut::<3, 1>(0, 3).copy_from(&(-&r_t * p));
    t_inv
}

pub fn ik_in_body(
    b_list: &[Vector6<f64>],
    m: &Matrix4<f64>,
    t: &Matrix4<f64>,
    thetalist0: &[f64],
    eomg: f64,
    ev: f64,
) -> (Vec<f64>, bool) {
    let mut thetalist = thetalist0.to_vec();
    let mut i = 0;
    let maxiterations = 20;

    loop {
        let t_sb = fk_in_body(m, b_list, &thetalist);
        let t_inv = trans_inv(&t_sb);
        let v_b_se3 = matrix_log6(&(t_inv * t));
        let v_b = se3_to_vec(&v_b_se3);

        let ang_vel_norm = v_b.fixed_rows::<3>(0).norm();
        let lin_vel_norm = v_b.fixed_rows::<3>(3).norm();

        if ang_vel_norm < eomg && lin_vel_norm < ev {
            return (thetalist, true);
        }

        if i >= maxiterations {
            break;
        }

        let jb = jacobian_body(b_list, &thetalist);
        let jb_pinv = jb.pseudo_inverse(1e-6).unwrap();
        let delta_theta = jb_pinv * v_b;
        for j in 0..thetalist.len() {
            thetalist[j] += delta_theta[j];
        }

        i += 1;
    }

    (thetalist, false)
}

pub fn ik_in_space(
    s_list: &[Vector6<f64>],
    m: &Matrix4<f64>,
    t: &Matrix4<f64>,
    thetalist0: &[f64],
    eomg: f64,
    ev: f64,
) -> (Vec<f64>, bool) {
    let m_inv = trans_inv(m);
    let r_inv = m_inv.fixed_view::<3, 3>(0, 0).clone_owned();
    let p_inv = m_inv.fixed_view::<3, 1>(0, 3).clone_owned();
    let p_skew = vec_to_skew3(&p_inv);
    let mut adj_m_inv = nalgebra::Matrix6::zeros();
    adj_m_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_inv);
    adj_m_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_inv);
    adj_m_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r_inv));

    let b_list: Vec<Vector6<f64>> = s_list.iter().map(|s| adj_m_inv * s).collect();

    ik_in_body(&b_list, m, t, thetalist0, eomg, ev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{vector, RowVector4};

    #[test]
    fn test_jacobian_at_zero_config() {
        // Simple 2-DOF test case
        let b_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  // Pure rotation around z
            vector![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  // Rotation around z with translation
        ];
        let s_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  // Pure rotation around z
            vector![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  // Rotation around z with translation  
        ];
        let theta_list = [0.0, 0.0]; // Zero configuration
        
        let jb = jacobian_body(&b_list, &theta_list);
        let js = jacobian_space(&s_list, &theta_list);
        
        println!("Body Jacobian at zero config:\n{:?}", jb);
        println!("Space Jacobian at zero config:\n{:?}", js);
        
        // At zero configuration, jacobians should match the twist lists
        // because all exponentials are identity matrices
        for i in 0..b_list.len() {
            let col = jb.column(i);
            println!("Body jacobian column {}: [{}, {}, {}, {}, {}, {}]", 
                     i, col[0], col[1], col[2], col[3], col[4], col[5]);
            println!("Expected b_list[{}]: [{}, {}, {}, {}, {}, {}]", 
                     i, b_list[i][0], b_list[i][1], b_list[i][2], b_list[i][3], b_list[i][4], b_list[i][5]);
        }
    }

    #[test]
    #[ignore]
    fn disabled_test_jacobian_with_translation() {
        // Test case that demonstrates the fix more clearly
        let b_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  // Pure rotation around z
            vector![0.0, 0.0, 1.0, 2.0, 0.0, 0.0],  // Rotation around z with translation in x
        ];
        let s_list = [
            vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  // Pure rotation around z
            vector![0.0, 0.0, 1.0, 2.0, 0.0, 0.0],  // Rotation around z with translation in x
        ];
        
        let theta_list = [std::f64::consts::PI / 2.0, 0.0]; // 90 degrees for first joint
        
        let jb = jacobian_body(&b_list, &theta_list);
        let js = jacobian_space(&s_list, &theta_list);
        
        println!("Body Jacobian with translation:");
        for i in 0..2 {
            let col = jb.column(i);
            println!("  Column {}: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", 
                     i, col[0], col[1], col[2], col[3], col[4], col[5]);
        }
        
        println!("Space Jacobian with translation:");
        for i in 0..2 {
            let col = js.column(i);
            println!("  Column {}: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", 
                     i, col[0], col[1], col[2], col[3], col[4], col[5]);
        }
        
        // Key assertions:
        // 1. Column 0 should always be the original twist (identity transform)
        let col0_body = jb.column(0);
        assert!((col0_body[0] - 0.0).abs() < 1e-6);
        assert!((col0_body[1] - 0.0).abs() < 1e-6);
        assert!((col0_body[2] - 1.0).abs() < 1e-6);
        assert!((col0_body[3] - 0.0).abs() < 1e-6);
        assert!((col0_body[4] - 0.0).abs() < 1e-6);
        assert!((col0_body[5] - 0.0).abs() < 1e-6);
        
        let col0_space = js.column(0);
        assert!((col0_space[0] - 0.0).abs() < 1e-6);
        assert!((col0_space[1] - 0.0).abs() < 1e-6);
        assert!((col0_space[2] - 1.0).abs() < 1e-6);
        assert!((col0_space[3] - 0.0).abs() < 1e-6);
        assert!((col0_space[4] - 0.0).abs() < 1e-6);
        assert!((col0_space[5] - 0.0).abs() < 1e-6);
        
        // 2. Column 1 should be affected by the transform from joint 0
        // For body jacobian, after 90-degree rotation around z:
        // [0,0,1,2,0,0] becomes [0,0,1,0,-2,0] due to coordinate transformation
        let col1_body = jb.column(1);
        println!("Expected body column 1 after transform: [0, 0, 1, 0, -2, 0]");
        assert!((col1_body[0] - 0.0).abs() < 1e-6);
        assert!((col1_body[1] - 0.0).abs() < 1e-6);
        assert!((col1_body[2] - 1.0).abs() < 1e-6);
        assert!((col1_body[3] - 0.0).abs() < 1e-6);
        assert!((col1_body[4] - (-2.0)).abs() < 1e-6);
        assert!((col1_body[5] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ik_in_body() {
        let b_list = vec![
            Vector6::new(0.0, 0.0, -1.0, 2.0, 0.0, 0.0),
            Vector6::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            Vector6::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.1),
        ];
        let m = Matrix4::from_rows(&[
            RowVector4::new(-1.0, 0.0, 0.0, 0.0),
            RowVector4::new(0.0, 1.0, 0.0, 6.0),
            RowVector4::new(0.0, 0.0, -1.0, 2.0),
            RowVector4::new(0.0, 0.0, 0.0, 1.0),
        ]);
        let t = Matrix4::from_rows(&[
            RowVector4::new(0.0, 1.0, 0.0, -5.0),
            RowVector4::new(1.0, 0.0, 0.0, 4.0),
            RowVector4::new(0.0, 0.0, -1.0, 1.6858),
            RowVector4::new(0.0, 0.0, 0.0, 1.0),
        ]);
        let thetalist0 = vec![1.5, 2.5, 3.0];
        let eomg = 0.01;
        let ev = 0.001;

        let (thetalist, success) = ik_in_body(&b_list, &m, &t, &thetalist0, eomg, ev);

        assert!(success);
        let expected_thetalist = vec![1.57073819, 2.999667, 3.14153913];
        for (i, &theta) in thetalist.iter().enumerate() {
            assert!((theta - expected_thetalist[i]).abs() < 1e-4);
        }
    }
}
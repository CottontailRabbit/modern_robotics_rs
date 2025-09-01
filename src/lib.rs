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
        let omega_hat = so3_mat / theta;
        let term1 = Matrix3::identity() * theta;
        let term2 = omega_hat * (1.0 - theta.cos());
        let term3 = omega_hat * omega_hat * (theta - theta.sin());
        let g_inv = (term1 + term2 + term3).try_inverse().unwrap();
        se3_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_mat);
        se3_mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&(g_inv * p));
    }
    se3_mat
}

/// ======================================================================
/// fkin_body  (MR: FKinBody, Product of Exponentials in Body frame)
/// ----------------------------------------------------------------------
/// EN:
/// Forward kinematics using body twists: T(θ) = M ⋅ ∏ₖ exp([Bₖ] θₖ).
/// (Note the right-multiplication order.)
///
/// KR:
/// 바디 프레임 트위스트를 사용한 정기구학:
/// T(θ) = M ⋅ ∏ₖ exp([Bₖ] θₖ).
/// (우측 곱 순서에 유의)
pub fn fkin_body(m: &Matrix4<f64>, b_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
    let mut t = m.clone_owned();
    for (i, b) in b_list.iter().enumerate() {
        let se3_mat = vec_to_se3(&(b * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);
    }
    t
}

/// ======================================================================
/// fkin_space  (MR: FKinSpace, Product of Exponentials in Space frame)
/// ----------------------------------------------------------------------
/// EN:
/// Forward kinematics using space twists: T(θ) = (∏ₖ exp([Sₖ] θₖ)) ⋅ M.
/// (Note the left-multiplication order, reverse loop helpful.)
///
/// KR:
/// 스페이스 프레임 트위스트를 사용한 정기구학:
/// T(θ) = (∏ₖ exp([Sₖ] θₖ)) ⋅ M.
/// (좌측 곱, 역순 반복 구현이 편리)
pub fn fkin_space(m: &Matrix4<f64>, s_list: &[Vector6<f64>], theta_list: &[f64]) -> Matrix4<f64> {
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
    let mut jb = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::zeros_generic(nalgebra::U6, nalgebra::Dyn(n));
    let mut t = Matrix4::identity();

    for i in 0..n {
        let r = t.fixed_view::<3, 3>(0, 0).clone_owned();
        let p = t.fixed_view::<3, 1>(0, 3).clone_owned();
        let p_skew = vec_to_skew3(&p);

        let mut adj_t = nalgebra::Matrix6::zeros();
        adj_t.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        adj_t.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        adj_t.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r));

        jb.set_column(i, &(adj_t * b_list[i]));

        // Update transform for next iteration (after setting current column)
        let se3_mat = vec_to_se3(&(-b_list[i] * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);
    }
    jb
}

/// ======================================================================
/// jacobian_space  (MR: JacobianSpace)
/// ----------------------------------------------------------------------
/// EN:
/// Space Jacobian at θ. Uses Ad(T⁻¹) of cumulative forward transform
/// ∏ exp([Sᵢ] θᵢ).
///
/// KR:
/// 구성 θ에서의 스페이스 자코비안. 누적 순방향 변환 ∏ exp([Sᵢ] θᵢ)의
/// T⁻¹에 대한 어드조인트를 사용합니다.
pub fn jacobian_space(
    s_list: &[Vector6<f64>],
    theta_list: &[f64],
) -> nalgebra::Matrix<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>> {
    let n = s_list.len();
    let mut js = nalgebra::Matrix::<f64, nalgebra::U6, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::U6, nalgebra::Dyn>>::zeros_generic(nalgebra::U6, nalgebra::Dyn(n));
    let mut t = Matrix4::identity();

    for i in 0..n {
        let t_inv   = t.try_inverse().unwrap();
        let r_inv   = t_inv.fixed_view::<3, 3>(0, 0).clone_owned();
        let p_inv   = t_inv.fixed_view::<3, 1>(0, 3).clone_owned();
        let p_skew  = vec_to_skew3(&p_inv);

        let mut adj_t_inv = nalgebra::Matrix6::zeros();
        adj_t_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_inv);
        adj_t_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&r_inv);
        adj_t_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&(p_skew * r_inv));

        js.set_column(i, &(adj_t_inv * s_list[i]));

        // Update transform for next iteration (after setting current column)
        let se3_mat = vec_to_se3(&(s_list[i] * theta_list[i]));
        t = t * matrix_exp6(&se3_mat);
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

/// ======================================================================
/// dls_step
/// ----------------------------------------------------------------------
/// EN:
/// Damped least-squares step:
///   Δθ = Jᵀ (J Jᵀ + λ² I)^{-1} V
/// Uses Cholesky when possible (SPD), falls back to LU.
/// KR:
/// 감쇠 최소자승 스텝:
///   Δθ = Jᵀ (J Jᵀ + λ² I)^{-1} V
/// SPD일 때 Cholesky, 아니면 LU로 풉니다.
fn dls_step(j: &Mat6xX, v: &Vector6<f64>, lambda: f64) -> Vec<f64> {
    // Δθ = Jᵀ (J Jᵀ + λ² I)⁻¹ V
    let jj_t: Matrix6<f64> = j * j.transpose();      
    let a = jj_t + Matrix6::identity() * (lambda * lambda);

    let y = if let Some(cho) = a.cholesky() {
        cho.solve(v)
    } else {
        a.lu().solve(v).expect("DLS: solve failed")
    };

    let jt = j.transpose();                           
    let delta = jt * y;                               
    delta.iter().copied().collect()
}

/// ======================================================================
/// twist_converged
/// ----------------------------------------------------------------------
/// EN:
/// Tests convergence with separate angular (‖ω‖) and linear (‖v‖) tolerances.
/// KR:
/// 각속/선속 오차 노름을 분리 임계값으로 검사.
fn twist_converged(v: &Vector6<f64>, eomg: f64, ev: f64) -> bool {
    let omg = Vector3::new(v[0], v[1], v[2]).norm();
    let lin = Vector3::new(v[3], v[4], v[5]).norm();
    omg < eomg && lin < ev
}

/// ======================================================================
/// ikin_body  (MR: IKinBody; Damped Newton / LM-style)
/// ----------------------------------------------------------------------
/// EN:
/// Inverse kinematics in the body frame.
/// Iterate:
///   V_b = se3ToVec( log( T(θ)^{-1} T_d ) )
///   Δθ  = J_bᵀ (J_b J_bᵀ + λ² I)^{-1} V_b
///   θ  ← θ + α Δθ     (with simple backtracking on α)
/// Converge when ‖ω_err‖ < eomg and ‖v_err‖ < ev.
/// Returns: (θ, success, iters)
///
/// KR:
/// 바디 프레임 역기구학.
/// 반복:
///   V_b = se3ToVec( log( T(θ)^{-1} T_d ) )
///   Δθ  = J_bᵀ (J_b J_bᵀ + λ² I)^{-1} V_b
///   θ  ← θ + α Δθ     (α는 간단 백트래킹)
/// 수렴: ‖ω_err‖ < eomg AND ‖v_err‖ < ev.
/// 반환: (θ, 성공여부, 반복횟수)
pub fn ikin_body(
    m: &Matrix4<f64>,
    b_list: &[Vector6<f64>],
    t_goal: &Matrix4<f64>,
    theta0: &[f64],
    eomg: f64,
    ev: f64,
    max_iter: usize,
) -> (Vec<f64>, bool, usize) {
    let lambda = 1e-6;
    let mut theta = theta0.to_vec();

    // 현재 에러 노름을 계산하는 헬퍼
    let err_norms = |th: &Vec<f64>| -> (f64, f64) {
        let t_now = fkin_body(m, b_list, th);
        let t_err = trans_inv(&t_now) * t_goal;
        let se3_err = matrix_log6(&t_err);
        let v_b = se3_to_vec(&se3_err);
        (
            Vector3::new(v_b[0], v_b[1], v_b[2]).norm(),
            Vector3::new(v_b[3], v_b[4], v_b[5]).norm(),
        )
    };

    for it in 0..max_iter {
        // 에러 트위스트 (Body)
        let t_now = fkin_body(m, b_list, &theta);
        let t_err = trans_inv(&t_now) * t_goal;
        let se3_err = matrix_log6(&t_err);
        let v_b = se3_to_vec(&se3_err);
        if twist_converged(&v_b, eomg, ev) {
            return (theta, true, it);
        }

        // 자코비안 & DLS 스텝
        let jb = jacobian_body(b_list, &theta); // 6×n
        let dtheta = dls_step(&jb, &v_b, lambda);

        // 간단한 백트래킹 라인서치 (α ∈ {1.0, 0.5, 0.25, 0.1})
        let (omg0, v0) = (
            Vector3::new(v_b[0], v_b[1], v_b[2]).norm(),
            Vector3::new(v_b[3], v_b[4], v_b[5]).norm(),
        );
        let mut accepted = false;
        for alpha in [1.0, 0.5, 0.25, 0.1] {
            let mut th_try = theta.clone();
            for i in 0..th_try.len() {
                th_try[i] += alpha * dtheta[i];
            }
            let (omg1, v1) = err_norms(&th_try);
            if omg1 <= omg0 && v1 <= v0 {
                theta = th_try;
                accepted = true;
                break;
            }
        }
        if !accepted {
            // 그래도 진행 (스텝 매우 작거나 수렴 직전일 수 있음)
            for i in 0..theta.len() {
                theta[i] += 0.1 * dtheta[i];
            }
        }
    }
    (theta, false, max_iter)
}

/// ======================================================================
/// ikin_space  (MR: IKinSpace; Damped Newton / LM-style)
/// ----------------------------------------------------------------------
/// EN:
/// Inverse kinematics in the space frame.
/// Iterate:
///   V_s = se3ToVec( log( T_d T(θ)^{-1} ) )
///   Δθ  = J_sᵀ (J_s J_sᵀ + λ² I)^{-1} V_s
///   θ  ← θ + α Δθ
/// Converge when ‖ω_err‖ < eomg and ‖v_err‖ < ev.
/// Returns: (θ, success, iters)
///
/// KR:
/// 스페이스 프레임 역기구학.
/// 반복:
///   V_s = se3ToVec( log( T_d T(θ)^{-1} ) )
///   Δθ  = J_sᵀ (J_s J_sᵀ + λ² I)^{-1} V_s
///   θ  ← θ + α Δθ
/// 수렴: ‖ω_err‖ < eomg AND ‖v_err‖ < ev.
/// 반환: (θ, 성공여부, 반복횟수)
pub fn ikin_space(
    m: &Matrix4<f64>,
    s_list: &[Vector6<f64>],
    t_goal: &Matrix4<f64>,
    theta0: &[f64],
    eomg: f64,
    ev: f64,
    max_iter: usize,
) -> (Vec<f64>, bool, usize) {
    let lambda = 1e-6;
    let mut theta = theta0.to_vec();

    let err_norms = |th: &Vec<f64>| -> (f64, f64) {
        let t_now = fkin_space(m, s_list, th);
        let t_err = t_goal * trans_inv(&t_now);
        let se3_err = matrix_log6(&t_err);
        let v_s = se3_to_vec(&se3_err);
        (
            Vector3::new(v_s[0], v_s[1], v_s[2]).norm(),
            Vector3::new(v_s[3], v_s[4], v_s[5]).norm(),
        )
    };

    for it in 0..max_iter {
        let t_now = fkin_space(m, s_list, &theta);
        let t_err = t_goal * trans_inv(&t_now);
        let se3_err = matrix_log6(&t_err);
        let v_s = se3_to_vec(&se3_err);
        if twist_converged(&v_s, eomg, ev) {
            return (theta, true, it);
        }

        let js = jacobian_space(s_list, &theta); // 6×n
        let dtheta = dls_step(&js, &v_s, lambda);

        // 간단 백트래킹
        let (omg0, v0) = (
            Vector3::new(v_s[0], v_s[1], v_s[2]).norm(),
            Vector3::new(v_s[3], v_s[4], v_s[5]).norm(),
        );
        let mut accepted = false;
        for alpha in [1.0, 0.5, 0.25, 0.1] {
            let mut th_try = theta.clone();
            for i in 0..th_try.len() {
                th_try[i] += alpha * dtheta[i];
            }
            let (omg1, v1) = err_norms(&th_try);
            if omg1 <= omg0 && v1 <= v0 {
                theta = th_try;
                accepted = true;
                break;
            }
        }
        if !accepted {
            for i in 0..theta.len() {
                theta[i] += 0.1 * dtheta[i];
            }
        }
    }
    (theta, false, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::vector;

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
    fn test_jacobian_with_translation() {
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
}
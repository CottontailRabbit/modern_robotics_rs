# Modern Robotics Rust

This repository is a Rust port of the essential functions from the [Modern Robotics: Mechanics, Planning, and Control (C++ version)](https://github.com/Le0nX/ModernRoboticsCpp) library.

The goal is to provide a robust and idiomatic Rust implementation of key robotics algorithms, leveraging Rust's safety and performance features.

## Implemented Features

Currently, the following core functionalities have been ported:

*   **Mathematical Utilities:**
    *   `vec_to_skew3`: Converts a 3-vector to a 3x3 skew-symmetric matrix.
    *   `skew3_to_vec`: Converts a 3x3 skew-symmetric matrix to a 3-vector.
    *   `vec_to_se3`: Converts a 6-vector (twist) to a 4x4 se3 matrix.
    *   `se3_to_vec`: Converts a 4x4 se3 matrix to a 6-vector (twist).

*   **Exponential Coordinates:**
    *   `matrix_exp3`: Computes the matrix exponential of a 3x3 skew-symmetric matrix (so3).
    *   `matrix_log3`: Computes the matrix logarithm of a 3x3 rotation matrix (SO3).
    *   `matrix_exp6`: Computes the matrix exponential of a 4x4 se3 matrix (twist).
    *   `matrix_log6`: Computes the matrix logarithm of a 4x4 homogeneous transformation matrix (SE3).

*   **Kinematics:**
    *   `fkin_body`: Computes the forward kinematics in the body frame.
    *   `fkin_space`: Computes the forward kinematics in the space frame.
    *   `jacobian_body`: Computes the body Jacobian.
    *   `jacobian_space`: Computes the space Jacobian.

## Usage

To use this library in your Rust project, add it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
modern_robotics_rs = { git = "https://github.com/CottontailRabbit/modern_robotics_rs.git" }
nalgebra = "0.32"
approx = "0.5"
```

Then, you can use the functions in your Rust code:

```rust
use modern_robotics_rs::*;
use nalgebra::{vector, matrix};

fn main() {
    let so3_mat = matrix![
        0.0, -1.0,  0.0;
        1.0,  0.0,  0.0;
        0.0,  0.0,  0.0;
    ];
    let exp_mat = matrix_exp3(&so3_mat);
    println!("Exponential of so3 matrix:\n{:?}", exp_mat);

    let b_list = [
        vector![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vector![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ];
    let theta_list = [std::f64::consts::PI / 2.0, 0.0];
    let jb = jacobian_body(&b_list, &theta_list);
    println!("Body Jacobian:\n{:?}", jb);
}
```

## Building and Testing

To build the library:

```bash
cargo build
```

To run the tests:

```bash
cargo test
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License, consistent with the original Modern Robotics C++ library.

## Original Project

This project is a port of the [Modern Robotics: Mechanics, Planning, and Control (C++ version)](https://github.com/Le0nX/ModernRoboticsCpp).

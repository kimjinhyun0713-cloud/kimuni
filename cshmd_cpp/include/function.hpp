#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline std::tuple<std::array<double, 3>, std::array<double, 3>, std::array<double, 3>>
calLattice(const std::vector<std::vector<double>>& arr) {
    if (arr.size() < 3) {
        throw std::invalid_argument("Array must have at least 3 rows.");
    }

    double xlo = 0.0, xhi = 0.0, xy = 0.0;
    double ylo = 0.0, yhi = 0.0, xz = 0.0;
    double zlo = 0.0, zhi = 0.0, yz = 0.0;

    // Pythonの arr.shape[1] == 3 or 2 の判定に相当
    if (arr[0].size() >= 3) {
        xlo = arr[0][0]; xhi = arr[0][1]; xy = arr[0][2];
        ylo = arr[1][0]; yhi = arr[1][1]; xz = arr[1][2];
        zlo = arr[2][0]; zhi = arr[2][1]; yz = arr[2][2];
    } else if (arr[0].size() == 2) {
        xlo = arr[0][0]; xhi = arr[0][1];
        ylo = arr[1][0]; yhi = arr[1][1];
        zlo = arr[2][0]; zhi = arr[2][1];
        // xy, xz, yz は 0.0 のまま
    } else {
        throw std::invalid_argument("Array columns must be 2 or 3.");
    }

    // np.max / np.min の C++ 代替
    xhi -= std::max({0.0, xy, xz, xy + xz});
    xlo -= std::min({0.0, xy, xz, xy + xz});
    ylo -= std::min({0.0, yz});
    yhi -= std::max({0.0, yz});

    double x_length = xhi - xlo;
    double y_length = yhi - ylo;
    double z_length = zhi - zlo;

    // 規定ベクトルの作成
    std::array<double, 3> a_vec = {x_length, 0.0, 0.0};
    std::array<double, 3> b_vec = {xy, y_length, 0.0};
    std::array<double, 3> c_vec = {xz, yz, z_length};

    // ノルム（長さ）と内積を計算するラムダ関数
    auto norm = [](const std::array<double, 3>& v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };
    auto dot = [](const std::array<double, 3>& v1, const std::array<double, 3>& v2) {
        return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    };

    double a = norm(a_vec);
    double b = norm(b_vec);
    double c = norm(c_vec);

    // 浮動小数点誤差による acos(1.00000000001) の NaN を防ぐための安全措置
    auto safe_acos = [](double val) {
        return std::acos(std::clamp(val, -1.0, 1.0));
    };

    // 角度の計算 (ラジアン -> 度数法)
    double alpha = safe_acos(dot(b_vec, c_vec) / (b * c)) * 180.0 / M_PI;
    double beta  = safe_acos(dot(a_vec, c_vec) / (a * c)) * 180.0 / M_PI;
    double gamma = safe_acos(dot(a_vec, b_vec) / (a * b)) * 180.0 / M_PI;

    // 戻り値の構築
    std::array<double, 3> lattice = {a, b, c};
    std::array<double, 3> angle = {alpha, beta, gamma};
    std::array<double, 3> zeropoint = {xlo, ylo, zlo};

    return {lattice, angle, zeropoint};
}
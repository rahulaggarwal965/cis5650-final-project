/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_ddc, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_ddirect_color = dL_ddc + idx;
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[0] = dRGBdsh1 * dL_dRGB;
		dL_dsh[1] = dRGBdsh2 * dL_dRGB;
		dL_dsh[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[2];
		dRGBdy = -SH_C1 * sh[0];
		dRGBdz = SH_C1 * sh[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[3] = dRGBdsh4 * dL_dRGB;
			dL_dsh[4] = dRGBdsh5 * dL_dRGB;
			dL_dsh[5] = dRGBdsh6 * dL_dRGB;
			dL_dsh[6] = dRGBdsh7 * dL_dRGB;
			dL_dsh[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
			dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
			dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[8] = dRGBdsh9 * dL_dRGB;
				dL_dsh[9] = dRGBdsh10 * dL_dRGB;
				dL_dsh[10] = dRGBdsh11 * dL_dRGB;
				dL_dsh[11] = dRGBdsh12 * dL_dRGB;
				dL_dsh[12] = dRGBdsh13 * dL_dRGB;
				dL_dsh[13] = dRGBdsh14 * dL_dRGB;
				dL_dsh[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[8] * 3.f * 2.f * xy +
					SH_C3[1] * sh[9] * yz +
					SH_C3[2] * sh[10] * -2.f * xy +
					SH_C3[3] * sh[11] * -3.f * 2.f * xz +
					SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[13] * 2.f * xz +
					SH_C3[6] * sh[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[8] * 3.f * (xx - yy) +
					SH_C3[1] * sh[9] * xz +
					SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[11] * -3.f * 2.f * yz +
					SH_C3[4] * sh[12] * -2.f * xy +
					SH_C3[5] * sh[13] * -2.f * yz +
					SH_C3[6] * sh[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[9] * xy +
					SH_C3[2] * sh[10] * 4.f * 2.f * yz +
					SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[12] * 4.f * 2.f * xz +
					SH_C3[5] * sh[13] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	
	const float x_grad_mul = 1;
	const float y_grad_mul = 1;
	const float z_grad_mul = 1;

	float dis_inv = 1.0f / (sqrt(t.x * t.x + t.y * t.y + t.z * t.z) + 0.0000001f);

	const float fx = max(512.f, h_x);
	const float fy = max(512.f, h_y);

	float3 mu = { t.x * dis_inv, t.y * dis_inv, t.z * dis_inv};

	float mut_xyz = mu.x * t.x + mu.y * t.y + mu.z * t.z;
	float mut_xyz2 = mut_xyz * mut_xyz;
	float mut_xyz2_inv = 1.0f / (mut_xyz2 + 0.0000001f);

	float theta = atan2(-mu.y, sqrt(mu.x * mu.x + mu.z * mu.z)); 
	float phi = atan2(mu.x, mu.z);
	
	float sin_phi = sin(phi);
	float cos_phi = cos(phi);

	float sin_theta = sin(theta);
	float cos_theta = cos(theta);

	glm::mat3 J = glm::mat3(
		fx * (
			(mu.x * t.z * sin_phi + mu.y * t.y * cos_phi + mu.z * t.z * cos_phi) * mut_xyz2_inv
		),
		fx * (
			(mu.y * (-t.x * cos_phi + t.z * sin_phi)) * mut_xyz2_inv
		),
		fx * (
			-(mu.x * t.x * sin_phi + mu.y * t.y * sin_phi + mu.z * t.x * cos_phi) * mut_xyz2_inv
		),

		fy * (
			(-mu.x * t.y * cos_theta - mu.x * t.z * sin_theta * cos_phi + mu.y * t.y * sin_phi * sin_theta + mu.z * t.z * sin_phi * sin_theta) * mut_xyz2_inv
		),
		fy * (
			(mu.x * t.x * cos_theta - mu.y * t.x * sin_phi * sin_theta - mu.y * t.z * sin_theta * cos_phi + mu.z * t.z * cos_theta) * mut_xyz2_inv
		),
		fy * (
			(mu.x * t.x * sin_theta * cos_phi + mu.y * t.y * sin_theta * cos_phi - mu.z * t.x * sin_phi * sin_theta - mu.z * t.y * cos_theta) * mut_xyz2_inv
		),

		0.0f,
		0.0f,
		0.0f
	);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float dL_dJ01 = W[1][0] * dL_dT00 + W[1][1] * dL_dT01 + W[1][2] * dL_dT02;
 	float dL_dJ10 = W[0][0] * dL_dT10 + W[0][1] * dL_dT11 + W[0][2] * dL_dT12;

	float x2 = t.x * t.x;
	float y2 = t.y * t.y;
	float z2 = t.z * t.z;


	float j00_x = fx * t.x * t.z * (-2.f * x2 - y2 - 2.f * z2) 
			/ pow(x2 + z2, 1.5f) / pow(x2 + y2 + z2, 1.5f);
	float j01_x = 0.0f;
	float j02_x = fx * (x2 * x2 - y2 * z2 - z2 * z2) 
				/ pow(x2 + z2, 0.5f) / pow(x2 + y2 + z2, 0.5f) / (x2 * x2 + x2 * y2 + 2.0f * x2 * z2 + y2 * z2 + z2 * z2);
	float j10_x = fy * t.y * (2.f * x2 * x2 + x2 * z2 - y2 * z2 - z2 * z2) 
					/ pow(x2 + z2, 0.5f) / (x2 * x2 * x2 + 2.f * x2 * x2 * y2 + 3. * x2 * x2 * z2 + x2 * y2 * y2 + 4.f * x2 * y2 * z2 + 3.f * x2 * z2 * z2 + y2 * y2 * z2 + 2.f * y2 * z2 * z2 + z2 * z2 * z2);
	float j11_x = fy * t.x * (-x2 + y2 - z2) 
					/ pow(x2 + z2, 0.5f) / (x2 * x2 + 2.f * x2 * y2 + 2.f * x2 * z2 + y2 * y2 + 2.f * y2 * z2 + z2 * z2);
	float j12_x = fy * t.x * t.y * t.z * (2.f * y2 * (x2 * z2) + y2 * (x2 + y2 + z2) + 2.f * pow(x2 + z2, 2.f) + (x2 + z2) * (x2 + y2 + z2)) 
				/ pow(x2 + z2, 1.5f) / pow(x2 + y2 + z2, 3.f);

	float j00_y = -fx * t.y * t.z 
				/ pow(x2 + z2, 0.5f) / pow(x2 + y2 + z2, 1.5f);
	float j01_y = 0.0f;
	float j02_y = fx * t.x * t.y 
				/ pow(x2 + z2, 0.5f) / pow(x2 + y2 + z2, 1.5f);
	float j10_y = fy * t.x * (-x2 + y2 - z2) 
					/ pow(x2 + z2, 0.5f) / (x2 * x2 + 2.f * x2 * y2 + 2.f * x2 * z2 + y2 * y2 + 2.f * y2 * z2 + z2 * z2);
	float j11_y = -2.f * fy * t.y * pow(x2 + z2, 0.5f) /
					(x2 * x2 + 2.f * x2 * y2 + 2.f * x2 * z2 + y2 * y2 + 2.f * y2 * z2 + z2 * z2);
	float j12_y = fy * t.z * (-x2 + y2 - z2) 
					/ pow(x2 + z2, 0.5f) / (x2 * x2 + 2.f * x2 * y2 + 2.f * x2 * z2 + y2 * y2 + 2.f * y2 * z2 + z2 * z2);
	
	float j00_z = fx * (x2 * x2 + x2 * y2 - z2 * z2) 
				/ pow(x2 + z2, 0.5f) / pow(x2 + y2 + z2, 0.5f) / (x2 * x2 + x2 * y2 + 2.f * x2 * z2 + y2 * z2 + z2 * z2);
	float j01_z = 0.0f;
	float j02_z = fx * t.x * t.z * (2.f * x2 + y2 + 2.f * z2) 
			/ pow(x2 + z2, 1.5f) / pow(x2 + y2 + z2, 1.5f);
	float j10_z = fy * t.x * t.y * t.z * (2.f * y2 * (x2 + z2) + y2 * (x2 + y2 + z2) + 2.f * pow(x2 + z2, 2.f) + (x2 + z2) * (x2 + y2 + z2)) 
				/ pow(x2 + z2, 1.5f) / pow(x2 + y2 + z2, 3.f);
	float j11_z = fy * t.z * (-x2 + y2 - z2) 
				/ pow(x2 + z2, 0.5f) / (x2 * x2 + 2.f * x2 * y2 + 2.f * x2 * z2 + y2 * y2 + 2.f * y2 * z2 + z2 * z2);
	float j12_z = fy * t.y * (-x2 * x2 - x2 * y2 + x2 * z2 + 2.f * z2 * z2)
				/ pow(x2 + z2, 0.5f) / (x2 * x2 * x2 + 2.f * x2 * x2 * y2 + 3.f * x2 * x2 * z2 + x2 * y2 * y2 + 4.f * x2 * y2 * z2 + 3.f * x2 * z2 * z2 + y2 * y2 * z2 + 2.f * y2 * z2 * z2 + z2 * z2 * z2);


	float dL_dtx = x_grad_mul * (j00_x * dL_dJ00 + j01_x * dL_dJ01 + j02_x * dL_dJ02 + j10_x * dL_dJ10 + j11_x * dL_dJ11 + j12_x * dL_dJ12);
	float dL_dty = y_grad_mul * (j00_y * dL_dJ00 + j01_y * dL_dJ01 + j02_y * dL_dJ02 + j10_y * dL_dJ10 + j11_y * dL_dJ11 + j12_y * dL_dJ12);
	float dL_dtz = z_grad_mul * (j00_z * dL_dJ00 + j01_z * dL_dJ01 + j02_z * dL_dJ02 + j10_z * dL_dJ10 + j11_z * dL_dJ11 + j12_z * dL_dJ12);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, dc, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_ddc, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template<uint32_t C>
__global__ void
PerGaussianRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	const uint32_t* __restrict__ bucket_to_tile,
	const float* __restrict__ sampled_T, const float* __restrict__ sampled_ar,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float focal_x, float focal_y
) {
	// global_bucket_idx = warp_idx
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	// What is the number of buckets before me? what is my offset?
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
	float c[C] = {0.0f};
	if (valid_splat) {
		gaussian_idx = point_list[splat_idx_global];
		xy = points_xy_image[gaussian_idx];
		con_o = conic_opacity[gaussian_idx];
		for (int ch = 0; ch < C; ++ch)
			c[ch] = colors[gaussian_idx * C + ch];
	}

	// Gradient accumulation variables
	float Register_dL_dmean2D_x = 0.0f;
	float Register_dL_dmean2D_y = 0.0f;
	float Register_dL_dconic2D_x = 0.0f;
	float Register_dL_dconic2D_y = 0.0f;
	float Register_dL_dconic2D_w = 0.0f;
	float Register_dL_dopacity = 0.0f;
	float Register_dL_dcolors[C] = {0.0f};
	
	// tile metadata
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

	// values useful for gradient calculation
	float T;
	float T_final;
	float last_contributor;
	float ar[C];
	float dL_dpixel[C];
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// const float cx = 0.5 * W - 0.5;
	// const float cy = 0.5 * H - 0.5;
	// const float inv_x = 1.0f / focal_x;
	// const float inv_y = 1.0f / focal_y;
	// // const float f_x = focal_x;
	// // const float f_y = focal_y;
	// const float hw = 0.5 * W;
	// const float hh = 0.5 * H;
	// const float tan_fovx = hw * inv_x;
	// const float tan_fovy = hh * inv_y;

	// float3 mu = {
	// 	(xy.x - cx) * inv_x,
	// 	(xy.y - cy) * inv_y,
	// 	1
	// };

	// float theta = atan2(-mu.y, sqrt(mu.x * mu.x + mu.z * mu.z)); 
	// float phi = atan2(mu.x, mu.z);
	
	// float sin_phi = sin(phi);
	// float cos_phi = cos(phi);

	// float sin_theta = sin(theta);
	// float cos_theta = cos(theta);

	// mu = {
	// 	cos_theta * sin_phi,
	// 	-sin_theta,
	// 	cos_theta * cos_phi
	// };

	// iterate over all pixels in the tile
	for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
		// SHUFFLING

		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		T_final = my_warp.shfl_up(T_final, 1);
		for (int ch = 0; ch < C; ++ch) {
			ar[ch] = my_warp.shfl_up(ar[ch], 1);
			dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
		}

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
		const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
		const uint32_t pix_id = W * pix.y + pix.x;
		const float2 pixf = {(float) pix.x, (float) pix.y};
		bool valid_pixel = pix.x < W && pix.y < H;
		
		// every 32nd thread should read the stored state from memory
		// TODO: perhaps store these things in shared memory?
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
			T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
			T_final = final_Ts[pix_id];
			for (int ch = 0; ch < C; ++ch)
				ar[ch] = -(pixel_colors[ch * H * W + pix_id] - T_final * bg_color[ch]) + sampled_ar[global_bucket_idx * BLOCK_SIZE * C + ch * BLOCK_SIZE + idx];
			last_contributor = n_contrib[pix_id];
			for (int ch = 0; ch < C; ++ch) {
				dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
			}
		}

		// do work
		if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
			if (W <= pix.x || H <= pix.y) continue;

			if (splat_idx_in_tile >= last_contributor) continue;

			// // add for min error
			// float3 t = {
			// 	(pixf.x - cx) * inv_x,
			// 	(pixf.y - cy) * inv_y,
			// 	1
			// };

			// theta = atan2(-t.y, sqrt(t.x * t.x + t.z * t.z)); 
			// phi = atan2(t.x, t.z);
			
			// sin_phi = sin(phi);
			// cos_phi = cos(phi);

			// sin_theta = sin(theta);
			// cos_theta = cos(theta);

			// t = {
			// 	cos_theta * sin_phi,
			// 	-sin_theta,
			// 	cos_theta * cos_phi
			// };

			// // NOTE(rahul): we early out here because the gaussian projection and the pixel ray
			// // are orthogonal. Do I need to specially handle gradients here?
			// if (mu.x * t.x + mu.y * t.y + mu.z * t.z < 0.0000001f)
			// {
			// 	continue;
			// }

			// float u_xy = 0.0f;
			// float v_xy = 0.0f;

			// const float uv_pixf = (mu.x * t.x + mu.y * t.y + mu.z * t.z);
			// const float uv_pixf_inv = 1.f / uv_pixf;

			// float u_pixf = max(512.f, focal_x) * (t.x * cos_phi - t.z * sin_phi) * uv_pixf_inv;
			// float v_pixf = max(512.f, focal_y) * (t.x * sin_phi * sin_theta + t.y * cos_theta + t.z * sin_theta * cos_phi) * uv_pixf_inv;

			// const float2 d = { u_xy - u_pixf, v_xy - v_pixf }; 
			// compute blending values
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;
			const float dchannel_dcolor = alpha * T;

			// add the gradient contribution of this pixel to the gaussian
			float bg_dot_dpixel = 0.0f;
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ++ch) {
				ar[ch] += T * alpha * c[ch];
				const float &dL_dchannel = dL_dpixel[ch];
				Register_dL_dcolors[ch] += dchannel_dcolor * dL_dchannel;
				dL_dalpha += ((c[ch] * T) - (1.0f / (1.0f - alpha)) * (-ar[ch])) * dL_dchannel;

				bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
			}
			dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;
			T *= (1.0f - alpha);


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// const float x = (xy.x - cx) * inv_x;
			// const float y = (xy.y - cy) * inv_y;

			// const float tx = t.x;
			// const float ty = t.y;

			// const float tx2 = tx * tx;
			// const float ty2 = ty * ty;
			// const float x2 = x * x;
			// const float y2 = y * y;

			// const float x2_1 = x2 + 1;
			// const float x2_y2_1 = x2_1 + y * y;
			// const float txx_1 = tx * x + 1;
			// const float txx_tyy_1 = txx_1 + ty * y;
			// const float tmp1 = - tx * (x2_y2_1) + x * (txx_tyy_1);

			// const float x2_1_pow1_5 = pow(x2_1, 1.5);
			// const float txx_tyy_1_pow2 = pow(txx_tyy_1, 2);
			// const float x2_y2_1_pow0_5 = sqrt(x2_y2_1);
			// const float x2_1_pow0_5 = sqrt(x2_1);

			// const float inv_x2_1_pow1_5 = 1.0f / x2_1_pow1_5;
			// const float inv_x2_1_pow0_5 = 1.0f / x2_1_pow0_5;
			// const float inv_txx_tyy_1_pow2 = 1.0f / txx_tyy_1_pow2;
			// const float inv_x2_y2_1_pow0_5 = 1.0f / x2_y2_1_pow0_5;
			

			// const float ddelx_dx = focal_x * (- (tx - x) * (x2_1) * (tmp1) + (x2_y2_1) * (txx_1) * (txx_tyy_1))
			// 						* inv_x2_1_pow1_5 * inv_txx_tyy_1_pow2 * inv_x2_y2_1_pow0_5;
			// const float ddely_dx = focal_y * ((x2_1) * (tmp1) * (- ty * (x2_1) + y * (txx_1)) - (txx_tyy_1) * (- ty * x * pow(x2_1, 2) + x * y * (x2_1) * (txx_1) + x * y * (txx_1) * (x2_y2_1) + (x2_1) * (- tx * y + ty * x) * (x2_y2_1))) 
			// 						* inv_x2_1_pow1_5 * inv_txx_tyy_1_pow2 / ((x2_y2_1));
			// const float ddelx_dy = focal_x * (tx - x) * (ty * (x2_y2_1) - y * (txx_tyy_1)) 
			// 						* inv_x2_1_pow0_5 * inv_txx_tyy_1_pow2 * inv_x2_y2_1_pow0_5;
			// const float ddely_dy = focal_y * (tx2 * x2 + 2 * tx * x + ty2 * x2 + ty2 + 1) 
			// 						* inv_x2_1_pow0_5 / ((tx2 * x2 + 2 * tx * ty * x * y + 2 * tx * x + ty2 * y2 + 2 * ty * y + 1));

			// accumulate the gradients
			const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
			// const float tmp_x = dL_dG * (dG_ddelx * ddelx_dx + dG_ddely * ddely_dx) * tan_fovx;
			Register_dL_dmean2D_x += tmp_x;
			const float tmp_y = dL_dG * dG_ddely * ddely_dy;
			// const float tmp_y = dL_dG * (dG_ddelx * ddelx_dy + dG_ddely * ddely_dy) * tan_fovy;
			Register_dL_dmean2D_y += tmp_y;

			Register_dL_dconic2D_x += -0.5f * gdx * d.x * dL_dG;
			Register_dL_dconic2D_y += -0.5f * gdx * d.y * dL_dG;
			Register_dL_dconic2D_w += -0.5f * gdy * d.y * dL_dG;
			Register_dL_dopacity += G * dL_dalpha;
		}
	}

	// finally add the gradients using atomics
	if (valid_splat) {
		atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
		atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].x, Register_dL_dconic2D_x);
		atomicAdd(&dL_dconic2D[gaussian_idx].y, Register_dL_dconic2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].w, Register_dL_dconic2D_w);
		atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
		for (int ch = 0; ch < C; ++ch) {
			atomicAdd(&dL_dcolors[gaussian_idx * C + ch], Register_dL_dcolors[ch]);
		}
	}
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float focal_x, float focal_y)
{
	const bool use_atomic = false;

	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float cx = 0.5 * W - 0.5;
	const float cy = 0.5 * H - 0.5;
	const float inv_x = 1.0f / focal_x;
	const float inv_y = 1.0f / focal_y;
	const float f_x = focal_x;
	const float f_y = focal_y;
	const float hw = 0.5 * W;
	const float hh = 0.5 * H;
	const float tan_fovx = hw * inv_x;
	const float tan_fovy = hh * inv_y;

	// add for min error
	float3 t = {
		(pixf.x - cx) * inv_x,
		(pixf.y - cy) * inv_y,
		1
	};

	float theta = atan2(-t.y, sqrt(t.x * t.x + t.z * t.z)); 
	float phi = atan2(t.x, t.z);
			
	float sin_phi = sin(phi);
	float cos_phi = cos(phi);

	float sin_theta = sin(theta);
	float cos_theta = cos(theta);

	t = {
		cos_theta * sin_phi,
		-sin_theta,
		cos_theta * cos_phi
	};

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		static constexpr int REDUCTION_BATCH_SIZE = 16;
		int cur_reduction_batch_idx = 0;
		__shared__ int batch_j[REDUCTION_BATCH_SIZE];
		__shared__ float batch_dL_dcolors[REDUCTION_BATCH_SIZE][NUM_WARPS][C];
		__shared__ float2 batch_dL_dmean2D[REDUCTION_BATCH_SIZE][NUM_WARPS];
		__shared__ float4 batch_dL_dconic2D_dopacity[REDUCTION_BATCH_SIZE][NUM_WARPS];

		// Iterate over Gaussians
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			float cur_dL_dcolors[C] = { 0 };
			float2 cur_dL_dmean2D = { 0, 0 };
			float4 cur_dL_dconic2D_dopacity = { 0, 0, 0, 0 };

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (!done && contributor < last_contributor) {

				// Compute blending values, as before.
				const float2 xy = collected_xy[j];
				// const float2 d = { xy.x - pixf.x, xy.y - pixf.y };

				float3 mu = {
					(xy.x - cx) * inv_x,
					(xy.y - cy) * inv_y,
					1
				};

				theta = atan2(-mu.y, sqrt(mu.x * mu.x + mu.z * mu.z));
				phi = atan2(mu.x, mu.z);

				sin_phi = sin(phi);
				cos_phi = cos(phi);

				sin_theta = sin(theta);
				cos_theta = cos(theta);

				mu = {
					cos_theta * sin_phi,
					-sin_theta,
					cos_theta * cos_phi
				};

				float u_xy = 0.0f;
				float v_xy = 0.0f;

				const float uv_pixf = (mu.x * t.x + mu.y * t.y + mu.z * t.z);
				const float uv_pixf_inv = 1.f / uv_pixf;

				float u_pixf = max(512.f, focal_x) * (t.x * cos_phi - t.z * sin_phi) * uv_pixf_inv;
				float v_pixf = max(512.f, focal_y) * (t.x * sin_phi * sin_theta + t.y * cos_theta + t.z * sin_theta * cos_phi) * uv_pixf_inv;

				float2 d = { u_xy - u_pixf, v_xy - v_pixf };

				const float4 con_o = collected_conic_opacity[j];
				const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

				const float G = exp(power);
				const float alpha = min(0.99f, con_o.w * G);

				if (mu.x * t.x + mu.y * t.y + mu.z * t.z >= 0.0000001f && power <= 0.0f && alpha >= 1.0f / 255.0f) {

					T = T / (1.f - alpha);
					const float dchannel_dcolor = alpha * T;

					// Propagate gradients to per-Gaussian colors and keep
					// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
					// pair).
					float dL_dalpha = 0.0f;
					const int global_id = collected_id[j];
					for (int ch = 0; ch < C; ch++)
					{
						const float c = collected_colors[ch * BLOCK_SIZE + j];
						// Update last color (to be used in the next iteration)
						accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
						last_color[ch] = c;

						const float dL_dchannel = dL_dpixel[ch];
						dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
						// Update the gradients w.r.t. color of the Gaussian. 
						// Atomic, since this pixel is just one of potentially
						// many that were affected by this Gaussian.

						if (use_atomic) {
							atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
						}
						else {
							cur_dL_dcolors[ch] = dchannel_dcolor * dL_dchannel;
						}
					}
					dL_dalpha *= T;
					// Update last alpha (to be used in the next iteration)
					last_alpha = alpha;

					// Account for fact that alpha also influences how much of
					// the background color is added if nothing left to blend
					float bg_dot_dpixel = 0;
					for (int i = 0; i < C; i++)
						bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
					dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


					// Helpful reusable temporary variables
					const float dL_dG = con_o.w * dL_dalpha;
					const float gdx = G * d.x;
					const float gdy = G * d.y;
					const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
					const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

					if (con_o.w * G <= 0.99f) {

						const float x = (xy.x - cx) * inv_x;
						const float y = (xy.y - cy) * inv_y;

						const float tx = t.x;
						const float ty = t.y;

						const float tx2 = tx * tx;
						const float ty2 = ty * ty;
						const float x2 = x * x;
						const float y2 = y * y;

						const float x2_1 = x2 + 1;
						const float x2_y2_1 = x2_1 + y * y;
						const float txx_1 = tx * x + 1;
						const float txx_tyy_1 = txx_1 + ty * y;
						const float tmp1 = - tx * (x2_y2_1) + x * (txx_tyy_1);

						const float x2_1_pow1_5 = pow(x2_1, 1.5);
						const float txx_tyy_1_pow2 = pow(txx_tyy_1, 2);
						const float x2_y2_1_pow0_5 = sqrt(x2_y2_1);
						const float x2_1_pow0_5 = sqrt(x2_1);

						const float inv_x2_1_pow1_5 = 1.0f / x2_1_pow1_5;
						const float inv_x2_1_pow0_5 = 1.0f / x2_1_pow0_5;
						const float inv_txx_tyy_1_pow2 = 1.0f / txx_tyy_1_pow2;
						const float inv_x2_y2_1_pow0_5 = 1.0f / x2_y2_1_pow0_5;


						const float ddelx_dx = f_x * (- (tx - x) * (x2_1) * (tmp1) + (x2_y2_1) * (txx_1) * (txx_tyy_1))
												* inv_x2_1_pow1_5 * inv_txx_tyy_1_pow2 * inv_x2_y2_1_pow0_5;
						const float ddely_dx = f_y * ((x2_1) * (tmp1) * (- ty * (x2_1) + y * (txx_1)) - (txx_tyy_1) * (- ty * x * pow(x2_1, 2) + x * y * (x2_1) * (txx_1) + x * y * (txx_1) * (x2_y2_1) + (x2_1) * (- tx * y + ty * x) * (x2_y2_1)))
												* inv_x2_1_pow1_5 * inv_txx_tyy_1_pow2 / ((x2_y2_1));
						const float ddelx_dy = f_x * (tx - x) * (ty * (x2_y2_1) - y * (txx_tyy_1))
												* inv_x2_1_pow0_5 * inv_txx_tyy_1_pow2 * inv_x2_y2_1_pow0_5;
						const float ddely_dy = f_y * (tx2 * x2 + 2 * tx * x + ty2 * x2 + ty2 + 1)
												* inv_x2_1_pow0_5 / ((tx2 * x2 + 2 * tx * ty * x * y + 2 * tx * x + ty2 * y2 + 2 * ty * y + 1));



						// Update gradients w.r.t. 2D mean position of the Gaussian

						if (use_atomic) {
							// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
							// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
							atomicAdd(&dL_dmean2D[global_id].x, dL_dG * (dG_ddelx * ddelx_dx + dG_ddely * ddely_dx) * tan_fovx);
							atomicAdd(&dL_dmean2D[global_id].y, dL_dG * (dG_ddelx * ddelx_dy + dG_ddely * ddely_dy) * tan_fovy);

							// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
							atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
							atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
							atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

							// Update gradients w.r.t. opacity of the Gaussian
							atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
						}
						else {
							cur_dL_dmean2D = {
								dL_dG * (dG_ddelx * ddelx_dx + dG_ddely * ddely_dx) * tan_fovx,
								dL_dG * (dG_ddelx * ddelx_dy + dG_ddely * ddely_dy) * tan_fovy
							};
							cur_dL_dconic2D_dopacity = {
								-0.5f * gdx * d.x * dL_dG,
								-0.5f * gdx * d.y * dL_dG,
								G * dL_dalpha,
								-0.5f * gdy * d.y * dL_dG
							};
						}
					}
				}
			}

			if (!use_atomic) {
				// Perform warp-level reduction
				#pragma unroll
				for (int offset = 32/2; offset > 0; offset /= 2) {
					#pragma unroll
					for (int ch = 0; ch < C; ch++)
						cur_dL_dcolors[ch] += __shfl_down_sync(0xFFFFFFFF, cur_dL_dcolors[ch], offset);
					cur_dL_dmean2D.x += __shfl_down_sync(0xFFFFFFFF, cur_dL_dmean2D.x, offset);
					cur_dL_dmean2D.y += __shfl_down_sync(0xFFFFFFFF, cur_dL_dmean2D.y, offset);
					cur_dL_dconic2D_dopacity.x += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.x, offset);
					cur_dL_dconic2D_dopacity.y += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.y, offset);
					cur_dL_dconic2D_dopacity.z += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.z, offset);
					cur_dL_dconic2D_dopacity.w += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.w, offset);
				}

				// Store the results in shared memory
				if (block.thread_rank() % WARP_SIZE == 0)
				{
					int warp_id = block.thread_rank() / WARP_SIZE;
					batch_j[cur_reduction_batch_idx] = j;
					#pragma unroll
					for (int ch = 0; ch < C; ch++)
						batch_dL_dcolors[cur_reduction_batch_idx][warp_id][ch] = cur_dL_dcolors[ch];
					batch_dL_dmean2D[cur_reduction_batch_idx][warp_id] = cur_dL_dmean2D;
					batch_dL_dconic2D_dopacity[cur_reduction_batch_idx][warp_id] = cur_dL_dconic2D_dopacity;
				}
				cur_reduction_batch_idx += 1;
			}

			// If this is the last Gaussian in the batch, perform block-level
			// reduction and store the results in global memory.
			if (cur_reduction_batch_idx == REDUCTION_BATCH_SIZE || (j == min(BLOCK_SIZE, toDo) - 1 && cur_reduction_batch_idx != 0))
			{
				// Make sure we can perform this reduction with one warp
				static_assert(NUM_WARPS <= WARP_SIZE);
				// Make sure the number of warps is a power of 2
				static_assert((NUM_WARPS & (NUM_WARPS - 1)) == 0);

				// Wait for all warps to finish storing
				block.sync();

				for (int batch_id = block.thread_rank() / WARP_SIZE; batch_id < cur_reduction_batch_idx; batch_id += NUM_WARPS) {
					int lane_id = block.thread_rank() % WARP_SIZE;

					// Perform warp-level reduction
					#pragma unroll
					for (int ch = 0; ch < C; ch++)
						cur_dL_dcolors[ch] = lane_id < NUM_WARPS ? batch_dL_dcolors[batch_id][lane_id][ch] : 0,
					cur_dL_dmean2D = lane_id < NUM_WARPS ? batch_dL_dmean2D[batch_id][lane_id] : float2{0, 0},
					cur_dL_dconic2D_dopacity = lane_id < NUM_WARPS ? batch_dL_dconic2D_dopacity[batch_id][lane_id] : float4{0, 0, 0, 0};

					#pragma unroll
					for (int offset = NUM_WARPS/2; offset > 0; offset /= 2) {
						#pragma unroll
						for (int ch = 0; ch < C; ch++)
							cur_dL_dcolors[ch] += __shfl_down_sync(0xFFFFFFFF, cur_dL_dcolors[ch], offset);
						cur_dL_dmean2D.x += __shfl_down_sync(0xFFFFFFFF, cur_dL_dmean2D.x, offset);
						cur_dL_dmean2D.y += __shfl_down_sync(0xFFFFFFFF, cur_dL_dmean2D.y, offset);
						cur_dL_dconic2D_dopacity.x += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.x, offset);
						cur_dL_dconic2D_dopacity.y += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.y, offset);
						cur_dL_dconic2D_dopacity.z += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.z, offset);
						cur_dL_dconic2D_dopacity.w += __shfl_down_sync(0xFFFFFFFF, cur_dL_dconic2D_dopacity.w, offset);
					}

					// Store the results in global memory
					if (lane_id == 0)
					{
						const int global_id = collected_id[batch_j[batch_id]];
						// if (global_id < 0 || global_id >= 208424)
							// printf("%d\n", global_id);
						#pragma unroll
						for (int ch = 0; ch < C; ch++)
							atomicAdd(&dL_dcolors[global_id * C + ch], cur_dL_dcolors[ch]);
						atomicAdd(&dL_dmean2D[global_id].x, cur_dL_dmean2D.x);
						atomicAdd(&dL_dmean2D[global_id].y, cur_dL_dmean2D.y);
						atomicAdd(&dL_dconic2D[global_id].x, cur_dL_dconic2D_dopacity.x);
						atomicAdd(&dL_dconic2D[global_id].y, cur_dL_dconic2D_dopacity.y);
						atomicAdd(&dL_dconic2D[global_id].w, cur_dL_dconic2D_dopacity.w);
						atomicAdd(&dL_dopacity[global_id], cur_dL_dconic2D_dopacity.z);
					}
				}

				// Wait for all warps to finish reducing
				if (j != min(BLOCK_SIZE, toDo) - 1)
					block.sync();

				cur_reduction_batch_idx = 0;
			}
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHAFFELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int R, int B,
	const uint32_t* per_bucket_tile_offset,
	const uint32_t* bucket_to_tile,
	const float* sampled_T, const float* sampled_ar,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const uint32_t* max_contrib,
	const float* pixel_colors,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float focal_x, float focal_y)
{
	renderCUDA<NUM_CHAFFELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		focal_x, focal_y
		);
	// const int THREADS = 32;
	// PerGaussianRenderCUDA<NUM_CHAFFELS> <<<((B*32) + THREADS - 1) / THREADS,THREADS>>>(
	// 	ranges,
	// 	point_list,
	// 	W, H, B,
	// 	per_bucket_tile_offset,
	// 	bucket_to_tile,
	// 	sampled_T, sampled_ar,
	// 	bg_color,
	// 	means2D,
	// 	conic_opacity,
	// 	colors,
	// 	final_Ts,
	// 	n_contrib,
	// 	max_contrib,
	// 	pixel_colors,
	// 	dL_dpixels,
	// 	dL_dmean2D,
	// 	dL_dconic2D,
	// 	dL_dopacity,
	// 	dL_dcolors,
	// 	focal_x, focal_y
	// 	);
}
// bf16gemm_neon_m8_nld.S -- standalone NEON M=8 BF16 no-C-load entry points.
//
// ABI matches bf16gemm_sve_m8_nld.S:
//   x0 A, x1 packed B, x2 C, x3 reordered A, x4 params
//   x5 bias for bf16gemm_k_nld_bias_f_m8
//
// Assumptions:
//   - M is a multiple of 8
//   - K is a multiple of 8
//   - N is a multiple of 8
//   - no tail handling

#define bf16gemm_k_ld       bf16gemm_neon_unused_ld
#define bf16gemm_k_ld_b     bf16gemm_neon_unused_ld_b
#define bf16gemm_k_ld1      bf16gemm_neon_unused_ld1
#define bf16gemm_k_ld2      bf16gemm_neon_unused_ld2
#define bf16gemm_k_ld4      bf16gemm_neon_unused_ld4
#define bf16gemm_k_ld1_b    bf16gemm_neon_unused_ld1_b
#define bf16gemm_k_ld2_b    bf16gemm_neon_unused_ld2_b
#define bf16gemm_k_ld4_b    bf16gemm_neon_unused_ld4_b
#define bf16gemm_k_nld_b    bf16gemm_k_nld_b_m8
#define bf16gemm_k_nld1_b   bf16gemm_neon_unused_nld1_b
#define bf16gemm_k_nld2_b   bf16gemm_neon_unused_nld2_b
#define bf16gemm_k_nld4_b   bf16gemm_neon_unused_nld4_b
#include "bf16gemm_k.S"

GEMM_BODY bf16gemm_k_nld_f_m8, 8, 8, LOAD_ZERO_8, STORE_C_ROWMAJOR, 2, 32

#undef bf16gemm_k_ld
#undef bf16gemm_k_ld_b
#undef bf16gemm_k_ld1
#undef bf16gemm_k_ld2
#undef bf16gemm_k_ld4
#undef bf16gemm_k_ld1_b
#undef bf16gemm_k_ld2_b
#undef bf16gemm_k_ld4_b
#undef bf16gemm_k_nld_b
#undef bf16gemm_k_nld1_b
#undef bf16gemm_k_nld2_b
#undef bf16gemm_k_nld4_b

	.purgem M_LOOP_BEGIN_1
	.purgem M_LOOP_BEGIN_2
	.purgem M_LOOP_BEGIN_4
	.purgem M_LOOP_BEGIN_8
	.purgem M_LOOP_END_1
	.purgem M_LOOP_END_2
	.purgem M_LOOP_END_4
	.purgem M_LOOP_END_8
	.purgem INIT
	.purgem RESTORE
	.purgem LOAD_ZERO_8
	.purgem LOAD_ZERO_2
	.purgem LOAD_ZERO_1
	.purgem LOAD_ZERO_4
	.purgem LOAD_A0_B0_8
	.purgem COMPUTE_A1_B1_8
	.purgem COMPUTE_A2_B2_8
	.purgem COMPUTE_A3_B3_8
	.purgem LOAD_A0_B0_2
	.purgem COMPUTE_A1_B1_2
	.purgem COMPUTE_A2_B2_2
	.purgem COMPUTE_A3_B3_2
	.purgem LOAD_A0_B0_4
	.purgem COMPUTE_A1_B1_4
	.purgem COMPUTE_A2_B2_4
	.purgem COMPUTE_A3_B3_4
	.purgem LOAD_A0_B0_STRIDED_8
	.purgem COMPUTE_A1_B1_STRIDED_8
	.purgem COMPUTE_A2_B2_STRIDED_8
	.purgem LOAD_A0_B0_STRIDED_1
	.purgem COMPUTE_A1_B1_STRIDED_1
	.purgem COMPUTE_A2_B2_STRIDED_1
	.purgem LOAD_A0_B0_STRIDED_2
	.purgem COMPUTE_A1_B1_STRIDED_2
	.purgem COMPUTE_A2_B2_STRIDED_2
	.purgem LOAD_A0_B0_STRIDED_4
	.purgem COMPUTE_A1_B1_STRIDED_4
	.purgem COMPUTE_A2_B2_STRIDED_4

#define bf16gemm_k_ld_bias_f  bf16gemm_k_nld_bias_f_m8
#define bf16gemm_k_ld1_bias_f bf16gemm_neon_unused_ld1_bias_f
#define bf16gemm_k_ld2_bias_f bf16gemm_neon_unused_ld2_bias_f
#define bf16gemm_k_ld4_bias_f bf16gemm_neon_unused_ld4_bias_f
#include "bf16gemm_k_bias.S"

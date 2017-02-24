/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/
//#define	complex_mul(a,b) (complex) (a.x*b.x - a.y*b.y,a.x*b.y + a.y*b*x)
typedef float2 complex;

complex	twiddle(uint k, float angle) {
	float x; float y = sincos(k*angle, &x); 	return (complex)(x, y);
}

complex complex_mul(float2 a, float2 b) {
	return (complex)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);  
}

__kernel void FFT_2( __global complex* input, __global complex* output, uint Ns ) {

	uint work_item_id = get_global_id(0);
	uint num_work_items = get_global_size(0);

	complex in0, in1;
	in0 = input[(0 * num_work_items) + work_item_id];
	in1 = input[(1 * num_work_items) + work_item_id];

	if (Ns != 1)
	{
		float angle = -2 * M_PI*(work_item_id % Ns) / (Ns * 2);
		in1 = complex_mul(in1,twiddle(1, angle));
	}

	complex  tmp;
	tmp = in0;
	in0 = in0 + in1;
	in1 = tmp - in1;

	uint Idout = (work_item_id / Ns)*Ns * 2 + (work_item_id%Ns);
	output[(0 * Ns) + Idout] = in0;
	output[(1 * Ns) + Idout] = in1;
}
__kernel void FFT_4( __global complex* input, __global complex* output, uint Ns )
{
	uint work_item_id = get_global_id(0);
	uint num_work_items = get_global_size(0);

	complex  in0, in1, in2, in3;
	in0 = input[(0 * num_work_items) + work_item_id];
	in1 = input[(1 * num_work_items) + work_item_id];
	in2 = input[(2 * num_work_items) + work_item_id];
	in3 = input[(3 * num_work_items) + work_item_id];

	if (Ns != 1)
	{
		float angle = -2 * M_PI*(work_item_id % Ns) / (Ns * 2);
		in1 = complex_mul(in1, twiddle(1, angle));
		in2 = complex_mul(in2, twiddle(2, angle));
		in3 = complex_mul(in3, twiddle(3, angle));

	}

	complex v0, v1, v2, v3;
	v0 = in0 + in2;
	v2 = in0 - in2;
	v1 = in1 + in3;
	v3 = (complex)(in1.y - in3.y, in3.x - in1.x);
	in0 = v0 + v1;
	in2 = v0 - v1;
	in1 = v2 + v3;
	in3 = v2 - v3;

	uint Idout = (work_item_id / Ns)*Ns * 4 + (work_item_id%Ns);
	output[(0 * Ns) + Idout] = in0;
	output[(1 * Ns) + Idout] = in1;
	output[(2 * Ns) + Idout] = in2;
	output[(3 * Ns) + Idout] = in3;
}

__kernel void FFT_8(__global complex* input, __global complex* output, uint Ns)
{
	uint work_item_id = get_global_id(0);
	uint num_work_items = get_global_size(0);

	complex  in0, in1, in2, in3, in4, in5, in6, in7;
	in0 = input[(0 * num_work_items) + work_item_id];
	in1 = input[(1 * num_work_items) + work_item_id];
	in2 = input[(2 * num_work_items) + work_item_id];
	in3 = input[(3 * num_work_items) + work_item_id];
	in4 = input[(4 * num_work_items) + work_item_id];
	in5 = input[(5 * num_work_items) + work_item_id];
	in6 = input[(6 * num_work_items) + work_item_id];
	in7 = input[(7 * num_work_items) + work_item_id];

	if (Ns != 1)
	{
		float angle = -2 * M_PI*(work_item_id % Ns) / (Ns * 2);
		in1 = complex_mul(in1, twiddle(1, angle));
		in2 = complex_mul(in2, twiddle(2, angle));
		in3 = complex_mul(in3, twiddle(3, angle));
		in4 = complex_mul(in4, twiddle(4, angle));
		in5 = complex_mul(in5, twiddle(5, angle));
		in6 = complex_mul(in6, twiddle(6, angle));
		in7 = complex_mul(in7, twiddle(7, angle));
	}

	complex  v0, v1, v2, v3, v4, v5, v6, v7;
	v0 = in0 + in4;
	v4 = in0 - in4;
	v1 = in1 + in5;
	v5 = in1 - in5;
	v2 = in2 + in6;
	v6 = (complex)(in2.y - in6.y, in6.x - in2.x);
	v3 = in3 + in7;
	v7 = (complex)(in3.y - in7.y, in7.x - in3.x);

	complex tmp;
	tmp = v0;
	v0 = v0 + v2;
	v2 = tmp - v2;
	tmp = v1;
	v1 = v1 + v3;
	v3 = (complex)(tmp.y - v3.y, v3.x - tmp.x);

	tmp = v4;
	v4 = v4 + v6;
	v6 = tmp - v6;
	tmp = v5;
	v5 = (complex)(M_SQRT1_2* (v5.x + v7.x + v5.y + v7.y), M_SQRT1_2*(v5.y + v7.y - v5.x - v7.x));
	v7 = (complex)(M_SQRT1_2* (v7.x - tmp.x - v7.y + tmp.y), M_SQRT1_2*(v7.y - tmp.y + v7.x - tmp.x));

	in0 = v0 + v1;
	in4 = v0 - v1;

	in1 = v4 + v5;
	in5 = v4 - v5;

	in2 = v2 + v3;
	in6 = v2 - v3;

	in3 = v6 + v7;
	in7 = v6 - v7;

	uint Idout = (work_item_id / Ns)*Ns * 8 + (work_item_id%Ns);
	output[(0 * Ns) + Idout] = in0;
	output[(1 * Ns) + Idout] = in1;
	output[(2 * Ns) + Idout] = in2;
	output[(3 * Ns) + Idout] = in3;
	output[(4 * Ns) + Idout] = in4;
	output[(5 * Ns) + Idout] = in5;
	output[(6 * Ns) + Idout] = in6;
	output[(7 * Ns) + Idout] = in7;
}


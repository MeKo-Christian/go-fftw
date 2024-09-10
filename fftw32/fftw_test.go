package fftw32

import (
	"math"
	"testing"
)

func TestNewArray(t *testing.T) {
	t.Parallel()
	for _, n := range []int{10, 100, 1000} {
		d := NewArray(n)
		if len(d.Elems) != n {
			t.Errorf("Expected %d elements got %d", n, len(d.Elems))
		}
	}
}

// Make sure that the memory allocated by fftw is getting properly GCed
func TestGC(t *testing.T) {
	t.Parallel()
	var tot float32 = 0.0
	for i := range 1000 {
		d := NewArray(1000000)                  // Allocate a bunch of memory
		d.Elems[10000] = complex(float32(i), 0) // Do something stupid with it so
		tot += real(d.Elems[10000])             // hopefully it doesn't get optimized out
	}
}

func TestNewArray2(t *testing.T) {
	t.Parallel()
	d100x50 := NewArray2(100, 50)
	n0, n1 := d100x50.Dims()
	if n0 != 100 {
		t.Fatalf("Expected dim[0] = 100, got %d", n0)
	}
	if n1 != 50 {
		t.Fatalf("Expected dim[1] = 50, got %d", n1)
	}
	var counter float32
	for i := range n0 {
		for j := range n1 {
			d100x50.Set(i, j, complex(counter, 0))
			counter += 1.0
		}
	}
	counter = 0.0
	for i := range n0 {
		for j := range n1 {
			if v := real(d100x50.At(i, j)); v != counter {
				t.Fatalf("Expected real(%d,%d) = %f, got %f", i, j, counter, v)
			}
			counter += 1.0
		}
	}
}

func TestNewArray3(t *testing.T) {
	t.Parallel()
	d100x20x10 := NewArray3(100, 20, 10)
	n0, n1, n2 := d100x20x10.Dims()
	if n0 != 100 {
		t.Fatalf("Expected dim[0] = 100, got %d", n0)
	}
	if n1 != 20 {
		t.Fatalf("Expected dim[1] = 20, got %d", n1)
	}
	if n2 != 10 {
		t.Fatalf("Expected dim[2] = 10, got %d", n1)
	}
	var counter float32 = 0.0
	for i := range n0 {
		for j := range n1 {
			for k := range n2 {
				d100x20x10.Set(i, j, k, complex(counter, 0))
				counter += 1.0
			}
		}
	}
	counter = 0.0
	for i := range n0 {
		for j := range n1 {
			for k := range n2 {
				if v := real(d100x20x10.At(i, j, k)); v != counter {
					t.Fatalf("Expected real(%d,%d,%d) = %f, got %f", i, j, k, counter, v)
				}
				counter += 1.0
			}
		}
	}
}

func peakVerifier(t *testing.T, s []complex64) {
	t.Helper()
	testAlmostEqual(t, real(s[0]), 0.0)
	testAlmostEqual(t, imag(s[0]), 0.0)
	testAlmostEqual(t, real(s[1]), float32(len(s))/2)
	testAlmostEqual(t, imag(s[1]), 0.0)
	for i := 2; i < len(s)-1; i++ {
		testAlmostEqual(t, real(s[i]), 0.0)
		testAlmostEqual(t, imag(s[i]), 0.0)
	}
	testAlmostEqual(t, real(s[len(s)-1]), float32(len(s))/2)
	testAlmostEqual(t, imag(s[len(s)-1]), 0.0)
}

func TestFFT(t *testing.T) {
	t.Parallel()
	signal := NewArray(16)
	newIn := NewArray(16)
	for i := range signal.Elems {
		signal.Elems[i] = complex(float32(i), float32(-i))
		newIn.Elems[i] = signal.Elems[i]
	}

	// A simple real cosine should result in transform with two spikes, one at S[1] and one at S[-1]
	// The spikes should be real and have amplitude equal to len(S)/2 (because fftw doesn't normalize)
	for i := range signal.Elems {
		signal.Elems[i] = complex(float32(math.Cos(float64(i)/float64(len(signal.Elems))*math.Pi*2)), 0)
		newIn.Elems[i] = signal.Elems[i]
	}
	NewPlan(signal, signal, Forward, Estimate).Execute().Destroy()
	peakVerifier(t, signal.Elems)
}

func TestFFT2(t *testing.T) {
	t.Parallel()
	signal := NewArray2(64, 8)
	n0, n1 := signal.Dims()
	for i := range n0 {
		for j := range n1 {
			signal.Set(i, j, complex(float32(i+j), float32(-i-j)))
		}
	}

	// As long as fx < dx/2 and fy < dy/2, where dx and dy are the lengths in each dimension,
	// there will be 2^n spikes, where n is the number of dimensions.  Each spike will be
	// real and have magnitude equal to dx*dy / 2^n
	dx := n0
	fx := float64(dx) / 4
	dy := n1
	fy := float64(dy) / 4
	for i := range n0 {
		for j := range n1 {
			cosx := math.Cos(float64(i) / float64(dx) * fx * math.Pi * 2)
			cosy := math.Cos(float64(j) / float64(dy) * fy * math.Pi * 2)
			signal.Set(i, j, complex(float32(cosx*cosy), 0))
		}
	}
	NewPlan2(signal, signal, Forward, Estimate).Execute().Destroy()
	for i := range n0 {
		for j := range n1 {
			if (i == int(fx) || i == dx-int(fx)) &&
				(j == int(fy) || j == dy-int(fy)) {
				testAlmostEqual(t, real(signal.At(i, j)), float32(dx*dy/4))
				testAlmostEqual(t, imag(signal.At(i, j)), 0.0)
			} else {
				testAlmostEqual(t, real(signal.At(i, j)), 0.0)
				testAlmostEqual(t, imag(signal.At(i, j)), 0.0)
			}
		}
	}
}

func TestFFT3(t *testing.T) {
	t.Parallel()
	signal := NewArray3(32, 16, 8)

	n0, n1, n2 := signal.Dims()
	for i := range n0 {
		for j := range n1 {
			for k := range n2 {
				signal.Set(i, j, k, complex(float32(i+j+k), float32(-i-j-k)))
			}
		}
	}

	// As long as fx < dx/2, fy < dy/2, and fz < dz/2, where dx,dy,dz  are the lengths in
	// each dimension, there will be 2^n spikes, where n is the number of dimensions.
	// Each spike will be real and have magnitude equal to dx*dy*dz / 2^n
	dx := n0
	fx := float64(dx) / 4
	dy := n1
	fy := float64(dy) / 4
	dz := n2
	fz := float64(dz) / 4
	for i := range n0 {
		for j := range n1 {
			for k := range n2 {
				cosx := math.Cos(float64(i) / float64(dx) * fx * math.Pi * 2)
				cosy := math.Cos(float64(j) / float64(dy) * fy * math.Pi * 2)
				cosz := math.Cos(float64(k) / float64(dz) * fz * math.Pi * 2)
				signal.Set(i, j, k, complex(float32(cosx*cosy*cosz), 0))
			}
		}
	}
	NewPlan3(signal, signal, Forward, Estimate).Execute().Destroy()
	for i := range n0 {
		for j := range n1 {
			for k := range n2 {
				if (i == int(fx) || i == dx-int(fx)) &&
					(j == int(fy) || j == dy-int(fy)) &&
					(k == int(fz) || k == dz-int(fz)) {
					testAlmostEqual(t, real(signal.At(i, j, k)), float32(dx*dy*dz/8))
					testAlmostEqual(t, imag(signal.At(i, j, k)), 0.0)
				} else {
					testAlmostEqual(t, real(signal.At(i, j, k)), 0.0)
					testAlmostEqual(t, imag(signal.At(i, j, k)), 0.0)
				}
			}
		}
	}
}

const almostEqualEpsilon = 0.000001

func almostEqual(v1, v2 float32) bool {
	return math.Abs(float64(v1-v2)) < almostEqualEpsilon
}

func testAlmostEqual(t *testing.T, v1, v2 float32) {
	t.Helper()
	if !almostEqual(v1, v2) {
		t.Fatalf("%f != %f (delta %f)", v1, v2, math.Abs(float64(v1-v2)))
	}
}

package fftw32

import "testing"

func expectPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic: %s", name)
		}
	}()
	f()
}

func TestNewPlanGuards(t *testing.T) {
	t.Parallel()

	var nilArray *Array
	expectPanic(t, "nil input", func() {
		NewPlan(nilArray, NewArray(1), Forward, Estimate)
	})
	expectPanic(t, "nil output", func() {
		NewPlan(NewArray(1), nilArray, Forward, Estimate)
	})
	expectPanic(t, "empty input", func() {
		NewPlan(NewArray(0), NewArray(0), Forward, Estimate)
	})
}

func TestNewPlan2Guards(t *testing.T) {
	t.Parallel()

	var nilArray *Array2
	expectPanic(t, "nil input", func() {
		NewPlan2(nilArray, NewArray2(1, 1), Forward, Estimate)
	})
	expectPanic(t, "nil output", func() {
		NewPlan2(NewArray2(1, 1), nilArray, Forward, Estimate)
	})
	expectPanic(t, "empty input", func() {
		NewPlan2(NewArray2(0, 1), NewArray2(0, 1), Forward, Estimate)
	})
}

func TestNewPlan3Guards(t *testing.T) {
	t.Parallel()

	var nilArray *Array3
	expectPanic(t, "nil input", func() {
		NewPlan3(nilArray, NewArray3(1, 1, 1), Forward, Estimate)
	})
	expectPanic(t, "nil output", func() {
		NewPlan3(NewArray3(1, 1, 1), nilArray, Forward, Estimate)
	})
	expectPanic(t, "empty input", func() {
		NewPlan3(NewArray3(1, 0, 1), NewArray3(1, 0, 1), Forward, Estimate)
	})
}

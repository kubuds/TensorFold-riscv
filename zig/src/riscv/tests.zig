//! tests.zig - the RISC-V backend's test root: kernel families against the
//! float64 reference, and the Qwen3 loader/forward smoke checks.
test {
    _ = @import("kernel_test.zig");
    _ = @import("model_test.zig");
}

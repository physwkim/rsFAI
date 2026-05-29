//! Verify `rsfai-io` (pure-Rust `rust-hdf5`) reads an h5py/libhdf5-written
//! NeXus file **bit-exactly** across the rsFAI dtypes. Golden produced by
//! `golden/gen_golden_io.py` (h5py 3.16, libhdf5 2.0): `frame.h5` plus the
//! `.npy` of every dataset.
//!
//! HDF5 stores f32/f64/i32 as raw IEEE / two's-complement bytes, so a correct
//! reader returns exactly what h5py wrote — the gate is bitwise. One dataset is
//! gzip-compressed + chunked, exercising `rust-hdf5`'s filter pipeline.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_npy_f32, load_npy_f64, load_npy_i32};
use rsfai_io::{read_dataset_f32, read_dataset_f64, read_dataset_i32};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_io")
}

fn golden_f32(name: &str) -> Vec<f32> {
    load_npy_f32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn golden_f64(name: &str) -> Vec<f64> {
    load_npy_f64(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn golden_i32(name: &str) -> Vec<i32> {
    load_npy_i32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn report(name: &str, ok: bool, fails: &mut usize) {
    eprintln!("  {name:38} {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

#[test]
fn reads_h5py_nexus_file_bit_exact() {
    let h5 = root().join("frame.h5");
    let h5 = h5.as_path();
    let mut fails = 0usize;

    // f32 image (contiguous).
    let data = read_dataset_f32(h5, "entry/data/data").expect("read entry/data/data");
    let g_data = golden_f32("frame__data_f32.npy");
    report(
        "entry/data/data (f32, contiguous)",
        data.shape == [32, 32] && compare_f32(&data.data, &g_data).is_bit_exact(),
        &mut fails,
    );

    // Same image, gzip-compressed + chunked: must decode to identical bytes.
    let data_gz = read_dataset_f32(h5, "entry/data/data_gzip").expect("read entry/data/data_gzip");
    report(
        "entry/data/data_gzip (f32, gzip+chunked)",
        data_gz.shape == [32, 32] && compare_f32(&data_gz.data, &g_data).is_bit_exact(),
        &mut fails,
    );

    // i32 counts (contiguous) — exact integer equality.
    let counts = read_dataset_i32(h5, "entry/data/counts").expect("read entry/data/counts");
    let g_counts = golden_i32("frame__counts_i32.npy");
    report(
        "entry/data/counts (i32, contiguous)",
        counts.shape == [32, 32] && counts.data == g_counts,
        &mut fails,
    );

    // f64 positions (1-D).
    let pos = read_dataset_f64(h5, "entry/instrument/detector/positions")
        .expect("read entry/instrument/detector/positions");
    let g_pos = golden_f64("frame__positions_f64.npy");
    report(
        "detector/positions (f64, 1-D)",
        pos.shape == [64] && compare_f64(&pos.data, &g_pos).is_bit_exact(),
        &mut fails,
    );

    // dims2 helper on a 2-D frame.
    assert_eq!(data.dims2(), (32, 32), "dims2() on the f32 frame");

    assert_eq!(fails, 0, "{fails} dataset(s) mismatched the h5py golden");
}

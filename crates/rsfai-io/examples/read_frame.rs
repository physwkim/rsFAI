//! Load a detector frame (and counts / positions) from a NeXus HDF5 file with
//! `rsfai-io`, offline. Reads the committed `golden/datasets_io/frame.h5`
//! (written by h5py / libhdf5) and prints each dataset's shape + stats — the
//! practical "open a NeXus file and pull a frame" path.
//!
//! Run: `cargo run --release --example read_frame`

use std::path::PathBuf;

use rsfai_io::{read_dataset_f32, read_dataset_f64, read_dataset_i32};

fn h5_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_io/frame.h5")
}

fn f32_stats(v: &[f32]) -> (f32, f32, f64) {
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    (min, max, mean)
}

fn main() {
    let path = h5_path();
    let path = path.as_path();
    println!("reading NeXus file {}\n", path.display());

    let frame = read_dataset_f32(path, "entry/data/data").expect("read entry/data/data");
    let (rows, cols) = frame.dims2();
    let (min, max, mean) = f32_stats(&frame.data);
    println!("entry/data/data            f32  {rows}x{cols}  min={min}  max={max}  mean={mean:.4}");

    let gz = read_dataset_f32(path, "entry/data/data_gzip").expect("read gzip dataset");
    println!(
        "entry/data/data_gzip       f32  {:?}  (gzip+chunked, decoded {} elements)",
        gz.shape,
        gz.len()
    );

    let counts = read_dataset_i32(path, "entry/data/counts").expect("read counts");
    let csum: i64 = counts.data.iter().map(|&x| x as i64).sum();
    println!(
        "entry/data/counts          i32  {:?}  sum={csum}",
        counts.shape
    );

    let pos =
        read_dataset_f64(path, "entry/instrument/detector/positions").expect("read positions");
    println!(
        "instrument/detector/positions  f64  {:?}  first={}  last={}",
        pos.shape,
        pos.data.first().copied().unwrap_or(f64::NAN),
        pos.data.last().copied().unwrap_or(f64::NAN)
    );
}

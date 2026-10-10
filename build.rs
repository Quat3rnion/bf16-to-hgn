use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn main() {
    let root = env::var("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let root = Path::new(&root);
    let files = [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/main.rs",
        "src/hf.rs",
        "src/source.rs",
        "src/hgn_source.rs",
        "src/gguf_source.rs",
        "src/gguf_decode.rs",
        "src/gguf_iq3_grid.rs",
        "src/architecture.rs",
        "src/container.rs",
        "src/conversion.rs",
        "src/manifest.rs",
        "src/verify.rs",
        "src/codec.rs",
        "native/ht_trellis.cpp",
    ];
    let mut hash = Sha256::new();
    for name in files {
        println!("cargo:rerun-if-changed={name}");
        let bytes =
            fs::read(root.join(name)).unwrap_or_else(|error| panic!("read {name}: {error}"));
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    println!(
        "cargo:rustc-env=BF16_HGN_ENCODER_SHA256={:x}",
        hash.finalize()
    );
    cc::Build::new()
        .cpp(true)
        .file("native/ht_trellis.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-fopenmp")
        .flag("-ffp-contract=off")
        .compile("hgn_trellis");
    println!("cargo:rustc-link-lib=gomp");
}

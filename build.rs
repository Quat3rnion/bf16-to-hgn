use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

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
        "src/progress.rs",
        "src/verification_manifest.rs",
        "src/verify_gpu.rs",
        "src/codec.rs",
        "native/ht_trellis.cpp",
        "native/ht_trellis_hip.hip",
        "native/ht_trellis_cuda.cu",
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
    let hip = env::var_os("CARGO_FEATURE_HIP").is_some();
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let mut cpu = cc::Build::new();
    cpu.cpp(true)
        .file("native/ht_trellis.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-fopenmp")
        .flag("-ffp-contract=off");
    if hip {
        // The HIP selection order mirrors this exact standard-library version.
        // A different CPU nth_element implementation is not a byte oracle.
        let guard = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"))
            .join("hip_cpu_compatibility.h");
        fs::write(
            &guard,
            "#include <algorithm>\n\
             #if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 16\n\
             #error HIP byte equivalence requires the CPU encoder to use libstdc++ 16\n\
             #endif\n\
             #ifdef _GLIBCXX_PARALLEL\n\
             #error HIP byte equivalence requires ordinary libstdc++ nth_element\n\
             #endif\n\
             #ifdef __FAST_MATH__\n\
             #error HIP byte equivalence requires IEEE floating point without fast-math\n\
             #endif\n",
        )
        .expect("write HIP CPU compatibility guard");
        cpu.flag("-include")
            .flag(guard.to_str().expect("UTF-8 output path"));
    }
    if cuda {
        // CUDA's supported nvcc host toolchain can use libstdc++ 12; the
        // differential selection probe and cross-host fingerprints qualify
        // its permutation against the libstdc++ 16 CPU/HIP reference.
        // Keep the stricter HIP guard above when both features are enabled.
        let guard = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"))
            .join("cuda_cpu_compatibility.h");
        fs::write(
            &guard,
            "#include <algorithm>\n\
             #if !defined(_GLIBCXX_RELEASE) || (_GLIBCXX_RELEASE != 12 && _GLIBCXX_RELEASE != 16)\n\
             #error CUDA byte equivalence requires the CPU encoder to use libstdc++ 12 or 16\n\
             #endif\n\
             #ifdef _GLIBCXX_PARALLEL\n\
             #error CUDA byte equivalence requires ordinary libstdc++ nth_element\n\
             #endif\n\
             #ifdef __FAST_MATH__\n\
             #error CUDA byte equivalence requires IEEE floating point without fast-math\n\
             #endif\n",
        )
        .expect("write CUDA CPU compatibility guard");
        cpu.flag("-include")
            .flag(guard.to_str().expect("UTF-8 output path"));
    }
    cpu.compile("hgn_trellis");
    println!("cargo:rustc-link-lib=gomp");
    if hip {
        println!("cargo:rerun-if-env-changed=HIPCC");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        let rocm = env::var_os("ROCM_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/opt/rocm"));
        let compiler = env::var_os("HIPCC")
            .map(PathBuf::from)
            .unwrap_or_else(|| rocm.join("bin/hipcc"));
        cc::Build::new()
            .cpp(true)
            .compiler(compiler)
            .file("native/ht_trellis_hip.hip")
            .flag("-std=c++17")
            .flag("-O3")
            .flag("--offload-arch=gfx1151")
            .flag("-ffp-contract=off")
            .flag("-fno-fast-math")
            .flag("-fno-gpu-flush-denormals-to-zero")
            // I4R code selection requires an IEEE RNE binary32 quotient.
            .flag("-fhip-fp32-correctly-rounded-divide-sqrt")
            .compile("hgn_trellis_hip");
        println!(
            "cargo:rustc-link-search=native={}",
            rocm.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            rocm.join("lib64").display()
        );
        println!("cargo:rustc-link-lib=amdhip64");
    }
    if cuda {
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=HGN_CUDA_ARCH");
        let arch = env::var("HGN_CUDA_ARCH").unwrap_or_else(|_| "89".to_owned());
        let capability = arch
            .parse::<u32>()
            .expect("HGN_CUDA_ARCH must be a numeric compute capability, e.g. 89");
        assert!(
            (60..=999).contains(&capability) && !arch.starts_with('0'),
            "HGN_CUDA_ARCH must be a numeric compute capability >= 60, e.g. 89"
        );
        cc::Build::new()
            .cuda(true)
            .ccbin(true) // Use the same CXX as the CPU oracle; never bypass nvcc's guard.
            .cudart("static")
            .debug(false) // cc otherwise adds -G in Cargo's dev/test profiles.
            .opt_level(3)
            .file("native/ht_trellis_cuda.cu")
            .flag("-std=c++17")
            .flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
            .flag(format!("-gencode=arch=compute_{arch},code=compute_{arch}"))
            .flag("--fmad=false")
            .flag("--ftz=false")
            .flag("--prec-div=true")
            .flag("--prec-sqrt=true")
            .flag("-Xcompiler=-ffp-contract=off")
            .flag("-Xcompiler=-fno-fast-math")
            .compile("hgn_trellis_cuda");
    }
}

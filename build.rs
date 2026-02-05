fn main() {
    // Add Swift runtime library path for ScreenCaptureKit
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    if let Ok(output) = std::process::Command::new("xcode-select")
        .arg("--print-path")
        .output()
    {
        if let Ok(path) = String::from_utf8(output.stdout) {
            let toolchain_path = format!("{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx", path.trim());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", toolchain_path);
        }
    }
}
